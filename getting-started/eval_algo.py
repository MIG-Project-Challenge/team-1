from collections import defaultdict
import numpy as np
from pathlib import Path
import argparse

# TODO: incorporate using open prices for buying and closing prices for selling (or should we just use close price?)


def calc_sharpe_ratio(portfolio_values):
    # returns annualized sharpe ratio

    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    risk_free_rate = 0
    average_daily_return = np.mean(daily_returns)
    volatility = np.std(daily_returns)

    # assuming 252 trading days in a year
    sharpe_ratio = np.sqrt(252) * (average_daily_return - risk_free_rate) / volatility

    return sharpe_ratio


def eval_actions(actions, prices, cash=10, verbose=True):
    cash = cash
    positions = [0] * len(actions)
    port_values = [0] * len(actions[0])

    # shorts positions are closed in FIFO ordering
    # ticker -> dequeue(short price, short amount)
    short_positions = defaultdict(list)

    def cashValid():
        return cash >= 0

    def calcShortValue(day):
        # calculates value of all the short positions on 'day'
        # note that this value can be positive or negative
        val = 0
        for stock in short_positions.keys():
            for short_price, short_amount in short_positions[stock]:
                val += (short_price - prices[stock][day]) * short_amount

        return val

    def calcPortfolioValue(day):
        # the sum of cash and positive positions (and their value on 'day') + short positions value
        # cash + long positions + short positions

        value = cash
        for stock in range(len(actions)):
            if positions[stock] > 0:
                value += prices[stock][day] * positions[stock]
        return value + calcShortValue(day)

    # --- main logic ---

    for day in range(len(actions[0])):
        port_values[day] = calcPortfolioValue(day)

        if port_values[day] < 0:
            # TODO: return portfolio here
            print("DEBT LIMIT HIT (likely too many short positions)")
            return 0, 0

        for stock in range(len(actions)):
            # case 1: we have a positive position and are buying
            # or don't have a position yet
            if positions[stock] >= 0 and actions[stock][day] > 0:
                cash -= prices[stock][day] * actions[stock][day]

                if not cashValid():
                    # TODO: return portfolio here
                    print("INVALID CASH AMOUNT, COULD NOT AFFORD TRANSACTION")
                    return 0, 0

                positions[stock] += actions[stock][day]

            # case 2: we have short a position and are buying
            elif positions[stock] < 0 and actions[stock][day] > 0:
                buy_amount = min(actions[stock][day] - abs(positions[stock]), 0)
                short_close_amount = min(abs(positions[stock]), actions[stock][day])

                while short_close_amount > 0:
                    # the amount of positions to close in the current short order
                    positions_to_close = min(
                        short_positions[stock][0][1], short_close_amount
                    )
                    short_close_amount -= positions_to_close

                    # the short value is = (short price - curr price)*amount of shares shorted
                    cash += (
                        short_positions[stock][0][0] - prices[stock][day]
                    ) * positions_to_close

                    if positions_to_close == short_positions[stock][0][1]:
                        short_positions[stock].pop(0)

                    else:
                        # subtract the amount of closed shares from the given price
                        short_positions[0][1] -= positions_to_close

                if buy_amount > 0:
                    cash -= prices[stock][day] * buy_amount
                    positions[stock] += buy_amount

            # case 3: we have a positive position and are selling/shorting
            elif positions[stock] > 0 and actions[stock][day] < 0:
                sell_amount = min(abs(actions[stock][day]), positions[stock])
                short_amount = max(abs(actions[stock][day]) - positions[stock], 0)

                cash += prices[stock][day] * sell_amount
                positions[stock] -= sell_amount

                if short_amount > 0:
                    short_price_amount = [prices[stock][day], short_amount]
                    short_positions[stock].append(short_price_amount)

                    positions[stock] -= short_amount

            # case 4: we have a short position and are selling/shorting
            # or don't have any position yet
            elif positions[stock] <= 0 and actions[stock][day] < 0:
                short_amount = abs(actions[stock][day])

                short_price_amount = [prices[stock][day], short_amount]
                short_positions[stock].append(short_price_amount)

                positions[stock] -= short_amount

    if verbose:
        print("final portfolio value:", port_values[-1])
        print("cash:", cash)
        print("positions:", positions)
        print("short position info:", short_positions)
        print("short value:", calcShortValue(len(actions[0]) - 1))

    return port_values, calc_sharpe_ratio(port_values)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate actions")
    parser.add_argument(
        "-a",
        "--actions",
        help="path to actions matrix file (should be .npy file)",
    )
    parser.add_argument(
        "-p",
        "--prices",
        help="path to stock prices matrix file (should be .npy file)",
    )

    args = parser.parse_args()

    prices = np.load(args.prices)
    actions = np.load(args.actions)

    assert prices.size() == actions.size(), "prices and actions must be the same size"

    print(eval_actions(actions, prices))
