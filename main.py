import warnings
import pandas as pd
import numpy as np
import math

from pydantic import BaseModel
from tqdm.auto import tqdm
from scipy.optimize import minimize


class Trade(BaseModel):
    stock: str
    price: float
    num_shares: int


class Algo:
    def __init__(self, data_path: str,
                 cash: int = 25000,
                 stop_loss: float = 0.0):
        self.original_data = pd.read_csv(data_path)
        self.original_data['Date'] = pd.to_datetime(self.original_data['Date'])
        self.original_data.set_index(['Ticker', 'Date'], inplace=True)
        self.original_data['Ret'] = self.original_data['Open'].groupby('Ticker').pct_change()
        self.original_data.dropna(inplace=True)
        self.returns = self.original_data['Ret'].unstack(level='Ticker')

        self.training_data = pd.read_csv('./getting-started/train_data_50.csv')
        self.training_data['Date'] = pd.to_datetime(self.training_data['Date'])
        self.training_data.set_index(['Ticker', 'Date'], inplace=True)
        self.training_returns = self.training_data['Open'].groupby('Ticker').pct_change()
        self.training_returns.dropna(inplace=True)

        self.testing_data = pd.concat(testing_data_slices)
        self.open_prices = self.testing_data['Open'].unstack(level='Ticker')

        self.testing_returns = self.returns

        self.stop_loss = stop_loss

        self.actions = pd.DataFrame()
        self.positions = [0] * len(self.returns.columns)
        self.daily_returns = pd.DataFrame()
        self.portfolio_value = pd.Series()
        self.open_trades: list[Trade] = []
        self.cash = cash
        self.assets = 0
        self.debt = 0

    @staticmethod
    def max_sharpe(ret: pd.DataFrame, short_only=False, long_only=False):
        def sharpe_func(weights):
            hist_mean = ret.mean(axis=0).to_frame()
            hist_cov = ret.cov()

            port_ret = np.dot(weights.T, hist_mean.values) * 252
            port_std = np.sqrt(np.dot(weights.T, np.dot(hist_cov, weights)) * 252)
            return -1 * port_ret / port_std

        bounds = (-1, 1)
        if short_only:
            bounds = (-1, 0)
        elif long_only:
            bounds = (0, 1)
        def weight_cons(weights):
            return np.sum(weights) - 1

        bounds_lim = [bounds for x in range(len(ret.columns))]
        init = [1 / len(ret.columns) for i in range(len(ret.columns))]
        constraint = {'type': 'eq', 'fun': weight_cons}

        optimal = minimize(fun=sharpe_func,
                           x0=init,
                           bounds=bounds_lim,
                           constraints=constraint,
                           method='SLSQP'
                           )

        optimal = [round(i, 10) for i in optimal['x']]
        return dict(zip(ret.columns, optimal))

    @staticmethod
    def minimum_variance(ret: pd.DataFrame, short_only=False, long_only=False):
        def find_port_variance(weights):
            # this is actually std
            cov = ret.cov()
            port_var = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) * 252)
            return port_var

        bounds = (-1, 1)
        if short_only:
            bounds = (-1, 0)
        elif long_only:
            bounds = (0, 1)

        def weight_cons(weights):
            return np.sum(weights) - 1

        bounds_lim = [bounds for x in range(len(ret.columns))]
        init = [1 / len(ret.columns) for i in range(len(ret.columns))]
        constraint = {'type': 'eq', 'fun': weight_cons}

        optimal = minimize(fun=find_port_variance,
                           x0=init,
                           bounds=bounds_lim,
                           constraints=constraint,
                           method='SLSQP'
                           )
        optimal = [round(i, 10) for i in optimal['x']]
        return dict(zip(ret.columns, optimal))
#2 port opps, 1 that only shorts 1 that only longs each w half of our cash, optimize both of those and execute our
    # trades
    def run_port_opt(self):
        # go through all testing data
        for index, row in enumerate(self.testing_returns.iterrows()):
            # if it's the beginning of our rebalancing period, rebalance
            if not index % 21:
                # get our weights
                optimized_weights_long = self.max_sharpe(self.training_returns.iloc[index:], long_only=True)
                optimized_weights_short = self.max_sharpe(self.training_returns.iloc[index:], short_only=True)
                optimized_weights = dict(zip(list(optimized_weights_long.keys()),
                                             (np.array(list(optimized_weights_short.values())) + np.array(list(optimized_weights_long.values())) / 2)))
                # exit positions completely (will all be 0s on first iteration, then the opposite of our positions afterwards)
                actions = self.exit_positions(index)
                # for each stock in our universe...
                for index2, stock in enumerate(self.returns.columns):
                    # figure out how much cash to allocate

                    cash_to_allocate = self.cash * optimized_weights[stock]
                    # get current price
                    curr_price = self.open_prices[stock].iloc[index: index + 1].values[0]

                    # determine number of shares
                    num_shares = 0
                    if cash_to_allocate != 0:
                        num_shares = int(math.floor(cash_to_allocate / curr_price))

                    # if we do want to buy, (need to figure out selling to) then create a trade, and manipulate our cash and assets
                    # TODO: fix short so that we manage our debt and stuff. might want to break into if num_shares > 0
                    #  and if num_shares < 0 for shorting
                    if num_shares > 0 and num_shares * curr_price < self.cash:
                        self.open_trades.append(Trade(stock=stock, price=curr_price, num_shares=num_shares))
                        self.cash -= (num_shares * curr_price)
                        self.assets += (num_shares * curr_price)
                    elif num_shares < 0 and abs(self.cash + self.assets) > self.debt + abs(cash_to_allocate):
                        short_value = abs(num_shares) * curr_price
                        self.open_trades.append(Trade(stock=stock, price=curr_price, num_shares=num_shares))
                        self.debt += short_value
                    else:
                        continue

                    # if num_shares != 0 and num_shares * curr_price < self.cash:
                    #     self.open_trades.append(Trade(stock=stock, price=curr_price, num_shares=num_shares))
                    #     self.cash -= (num_shares * curr_price)
                    #     self.assets += (num_shares * curr_price)

                    # anything we want to buy again add back to actions. if we sold all our positions,
                    # and bought them all back, equivalent to doing nothing today
                    actions[index2] += num_shares
                    # same with positions
                    self.positions[index2] += num_shares

                # append the days actions to actions
                self.actions = pd.concat([self.actions, pd.Series(index=self.testing_returns.columns,
                                                                  data=actions,
                                                                  name=row[0])], axis=1)
                # roll over training returns so we extend our training window for portopt
                self.training_returns = pd.concat([self.training_returns, self.testing_returns.iloc[index: index + 1]])
                continue

            # fill with 0s when we don't rebalance
            self.actions = pd.concat([self.actions, pd.Series(index=self.testing_returns.columns,
                                                              data=len(self.testing_returns.columns) * [0],
                                                              name=row[0])], axis=1)
            self.training_returns = pd.concat([self.training_returns, self.testing_returns.iloc[index: index + 1]])



    def exit_positions(self, index):
        # start with blank actions
        actions = [0] * len(self.positions)
        win_rate = 0
        # for each open trade...
        for trade in self.open_trades:
            # get the index of this stock in our positions list
            stock_idx = list(self.testing_returns.columns).index(trade.stock)
            # reset positions (right now this assumes we are always selling to exit, because we always long
            # if we are shorting, need to check sign of num_shares to see if we buy back or sell off.
            curr_price = self.open_prices[trade.stock].iloc[index: index + 1].values[0]
            # Check the sign of num_shares to determine if we are buying back or selling off

            if trade.num_shares > 0:
                if (curr_price > trade.price):
                    win_rate += 1
                # exiting a long position
                self.positions[stock_idx] -= trade.num_shares
                actions[stock_idx] -= trade.num_shares
                self.cash += curr_price * trade.num_shares
                self.assets -= trade.num_shares * trade.price
            elif trade.num_shares < 0:
                if (curr_price < trade.price):
                    win_rate += 1
                # exiting a short position
                self.positions[stock_idx] += abs(trade.num_shares)
                actions[stock_idx] += abs(trade.num_shares)
                self.cash += (trade.price - curr_price) * trade.num_shares
                self.debt -= abs(trade.num_shares) * trade.price

            # get current price
            # statistics calculation for how often we win


            # manipulate cash and assets. again this is assumign we are only longing.
            # if we are shorting too, this needs to be in if conditions

        if len(self.open_trades):
            print(win_rate / len(self.open_trades))
        # clear our open trades since we did everything we could with them
        self.open_trades.clear()
        return actions


if __name__ == '__main__':
    algo = Algo(data_path='./getting-started/train_data_50.csv')
    algo.run_port_opt()
    with open('actions.npy', 'wb') as f:
        np.save(f, algo.actions)
    with open('prices.npy', 'wb') as f:
        np.save(f, algo.open_prices.T)
    print(algo.cash)
