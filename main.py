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

        training_data_slices = []
        testing_data_slices = []

        for ticker, group in self.original_data.groupby(level='Ticker'):
            slice_length = int(len(group) * 0.7)
            slice_70_percent = group.iloc[:slice_length]
            slice_30_percent = group.iloc[slice_length:]
            training_data_slices.append(slice_70_percent)
            testing_data_slices.append(slice_30_percent)

        self.training_data = pd.concat(training_data_slices)
        self.testing_data = pd.concat(testing_data_slices)
        self.open_prices = self.testing_data['Open'].unstack(level='Ticker')

        self.training_returns = self.returns.iloc[:int(len(self.returns) * 0.7)]
        self.testing_returns = self.returns.iloc[int(len(self.returns) * 0.7):]

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
    def max_sharpe(ret: pd.DataFrame):
        def sharpe_func(weights):
            hist_mean = ret.mean(axis=0).to_frame()
            hist_cov = ret.cov()

            port_ret = np.dot(weights.T, hist_mean.values) * 252
            port_std = np.sqrt(np.dot(weights.T, np.dot(hist_cov, weights)) * 252)
            return -1 * port_ret / port_std

        def weight_cons(weights):
            return np.sum(weights) - 1

        bounds_lim = [(0, 1) for x in range(len(ret.columns))]
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
    def minimum_variance(ret: pd.DataFrame):
        def find_port_variance(weights):
            # this is actually std
            cov = ret.cov()
            port_var = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) * 252)
            return port_var

        def weight_cons(weights):
            return np.sum(weights) - 1

        bounds_lim = [(0, 1) for x in range(len(ret.columns))]
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

    def run_port_opt(self):
        for index, row in enumerate(self.testing_returns.iterrows()):
            if not index % 20:
                optimized_weights = self.max_sharpe(self.training_returns.iloc[index:])
                actions = self.exit_positions(index)
                for index2, stock in enumerate(self.returns.columns):
                    cash_to_allocate = self.cash * optimized_weights[stock]
                    curr_price = self.open_prices[stock].iloc[index: index + 1].values[0]
                    num_shares = 0
                    if cash_to_allocate != 0:
                        num_shares = int(math.floor(cash_to_allocate / curr_price))
                    if num_shares != 0 and num_shares * curr_price < self.cash:
                        self.open_trades.append(Trade(stock=stock, price=curr_price, num_shares=num_shares))
                        self.cash -= (num_shares * curr_price)
                        self.assets += (num_shares * curr_price)
                    actions[index2] += num_shares
                    self.positions[index2] += num_shares

                self.actions = pd.concat([self.actions, pd.Series(index=self.testing_returns.columns,
                                                                  data=actions,
                                                                  name=row[0])], axis=1)
                self.training_returns = pd.concat([self.training_returns, self.testing_returns.iloc[index: index + 1]])
                continue

            self.actions = pd.concat([self.actions, pd.Series(index=self.testing_returns.columns,
                                                              data=len(self.testing_returns.columns) * [0],
                                                              name=row[0])], axis=1)
            self.training_returns = pd.concat([self.training_returns, self.testing_returns.iloc[index: index + 1]])

    def exit_positions(self, index):
        actions = [0] * len(self.positions)
        win_rate = 0
        for trade in self.open_trades:
            stock_idx = list(self.testing_returns.columns).index(trade.stock)
            self.positions[stock_idx] -= trade.num_shares
            actions[stock_idx] -= trade.num_shares

            curr_price = self.open_prices[trade.stock].iloc[index: index + 1].values[0]
            if (curr_price > trade.price):
                win_rate += 1
            self.cash += curr_price * trade.num_shares
            self.assets -= trade.num_shares * trade.price
        if len(self.open_trades):
            print(win_rate / len(self.open_trades))
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
