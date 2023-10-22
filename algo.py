import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller


class Trade(BaseModel):
    stock_a: str
    stock_b: str
    price_a: float
    num_a: int
    price_b: float
    num_b: int


class Algo:
    def __init__(self, data_path: Path, cash=25000, n_components=12, n_clusters=6, use_log_returns=False, even_weight_multiplier=10, stop_loss=2):
        warnings.simplefilter(action='ignore', category=Warning)

        self.original_data = pd.read_csv(data_path)
        self.original_data['Date'] = pd.to_datetime(self.original_data['Date'])
        self.original_data.set_index(['Ticker', 'Date'], inplace=True)

        self.use_log_returns = use_log_returns
        self.original_data['Ret'] = self.original_data['Open'].groupby('Ticker').pct_change()
        self.original_data.dropna(inplace=True)
        self.returns = self.original_data['Ret'].unstack(level='Ticker')
        if use_log_returns:
            self.original_data['Log Ret'] = np.log(
                self.original_data['Open'] / self.original_data['Open'].groupby('Ticker').shift(1))
            self.original_data.dropna(inplace=True)
            self.log_returns = self.original_data['Log Ret'].unstack(level='Ticker')

        self.n_components = n_components
        self.n_clusters = n_clusters
        self.even_weight_multiplier = even_weight_multiplier
        self.stop_loss = stop_loss

        self.actions = pd.DataFrame()
        self.positions = dict(zip(self.returns.columns, [0] * len(self.returns.columns)))
        self.open_trades: list[Trade] = []
        self.cash = cash
        self.assets = 0
        self.debt = 0

    def run_pairs(self):
        pairs = pd.DataFrame()
        filtered_pairs = {}
        curr_pairs = []
        look_back_window = 30
        day = look_back_window
        total = len(self.returns.iloc[look_back_window - 1:])
        for index, row in tqdm(enumerate(self.returns.iloc[look_back_window - 1:].iterrows()), total=total):
            if self.use_log_returns:
                curr_data = self.log_returns.iloc[day - look_back_window: day]
            else:
                curr_data = self.returns.iloc[day - look_back_window: day]
            stocks = list(curr_data.columns)
            # get pairs it's if the first of the month
            if day % look_back_window == 0:
                # TODO: need to finish up open trades here
                self.exit_positions(pairs, rebalance=True)
                pairs = pd.DataFrame()
                curr_pairs = []

                # scale data and run PCA
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(curr_data)
                pca = PCA(n_components=self.n_components)
                components = pca.fit(scaled_data).components_.T

                # run kmeans and get clusters
                kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=75, n_init=10, random_state=42)
                kmeans.fit(components)
                labels = kmeans.labels_

                # make a list of clusters to iterate through
                clusters = {}
                for stock, cluster in zip(stocks, labels):
                    if cluster not in clusters:
                        clusters[cluster] = []
                    clusters[cluster].append(stock)

                for cluster in clusters.values():
                    if len(cluster) > 3:
                        for i in range(len(cluster)):
                            for j in range(i + 1, len(cluster)):
                                curr_pairs.append((cluster[i], cluster[j]))

                curr_pairs = sorted(curr_pairs)

            initial = False
            if not index:
                initial = True

            filtered_pairs, pairs = self.update_data(curr_data=curr_data, curr_pairs=curr_pairs, initial_update=initial)

            # TODO: add stop loss logic to exit positions

            self.exit_positions(pairs)
            self.trade_pairs(filtered_pairs, pairs)
            day += 1

        print('hello')

    def trade_pairs(self, filtered_pairs: dict, pairs: pd.DataFrame):
        days_actions_data = dict(zip(self.returns.columns, [0] * len(self.returns.columns)))
        for stationarity, (stock_a, stock_b) in filtered_pairs:
            pair_name = '(' + stock_a + ', ' + stock_b + ')'
            # this is trading logic
            if pairs[pair_name + '_zscore'].iloc[-1] < \
                    pairs[pair_name + '_lower_threshold'].iloc[-1]:
                request_successful, days_actions_data = self.request_a_long_b_short(
                    days_actions=days_actions_data,
                    stock_a=stock_a,
                    price_a=self.original_data.loc[stock_a, 'Open'].loc[str(pairs.index.date[-1])],
                    stock_b=stock_b,
                    price_b=self.original_data.loc[stock_b, 'Open'].loc[str(pairs.index.date[-1])])

            elif pairs[pair_name + '_zscore'].iloc[-1] > \
                    pairs[pair_name + '_upper_threshold'].iloc[-1]:
                request_successful, days_actions_data = self.request_a_short_b_long(
                    days_actions=days_actions_data,
                    stock_a=stock_a,
                    price_a=self.original_data.loc[stock_a, 'Open'].loc[str(pairs.index.date[-1])],
                    stock_b=stock_b,
                    price_b=self.original_data.loc[stock_b, 'Open'].loc[str(pairs.index.date[-1])])

        days_actions = pd.Series(data=days_actions_data, name=pairs.index[-1])
        self.actions = pd.concat([self.actions, days_actions], axis=1)

    def request_a_short_b_long(self, days_actions: dict, stock_a: str, stock_b: str, price_a: float,
                               price_b: float) -> (bool, dict):
        if self.debt + (price_a * self.even_weight_multiplier) < self.cash + self.assets and self.cash > price_b and \
                (self.actions.empty or (self.actions.iloc[:, -10:].loc[stock_a].abs().sum() < 5 * self.even_weight_multiplier
                                        and self.actions.iloc[:, -10:].loc[stock_b].abs().sum() < 5 * self.even_weight_multiplier)):
            self.positions[stock_a] -= 1 * self.even_weight_multiplier
            self.positions[stock_b] += 1 * self.even_weight_multiplier
            self.debt += price_a * self.even_weight_multiplier
            self.cash -= price_b * self.even_weight_multiplier
            self.assets += price_b * self.even_weight_multiplier
            days_actions[stock_a] -= 1 * self.even_weight_multiplier
            days_actions[stock_b] += 1 * self.even_weight_multiplier
            self.open_trades.append(Trade(stock_a=stock_a, stock_b=stock_b, price_a=price_a, num_a=-1*self.even_weight_multiplier, price_b=price_b, num_b=self.even_weight_multiplier))
            return True, days_actions
        else:
            return False, days_actions

    def request_a_long_b_short(self, days_actions: dict, stock_a: str, stock_b: str, price_a: float,
                               price_b: float) -> (bool, dict):
        if self.debt + (price_b * self.even_weight_multiplier) < self.cash + self.assets and self.cash > price_a and \
                (self.actions.empty or (self.actions.iloc[:, -10:].loc[stock_a].abs().sum() < 5 * self.even_weight_multiplier
                                        and self.actions.iloc[:, -10:].loc[stock_b].abs().sum() < 5 * self.even_weight_multiplier)):
            self.positions[stock_a] += 1 * self.even_weight_multiplier
            self.positions[stock_b] -= 1 * self.even_weight_multiplier
            self.debt += price_b * self.even_weight_multiplier
            self.cash -= price_a * self.even_weight_multiplier
            self.assets += price_a * self.even_weight_multiplier
            days_actions[stock_a] += 1 * self.even_weight_multiplier
            days_actions[stock_b] -= 1 * self.even_weight_multiplier
            self.open_trades.append(Trade(stock_a=stock_a, stock_b=stock_b, price_a=price_a, num_a=self.even_weight_multiplier, price_b=price_b, num_b=-1*self.even_weight_multiplier))

            return True, days_actions
        else:
            return False, days_actions
    
    def exit_positions(self, pairs, rebalance=False):
        indices_to_delete = []
        for index, trade in enumerate(self.open_trades.copy()):
            curr_a_price = self.original_data.loc[trade.stock_a, 'Open'].loc[str(pairs.index.date[-1])]
            curr_b_price = self.original_data.loc[trade.stock_b, 'Open'].loc[str(pairs.index.date[-1])]
            pair_name = '(' + trade.stock_a + ', ' + trade.stock_b + ')'

            percent_change_a = ((curr_a_price - trade.price_a) / trade.price_a) * 100
            percent_change_b = ((curr_b_price - trade.price_b) / trade.price_b) * 100

            if trade.num_a < 0:
                if (percent_change_a > self.stop_loss or percent_change_b < -1*self.stop_loss) or (pairs[pair_name + '_zscore'].iloc[-1] <= pairs[pair_name + '_mean'][-1] or rebalance):
                    self.positions[trade.stock_a] -= trade.num_a
                    self.cash += (curr_a_price - trade.price_a) * abs(trade.num_a)
                    self.debt -= trade.price_a * abs(trade.num_a)

                    self.positions[trade.stock_b] -= trade.num_b
                    self.cash += curr_b_price * abs(trade.num_b)
                    self.assets -= trade.price_b * abs(trade.num_b)
                    indices_to_delete.append(index)

            if trade.num_b < 0:
                if (percent_change_a < -1*self.stop_loss or percent_change_b > self.stop_loss) or (pairs[pair_name + '_zscore'].iloc[-1] >= pairs[pair_name + '_mean'][-1] or rebalance):
                    self.positions[trade.stock_b] -= trade.num_b
                    self.cash += (curr_b_price - trade.price_b) * abs(trade.num_b)
                    self.debt -= trade.price_b * abs(trade.num_b)

                    self.positions[trade.stock_a] -= trade.num_a
                    self.cash += curr_a_price * abs(trade.num_a)
                    self.assets -= trade.price_a * abs(trade.num_a)
                    indices_to_delete.append(index)
            #stoploss: we need to have a check if our trade goes 2% below the point we bought it at. if it does, we
            #cover our losses and buy it back



        for index in reversed(indices_to_delete):
            self.open_trades.remove(self.open_trades[index])

    def update_data(self, curr_data: pd.DataFrame, curr_pairs: list[tuple], initial_update=False) \
            -> (list[tuple], pd.DataFrame):
        pairs = pd.DataFrame()
        filtered_pairs = []
        curr_pairs_to_delete = []
        for index, (stock_a, stock_b) in enumerate(curr_pairs.copy()):
            # converts into resid = stock_a - m(stock_b)
            stock_b_preprocessed = sm.add_constant(curr_data[stock_b])

            # Run OLS regression
            model = sm.OLS(curr_data[stock_a], stock_b_preprocessed).fit()
            pair_name = '(' + stock_a + ', ' + stock_b + ')'

            # ensure that the original stocks are not stationary, otherwise they can't be cointegrated
            if initial_update:
                stationary_a = adfuller(self.original_data.loc[stock_a, 'Open'].loc[curr_data.index])[1]
                stationary_b = adfuller(self.original_data.loc[stock_b, 'Open'].loc[curr_data.index])[1]
                if stationary_a < 0.05 and stationary_b < 0.05:
                    curr_pairs_to_delete.append(index)
                    continue

            # we are checking for stationarity of the residuals now to make sure the spreads themselves are mean
            # reverting. before this would work if on the original series if we were doing prices, because it would
            # just be the difference between them. but now since we are doing a regression, we need to check for
            # stationarity after we find our pairs, then filter them down here with an adfuller test
            hedge_ratio = model.params[1]
            spread = curr_data[stock_a] - hedge_ratio * curr_data[stock_b]
            residual_stationarity = adfuller(spread)[1]

            # spread_lag = spread.shift(1).dropna()
            # spread_lag, spread = spread_lag.align(spread)
            # spread_lag = sm.add_constant(spread_lag)
            # model = sm.OLS(spread, spread_lag).fit()
            lambda_estimate = model.params[1]
            if initial_update:
                if residual_stationarity < 0.05:
                    filtered_pairs.append((residual_stationarity, (stock_a, stock_b)))
                    pairs[pair_name] = spread
                    pairs[pair_name + '_mean'] = pairs[pair_name].mean()
                    pairs[pair_name + '_zscore'] = (pairs[pair_name] - pairs[pair_name + '_mean']) / pairs[pair_name].std()
                    pairs[pair_name + '_upper_threshold'] = pairs[pair_name + '_zscore'].mean() + (
                            2 * pairs[pair_name + '_zscore'].std())
                    pairs[pair_name + '_lower_threshold'] = pairs[pair_name + '_zscore'].mean() - (
                            2 * pairs[pair_name + '_zscore'].std())
                else:
                    curr_pairs_to_delete.append(index)
            else:
                filtered_pairs.append((residual_stationarity, (stock_a, stock_b)))
                pairs[pair_name] = model.resid
                pairs[pair_name + '_mean'] = pairs[pair_name].mean()
                pairs[pair_name + '_zscore'] = (pairs[pair_name] - pairs[pair_name + '_mean']) / pairs[pair_name].std()
                pairs[pair_name + '_upper_threshold'] = pairs[pair_name + '_zscore'].mean() + (
                        2 * pairs[pair_name + '_zscore'].std())
                pairs[pair_name + '_lower_threshold'] = pairs[pair_name + '_zscore'].mean() - (
                        2 * pairs[pair_name + '_zscore'].std())

            # self.plot_zscores(pairs, pair_name, stock_a, stock_b)

        for index in reversed(curr_pairs_to_delete):
            curr_pairs.remove(curr_pairs[index])

        return sorted(filtered_pairs), pairs

    def plot_zscores(self, pairs_df, pair_name, stock_a, stock_b):
        # Extracting z-score data
        zscores = pairs_df[pair_name + '_zscore']
        upper_threshold = pairs_df[pair_name + '_upper_threshold']
        lower_threshold = pairs_df[pair_name + '_lower_threshold']
        mean_zscore = pairs_df[pair_name + '_mean']

        price_a = self.original_data.loc[stock_a, 'Open'].loc[pairs_df.index]
        price_b = self.original_data.loc[stock_b, 'Open'].loc[pairs_df.index]

        # Plotting
        fig, ax1 = plt.subplots(figsize=(15, 7))

        # Plotting z-scores on primary y-axis
        ax1.plot(zscores.index, zscores, label='Z-Score', color='blue')
        ax1.plot(zscores.index, upper_threshold, label='Upper Threshold', linestyle='--', color='red')
        ax1.plot(zscores.index, lower_threshold, label='Lower Threshold', linestyle='--', color='green')
        ax1.plot(zscores.index, mean_zscore, color='black', linestyle='--', label='Mean')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Z-Score', color='blue')
        ax1.legend(loc='upper left')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(price_a.index, price_a, label=f'Price {stock_a}', color='magenta')
        ax2.plot(price_b.index, price_b, label=f'Price {stock_b}', color='cyan')
        ax2.set_ylabel('Stock Price', color='magenta')
        ax2.legend(loc='upper right')
        ax2.tick_params(axis='y', labelcolor='magenta')

        plt.title(f'Z-Scores & Prices for {pair_name}')
        plt.show()


if __name__ == '__main__':
    algo = Algo(Path('./getting-started/train_data_50.csv'), use_log_returns=True)
    algo.run_pairs()
