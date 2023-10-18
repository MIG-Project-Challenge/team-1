import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm

from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller


class Trade(BaseModel):
    stock_a: str
    stock_b: str
    price_a: float
    price_b: float


class Algo:
    def __init__(self, data_path: Path, cash=25000, n_components=30, n_clusters=5, use_log_returns=False):
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

        self.actions = pd.DataFrame(columns=self.returns.columns)
        self.positions = dict(zip(self.returns.columns, [0] * len(self.returns.columns)))
        self.open_trades: list[Trade] = []
        self.cash = cash
        self.assets = 0
        self.debt = 0

    def run_pairs(self):
        pairs = pd.DataFrame()
        curr_pairs = []
        look_back_window = 30
        for index, (date, row) in enumerate(self.returns.iloc[look_back_window - 1:].iterrows()):
            if index % look_back_window == 0:
                pairs = pd.DataFrame()
                curr_pairs = []

                # get last 60 days of data for PCA
                if self.use_log_returns:
                    curr_data = self.log_returns.iloc[index: index + look_back_window]
                else:
                    curr_data = self.returns.iloc[index: index + look_back_window]
                stocks = list(curr_data.columns)

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
                    for i in range(len(cluster)):
                        for j in range(i + 1, len(cluster)):
                            # WE DON'T NEED THIS. COINTEGRATION ONLY MATTERS ON PRICES, WE DON'T NEED IT
                            # ON LOGGED RETURNS OR RETURNS

                            # _, coint_p_val, _ = coint(curr_data[cluster[i]], curr_data[cluster[j]])
                            # a_stationary = adfuller(curr_data[cluster[i]])[1]
                            # b_stationary = adfuller(curr_data[cluster[j]])[1]
                            # if coint_p_val < 0.05 and a_stationary < 0.05 and b_stationary < 0.05:
                            curr_pairs.append((cluster[i], cluster[j]))

                curr_pairs = sorted(curr_pairs)
                filtered_pairs, pairs = self.initialize_month(curr_data=curr_data, pairs=pairs, curr_pairs=curr_pairs)

                days_actions = pd.Series()
                days_actions_data = dict(zip(self.returns.columns, [0] * len(self.returns.columns)))
                for stationarity, (stock_a, stock_b) in filtered_pairs:
                    pair_name = '(' + stock_a + ', ' + stock_b + ')'
                    # this is trading logic
                    if pairs[pair_name + '_zscore'].iloc[-1] < \
                            pairs[pair_name + '_lower_threshold'].iloc[-1]:
                        request_successful, days_actions_data = self.request_a_long_b_short(
                            date=pairs.index[-1],
                            days_actions=days_actions_data,
                            stock_a=stock_a,
                            price_a=self.original_data.loc[stock_a, 'Open'].iloc[-1],
                            stock_b=stock_b,
                            price_b=self.original_data.loc[stock_b, 'Open'].iloc[-1])

                    elif pairs[pair_name + '_zscore'].iloc[-1] > \
                            pairs[pair_name + '_upper_threshold'].iloc[-1]:
                        request_successful, days_actions_data = self.request_a_short_b_long(
                            date=pairs.index[-1],
                            days_actions=days_actions_data,
                            stock_a=stock_a,
                            price_a=self.original_data.loc[stock_a, 'Open'].iloc[-1],
                            stock_b=stock_b,
                            price_b=self.original_data.loc[stock_b, 'Open'].iloc[-1])

                days_actions = pd.Series(data=days_actions_data, name=pairs.index[-1])
                self.actions = pd.concat([self.actions, days_actions])

                print('hello')
        # pairs_spreads = pd.DataFrame()
        # for index, row in enumerate(data):
        # EVERY MONTH WE GET NEW PAIRS TO TRADE
        # if index % 30 == 0:
        # run pca on [i - 60: i]
        # get clusters for stocks
        # calc initial averages and put into pairs spreads
        # trade on this day
        # else
        # get spread of today and updates pairs_spreads
        # for pair in pairs.columns:
        # trade

    def request_a_short_b_long(self, date: datetime, days_actions: dict, stock_a: str, stock_b: str, price_a: float, price_b: float) -> (bool, dict):
        if self.debt + price_a < self.cash + self.assets and self.cash > price_b:
            self.positions[stock_a] -= 1
            self.positions[stock_b] += 1
            self.debt += price_a
            self.cash -= price_b
            self.assets += price_b
            days_actions[stock_a] -= 1
            days_actions[stock_b] += 1
            self.open_trades.append(Trade(stock_a=stock_a, stock_b=stock_b, price_a=-price_a, price_b=price_b))

            return True, days_actions
        else:
            return False

    def request_a_long_b_short(self, date: datetime, days_actions: dict, stock_a: str, stock_b: str, price_a: float, price_b: float) -> (bool, dict):
        if self.debt + price_b < self.cash + self.assets and self.cash > price_a:
            self.positions[stock_a] += 1
            self.positions[stock_b] -= 1
            self.debt += price_b
            self.cash -= price_a
            self.assets += price_a
            days_actions[stock_a] += 1
            days_actions[stock_b] -= 1
            self.open_trades.append(Trade(stock_a=stock_a, stock_b=stock_b, price_a=price_a, price_b=-price_b))

            return True, days_actions
        else:
            return False

    @staticmethod
    def initialize_month(curr_data: pd.DataFrame, pairs: pd.DataFrame, curr_pairs: list[tuple]) \
            -> (list[tuple], pd.DataFrame):
        filtered_pairs = []
        for stock_a, stock_b in curr_pairs:
            # converts into resid = stock_a - m(stock_b)
            stock_b_preprocessed = sm.add_constant(curr_data[stock_b])

            # Run OLS regression
            model = sm.OLS(curr_data[stock_a], stock_b_preprocessed).fit()
            pair_name = '(' + stock_a + ', ' + stock_b + ')'

            # we are checking for stationarity of the residuals now to make sure the spreads themselves are mean
            # reverting. before this would work if on the original series if we were doing prices, because it would
            # just be the difference between them. but now since we are doing a regression, we need to check for
            # stationarity after we find our pairs, then filter them down here with an adfuller test
            residual_stationarity = adfuller(model.resid)[1]
            if residual_stationarity < 0.05:
                filtered_pairs.append((residual_stationarity, (stock_a, stock_b)))
                pairs[pair_name] = model.resid
                pairs[pair_name + '_mean'] = pairs[pair_name].mean()
                pairs[pair_name + '_zscore'] = (pairs[pair_name] - pairs[pair_name + '_mean']) / pairs[pair_name].std()
                pairs[pair_name + '_upper_threshold'] = pairs[pair_name + '_zscore'].mean() + (
                        2 * pairs[pair_name + '_zscore'].std())
                pairs[pair_name + '_lower_threshold'] = pairs[pair_name + '_zscore'].mean() - (
                        2 * pairs[pair_name + '_zscore'].std())

        return sorted(filtered_pairs), pairs


if __name__ == '__main__':
    algo = Algo(Path('./getting-started/train_data_50.csv'), use_log_returns=True)
    algo.run_pairs()
