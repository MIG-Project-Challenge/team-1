import warnings
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm


class Algo:
    def __init__(self, data_path: Path, cash=25000, n_components=30, n_clusters=8, mean_window=10):
        warnings.simplefilter(action='ignore', category=Warning)

        self.original_data = pd.read_csv(data_path)
        self.original_data['Date'] = pd.to_datetime(self.original_data['Date'])
        self.original_data.set_index(['Ticker', 'Date'], inplace=True)

        self.original_data['Ret'] = self.original_data['Adj Close'].groupby('Ticker').pct_change()
        self.original_data.dropna(inplace=True)
        self.returns = self.original_data['Ret'].unstack(level='Ticker')

        self.n_components = n_components
        self.n_clusters = n_clusters
        self.mean_window = mean_window

        self.actions = pd.DataFrame(columns=self.returns.columns)
        self.positions = [0] * len(self.returns.columns)
        self.cash = cash
        self.debt = 0

    def run_pairs(self):
        pairs = pd.DataFrame()
        curr_pairs = []
        look_back_window = 30
        for index, (date, row) in enumerate(self.returns.iloc[look_back_window - 1:].iterrows()):
            if index % 30 == 0:
                pairs = pd.DataFrame()
                curr_pairs = []

                # get last 60 days of data for PCA
                curr_data = self.returns.iloc[index: index + look_back_window]
                stocks = list(curr_data.columns)

                # scale data and run PCA
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(curr_data)
                pca = PCA(n_components=self.n_components)
                components = pca.fit(scaled_data).components_.T

                # run kmeans and get clusters
                kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=50, n_init=10, random_state=7)
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
                            _, p_val, _ = coint(curr_data[cluster[i]], curr_data[cluster[j]])
                            if p_val < 0.05:
                                curr_pairs.append((p_val, (cluster[i], cluster[j])))

                curr_pairs = sorted(curr_pairs)
                pairs = self.initialize_month(curr_data=curr_data, pairs=pairs, curr_pairs=curr_pairs)

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

    def initialize_month(self, curr_data: pd.DataFrame, pairs: pd.DataFrame, curr_pairs: list[tuple]) -> pd.DataFrame:
        for p_val, (stock_a, stock_b) in curr_pairs:
            # converts into resid = stock_b - m(stock_a)
            stock_a_preprocessed = sm.add_constant(curr_data[stock_a])

            # Run OLS regression
            model = sm.OLS(curr_data[stock_b], stock_a_preprocessed).fit()
            pair_name = '(' + stock_a + ', ' + stock_b + ')'
            pairs[pair_name] = model.resid
            pairs[pair_name + '_' + str(self.mean_window) + 'MA'] = pairs[pair_name].rolling(
                self.mean_window).mean()
        return pairs


if __name__ == '__main__':
    algo = Algo(Path('./getting-started/train_data_50.csv'))
    algo.run_pairs()
