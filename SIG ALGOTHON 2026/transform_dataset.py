import numpy as np
import pandas as pd


def transform_price_ds(filepath):
    prices = pd.read_table(filepath, sep=r'\s+', engine='python')
    print(prices.columns)

    # row_df = pd.DataFrame(data=prices.columns.values.tolist())
    # row_df = row_df.T
    # row_df.columns = prices.columns
    # prices = pd.concat([row_df, prices], axis=0)
    #
    # prices.reset_index(drop=True, inplace=True)

    prices.astype(float)

    prices.to_csv('prices.csv', index=False)


def load_dataset(csv_filepath):
    prices = pd.read_csv(csv_filepath)
    return prices


transform_price_ds('/Users/neilkumar/Desktop/Python/SIG ALGOTHON 2026/prices.txt')
