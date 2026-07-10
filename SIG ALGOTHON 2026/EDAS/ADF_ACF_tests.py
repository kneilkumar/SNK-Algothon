from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
import pandas as pd
import numpy as np

prices = pd.read_csv('prices.csv')


def adf_test():
    for instruments in prices.columns:
        adf_test_results = adfuller(prices[instruments], autolag='BIC', regression='ct')
        print(f"instrument name: {instruments}\n"
              f"adf-stat: {adf_test_results[0]}\n"
              f"n-lags: {adf_test_results[2]}\n"
              f"p-val: {adf_test_results[1]}\n"
              f'Critical Values:')
        for key, value in adf_test_results[4].items():
            print('\t%s: %.3f' % (key, value))
        print('-------------------------------------')

        if (adf_test_results[1] <= 0.05) or (adf_test_results[0] <= adf_test_results[4]["1%"]) \
                or (adf_test_results[0] <= adf_test_results[4]["10%"]) or (
                adf_test_results[0] <= adf_test_results[4]["5%"]):
            if adf_test_results[1] <= 0.05:
                print("P-VAL IS SIG")

            if adf_test_results[0] <= adf_test_results[4]["1%"]:
                print("ADF-STAT MORE -VE THAN 1% LEVEL")

            if adf_test_results[0] <= adf_test_results[4]["10%"]:
                print("ADF-STAT MORE -VE THAN 10% LEVEL")

            if adf_test_results[0] <= adf_test_results[4]["5%"]:
                print("ADF-STAT MORE -VE THAN 5% LEVEL")
        print('-------------------------------------\n')


# def acf_test():
#     for instruments in prices.columns:


import itertools
import dcor

n_instruments = 51
results = []

for i, j in itertools.combinations(range(n_instruments), 2):
    dcorr = dcor.distance_correlation(prices.iloc[:, i].values, prices.iloc[:, j].values)
    print(prices.columns.values[i])
    results.append({'instrument_1': i, 'instrument_2': j, 'distance_corr': dcorr})

dcorr_df = pd.DataFrame(results).sort_values('distance_corr', ascending=False)
dcorr_df.to_csv('dcorr_df.csv')



