import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def load_data(data_csv):
    price_data = pd.read_csv(data_csv)
    return price_data


def add_lagged_prices(data_df):
    data_df_copy = data_df.copy()
    for lags in range(1,11):
        temp = data_df.shift(lags).dropna()
        temp.columns = [f"{inst}_{lags}" for inst in data_df.columns]
        data_df_copy = data_df_copy.iloc[1:]
        temp.index = data_df_copy.index
        data_df_copy = pd.concat([data_df_copy, temp], axis=1)
        data_df_copy.reset_index(drop=True, inplace=True)
        temp.reset_index(drop=True, inplace=True)
    return data_df_copy


def split_data(data_df, train_percentage):
    train_len = int(data_df.shape[0] * train_percentage)
    wfcv = TimeSeriesSplit(gap=0, max_train_size=train_len, n_splits=10, test_size=50)
    train_index_list = []
    test_index_list = []
    for i, (train_index, test_index) in enumerate(wfcv.split(data_df)):
        train_index_list.append(list(train_index))
        test_index_list.append(list(test_index))
    return train_index_list, test_index_list, wfcv


def do_seqn_select(data_df, train_indices, test_indices, target_candidates):
    lin_reg = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    svr = SVR()
    xgb = XGBRegressor()
    rfr = RandomForestRegressor()

    y = data_df[target_candidates]
    X = data_df.drop(target_candidates, axis=1)
    X = X.drop([feat for feat in X.columns if "_" not in feat], axis=1)
    ml_algo_list = [lin_reg, ridge, lasso, svr, xgb, rfr]
    algo_names = ['LinearRegression', 'Ridge', 'Lasso', 'SVR', 'XGBRegressor', 'RandomForestRegressor']
    added_feats = []
    mse_sum_list = []
    algo_feat_combos = {}
    y_train = y.iloc[train_indices[0]]
    y_train = y_train.values.ravel()
    comparison_df = pd.DataFrame()
    X_train = pd.DataFrame()
    i = 0

    for algos in ml_algo_list:
        current_min = 1e10
        running_mse = [current_min]
        while min(running_mse) <= current_min:
            not_added_yet = [inst for inst in X.columns if inst not in X_train.columns]
            mse_sum_list = []
            for inst in not_added_yet:
                X_dummy = pd.concat([X_train,X[[inst]]], axis=1)
                algos.fit(X_dummy.iloc[train_indices[0]], y_train)
                mse_list = []
                for folds in test_indices:
                    X_test, y_test = X_dummy.iloc[folds, ], y.iloc[folds]
                    y_test = y_test.values.ravel()
                    y_pred = algos.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    mse_list.append(mse)
                mse_sum_list.append(sum(mse_list))
            comparison_df['mse_sum_from_feat'] = mse_sum_list
            comparison_df['feats_tried'] = [inst for inst in X.columns if inst not in X_train.columns]
            lowest_mse_feat_loc = np.argmin(comparison_df['mse_sum_from_feat'])
            lowest_mse_feat_name = not_added_yet[lowest_mse_feat_loc]
            newest_addition = comparison_df['mse_sum_from_feat'].iloc[lowest_mse_feat_loc]
            comparison_df = pd.DataFrame()
            if ((min(running_mse) - newest_addition) < 0.0001) or (len(not_added_yet) == 1):
                break
            else:
                running_mse.append(newest_addition)
                current_min = min(running_mse)
                X_train[lowest_mse_feat_name] = X[lowest_mse_feat_name]
        print(f"{algo_names[i]} finished seq select")
        algo_feat_combos[algo_names[i]] = [X_train.columns.values.tolist(), min(running_mse)]
        X_train = pd.DataFrame()
        i += 1

    print(algo_feat_combos)
    best_feat_loc_seq_select = min([algo_feat_combos[keys][1] for keys in algo_feat_combos.keys()])
    ml_algo_name = [key for key, value in algo_feat_combos.items() if value[1] == best_feat_loc_seq_select]
    best_feats = algo_feat_combos[ml_algo_name[0]][0]
    best_model = ml_algo_name[0]

    return best_model, best_feats


def tune_model_params(best_model, best_feats, data_df, train_index, target_name, test_indices):
    def objective(trial):
        X = data_df[best_feats].iloc[train_index[0], :]
        y = y = data_df[target_name].iloc[train_index[0]]
        y = y.values.ravel()

        regressor_name = trial.suggest_categorical("regressor", [best_model])

        if regressor_name == 'LinearRegression':
            params = {
                'fit_intercept': trial.suggest_categorical("fit_intercept", [True, False]),
            }

            linreg = LinearRegression(**params)
            linreg.fit(X, y)
            mse_list = []
            for folds in test_indices:
                X_test, y_test = data_df[best_feats].iloc[folds,: ], data_df[target_name].iloc[folds]
                y_test = y_test.values.ravel()
                y_pred = linreg.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)
            score = sum(mse_list)
            return score

        elif regressor_name == 'Ridge':
            params = {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'alpha': trial.suggest_float("alpha", 1e-5, 1e5),
            }

            ridge = Ridge(**params)
            ridge.fit(X, y)
            mse_list = []
            for folds in test_indices:
                X_test, y_test = data_df[best_feats].iloc[folds,: ], data_df[target_name].iloc[folds]
                y_test = y_test.values.ravel()
                y_pred = ridge.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)
            score = sum(mse_list)
            return score

        elif regressor_name == 'Lasso':
            params = {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'alpha': trial.suggest_float("alpha", 1e-5, 1e5),
                'selection': trial.suggest_categorical('selection', ['random', 'cyclic']),
            }

            lasso = Lasso(**params)
            lasso.fit(X, y)
            mse_list = []
            for folds in test_indices:
                X_test, y_test = data_df[best_feats].iloc[folds,: ], data_df[target_name].iloc[folds]
                y_test = y_test.values.ravel()
                y_pred = lasso.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)
            score = sum(mse_list)
            return score

        elif regressor_name == 'SVR':
            params = {
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
                'C': trial.suggest_float("C", 1e-5, 1e5),
                'degree': trial.suggest_int("degree", 3, 9),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }

            svr = SVR(**params)
            svr.fit(X, y)
            mse_list = []
            for folds in test_indices:
                X_test, y_test = data_df[best_feats].iloc[folds,: ], data_df[target_name].iloc[folds]
                y_test = y_test.values.ravel()
                y_pred = svr.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)
            score = sum(mse_list)
            return score

        elif regressor_name == 'XGBRegressor':
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 50, 300),
                'max_depth': trial.suggest_int("max_depth", 3, 12),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                'subsample': trial.suggest_float("subsample", 0.5, 1.0),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
                'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
                'gamma': trial.suggest_float("gamma", 0.0, 5.0),
                'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            }

            xgbr = XGBRegressor(**params)
            xgbr.fit(X, y)
            mse_list = []
            for folds in test_indices:
                X_test, y_test = data_df[best_feats].iloc[folds,: ], data_df[target_name].iloc[folds]
                y_test = y_test.values.ravel()
                y_pred = xgbr.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)
            score = sum(mse_list)
            return score

        elif regressor_name == 'RandomForestRegressor':
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 50, 300),
                'max_depth': trial.suggest_int("max_depth", 3, 30),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
                'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 10, 1000),
                'min_impurity_decrease': trial.suggest_float("min_impurity_decrease", 0.0, 0.1),
                'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
                'ccp_alpha': trial.suggest_float("ccp_alpha", 0.0, 0.05),
                'criterion': trial.suggest_categorical("criterion",
                                                       ["squared_error", "absolute_error", "friedman_mse", "poisson"]),
            }

            rfr = RandomForestRegressor(**params)
            rfr.fit(X, y)
            mse_list = []
            for folds in test_indices:
                X_test, y_test = data_df[best_feats].iloc[folds,: ], data_df[target_name].iloc[folds]
                y_test = y_test.values.ravel()
                y_pred = rfr.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)
            score = sum(mse_list)
            return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, catch=(ValueError,))
    print(study.best_params)
    return study.best_params


def wf_validate_models(best_model, tuned_params, best_data_df, train_index_list, test_indices, data_df, target_name):
    X = best_data_df.iloc[train_index_list[0], :]
    y = data_df[target_name].iloc[train_index_list[0]]
    y = y.values.ravel()
    if best_model == 'XGBRegressor':
        final_model = XGBRegressor(**tuned_params)
    elif best_model == 'LinearRegression':
        final_model = LinearRegression(**tuned_params)
    elif best_model == 'Ridge':
        final_model = Ridge(**tuned_params)
    elif best_model == 'Lasso':
        final_model = Lasso(**tuned_params)
    elif best_model == 'SVR':
        final_model = SVR(**tuned_params)
    elif best_model == 'RandomForestRegressor':
        final_model = RandomForestRegressor(**tuned_params)
    else:
        raise ValueError(f"Unknown model: {best_model}")

    final_model.fit(X, y)

    mse_list = []
    for folds in test_indices:
        X_test, y_test = best_data_df.iloc[folds, :], data_df[target_name].iloc[folds]
        y_pred = final_model.predict(X_test)
        mse = mean_squared_error(y_test.values, y_pred)
        mse_list.append(mse)
    score = sum(mse_list)
    return score, mse_list


def main():
    price_data = load_data('prices.csv')
    price_data_featured = add_lagged_prices(price_data)
    for targets in ["AMRP", "DIHO", "SRTX", "MSDP", "GARI", "DUCT"]:
        print("----------------------")
        print("targets: ", targets, "\n")
        train_index_list, test_index_list, wfcv_object = split_data(price_data_featured, 0.5)
        model, feats = do_seqn_select(price_data_featured, train_index_list, test_index_list, [targets])
        print(model, feats)
        optimised_params = tune_model_params(model, feats, price_data_featured, train_index_list, targets, test_index_list)
        model_name = optimised_params['regressor']
        model_params = {k: v for k, v in optimised_params.items() if k != 'regressor'}
        mse_sum, mse_wfv = wf_validate_models(model_name, model_params, price_data_featured[feats], train_index_list,
                                              test_index_list, price_data_featured, targets)
        print(f"MSE: {mse_sum}")
        print(f"WFV: {mse_wfv}")
        print("----------------------")


if __name__ == '__main__':
    main()
