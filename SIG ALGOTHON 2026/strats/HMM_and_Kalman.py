import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

prices = pd.read_csv('prices.csv')
prices = prices
returns = prices.pct_change().dropna()

hmm_candidates = prices[['NAYO', 'MTNS', 'EELT', 'ULXY', 'BENI']]

candidates_train = hmm_candidates.iloc[:350, ]
candidates_test = hmm_candidates.iloc[350:, ]


# Step 1: Fit the HMM for each instrument

from hmmlearn import hmm

hidden_states = [2,3,4,5]

best_states = {}

for instruments in ['NAYO', 'MTNS', 'EELT', 'ULXY', 'BENI']:
    bics = []
    ks = []
    print("-------------------------")
    print(instruments, "\n")
    for state in hidden_states:
        reshaped = np.array(candidates_train[instruments]).reshape(-1,1)
        hmm_instance = hmm.GaussianHMM(n_components=state, covariance_type='full')
        hmm_instance.fit(reshaped)
        print("Transition Matrix:\n", hmm_instance.transmat_)
        print("Means of hidden states:\n", hmm_instance.means_)
        print("Covariances of hidden states:\n", hmm_instance.covars_)
        ll = hmm_instance.score(reshaped)
        n_features = reshaped.shape[1]
        n_states = hmm_instance.n_components
        params = (n_states - 1) + (n_states * (n_states - 1)) + n_states * (n_features + n_features * (n_features + 1) // 2)
        bic = -2 * ll + params * np.log(len(reshaped))
        bics.append(bic)
        print(f"hmm score: {ll}")
        print(f"BIC: {bic}")
        print(state, "\n\n")
        states = hmm_instance.predict(reshaped)
        counts = np.bincount(states)
        print(counts)

    print("-------------------------")
    print(f"for {instruments}, most -ve bic = {min(bics)}.  Associated k = {hidden_states[bics.index(min(bics))]}")
    best_states[instruments] = hidden_states[bics.index(min(bics))]

print(best_states)


best_states = pd.Series(best_states)
hmm_list = []

for instruments in ['NAYO', 'MTNS', 'EELT', 'ULXY', 'BENI']:
    hmm_instance = hmm.GaussianHMM(n_components=best_states[instruments], covariance_type='full')
    reshaped = np.array(candidates_train[instruments]).reshape(-1,1)
    hmm_instance.fit(reshaped)
    hmm_list.append(hmm_instance)
    oos_reshaped = np.array(candidates_test[instruments]).reshape(-1,1)

    oos_pred = hmm_instance.score(oos_reshaped)
    print(f"{instruments}: oos_pred = {oos_pred}, (ll)")

    posterior = hmm_instance.predict_proba(reshaped)
    # print(posterior)

    transition_matrix = hmm_instance.transmat_
    regime_length = 1.0 / (1.0 - np.diag(transition_matrix))

    print(f"{instruments}: regime_length = {regime_length}, (days)")


# Step 2: Fit the Model.  As of now (16/07) I'm assuming a modified OU process:
# price(t) = μ + φ(price(t-1) − μ) + noise.  μ comes from the HMM

for instruments in ['NAYO', 'MTNS', 'EELT', 'ULXY', 'BENI']:
    instrument = np.array(candidates_train[instruments])
    OU = AutoReg(instrument, lags=1, trend='ct')
    OU_fit = OU.fit()
    print(f"Instrument: {instruments} OU summary:\n\n")
    print(OU_fit.summary())
    pacf_coeffs = pacf(instrument, nlags=20)
    print(f"half life: {np.log(0.5)/np.log(OU_fit.params[2])}")
    print(f"mu: {OU_fit.params[0]/(1-OU_fit.params[2])}")

    # fig, ax = plt.subplots(figsize=(8, 5))
    # plot_pacf(instrument, lags=20, ax=ax, method='ywadjusted')
    #
    # plt.xlabel("Lags")
    # plt.ylabel("Partial Autocorrelation")
    # plt.title("Partial Autocorrelation Function (PACF)")
    # plt.show()


# Step 3: Set up the Kalman Filter, because somehow the handpicked stocks are all OU processes.

class KalmanFilter:
    def __init__(self, F, H, Q, R, P0, B, x0, ):
        """
        F is the motion model (our OU process)
        B is an input to the system (μ from our HMM)
        H is the observation matrix, our observation of our instruments (prolly use some actual plotted points)
        Q is the process noise covariance, estimate that myself?
        x0 is our price at t0
        P0 is our covariance estimate at P0
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P0
        self.B = B
        self.x = x0

    def predict(self, u):
        """use the motion model, predict the covariance"""
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T ), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K,y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

