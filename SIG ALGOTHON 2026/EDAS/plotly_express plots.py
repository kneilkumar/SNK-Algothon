import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import pandas as pd
import numpy as np

import plotly.io as pio
pio.renderers.default = "browser"

prices = pd.read_csv('prices.csv')
returns = prices.pct_change().dropna()

fig = px.line(prices, markers=False)
fig.show()

fig2 = px.histogram(prices)
fig2.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
pca = PCA(n_components=2)

prices = np.log(prices)
fig3 = px.line(prices, markers=False)
fig3.show()

X = prices[['AMRP','DUCT', 'MSDP', 'CUBO', 'ANSO', 'NWIG', 'ALUT', 'ACAC', 'GARI', 'ACIX', 'CCNS', 'MTNS',
                       'FWWG', 'HRND', 'NGTE', 'FARS', 'MHRM', 'EAFC']]




