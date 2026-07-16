import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import pandas as pd
import numpy as np

import plotly.io as pio
pio.renderers.default = "browser"

prices = pd.read_csv('prices.csv')
returns = prices.pct_change().dropna()

fig = px.line(prices, markers=True)
fig.show()

fig2 = px.histogram(prices)
fig2.show()

fig3 = px.line(returns,markers=True)
fig3.show()


