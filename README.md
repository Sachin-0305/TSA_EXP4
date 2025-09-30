# Fit-the-ARMA-model-for-any-data-set

### DEVELOPED BY: PRAJAN P
### REGISTER NO: 212223240121

# AIM:
To implement ARMA model in python.

# ALGORITHM:
1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.

# PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("results.csv")
data['date'] = pd.to_datetime(data['date'])

yearly_scores = data.groupby(data['date'].dt.year)['home_score'].mean().reset_index()
yearly_scores.rename(columns={'date': 'year', 'home_score': 'avg_home_score'}, inplace=True)

X = yearly_scores['avg_home_score'].dropna().values
N = 1000

plt.figure(figsize=(12, 6))
plt.plot(yearly_scores['year'], X, marker='o')
