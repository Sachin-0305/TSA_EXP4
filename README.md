# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 4/10/2025



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
Import Necessary Libraries:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```
Load dataset:
```
data = pd.read_csv("GoogleStockPrices.csv")
print(data.head())
```
Declare required variables and set figure size, and visualise the data:
```
X = data['Close']   
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(X)
plt.title('Google Stock Prices (Original Data)')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
```
Fitting the ARMA(1,1) model and deriving parameters:
```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
```
Simulate ARMA(1,1) Process:
```
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
```
Plot ACF and PACF for ARMA(1,1):
```
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
```
Fitting the ARMA(2,2) model and deriving parameters:
```
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
```
Simulate ARMA(2,2) Process:
```
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
```
Plot ACF and PACF for ARMA(2,2):
```
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

### OUTPUT:
Original data:

<img width="936" height="501" alt="image" src="https://github.com/user-attachments/assets/cc4e3200-76ed-4c3b-8dc3-17e94a8db9c5" />

Partial Autocorrelation:
<img width="942" height="231" alt="image" src="https://github.com/user-attachments/assets/3278a56c-c85c-4aa6-8d45-e4e6af71eb9f" />

Autocorrelation:
<img width="943" height="243" alt="image" src="https://github.com/user-attachments/assets/a467bdd3-cab6-4918-ae3a-8549c34b2ae0" />

SIMULATED ARMA(1,1) PROCESS:

<img width="934" height="504" alt="image" src="https://github.com/user-attachments/assets/62e76753-b783-4fb2-8ca5-1d1564f167ef" />

Partial Autocorrelation:

<img width="956" height="480" alt="image" src="https://github.com/user-attachments/assets/f0a46516-6888-4028-85c4-c3581a3137b1" />

Autocorrelation:

<img width="938" height="494" alt="image" src="https://github.com/user-attachments/assets/9df8545a-2ac0-4d7c-97c9-20074a7b45b0" />




SIMULATED ARMA(2,2) PROCESS:

<img width="932" height="487" alt="image" src="https://github.com/user-attachments/assets/5735cfdb-9d11-48b4-9672-61ee8730106a" />

Partial Autocorrelation:

<img width="936" height="498" alt="image" src="https://github.com/user-attachments/assets/a0349bbe-def2-4c64-9df5-1a461293ac98" />



Autocorrelation:

<img width="948" height="489" alt="image" src="https://github.com/user-attachments/assets/7cfeb5ad-db1b-4838-93f2-e379e061bb59" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
