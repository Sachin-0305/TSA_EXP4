# Fit-the-ARMA-model-for-any-data-set

### DEVELOPED BY: SACHIN M
### REGISTER NO: 212223040177

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
plt.title('Yearly Average Home Scores')
plt.xlabel("Year")
plt.ylabel("Avg Home Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_1, lags=40, ax=plt.gca())
plt.title("ACF of Simulated ARMA(1,1)")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_1, lags=40, ax=plt.gca())
plt.title("PACF of Simulated ARMA(1,1)")
```






# OUTPUT:

## Original data:

<img width="1290" height="669" alt="Screenshot 2025-09-15 160231" src="https://github.com/user-attachments/assets/6ee8de63-b37d-4d19-9d16-cecf63e21276" />

## Autocorrelation:

<img width="1381" height="349" alt="Screenshot 2025-09-15 160321" src="https://github.com/user-attachments/assets/2570ae48-74c2-482a-a683-26f9eb6a7512" />

## Partial Autocorrelation:

<img width="1370" height="349" alt="Screenshot 2025-09-15 160407" src="https://github.com/user-attachments/assets/e33c24d5-2709-453b-abac-829b69706a5d" />


## SIMULATED ARMA(1,1) PROCESS:

<img width="1251" height="650" alt="Screenshot 2025-09-15 160443" src="https://github.com/user-attachments/assets/5c409d5f-6730-48a2-bbc9-68ecbb6625a2" />


## Autocorrelation:

<img width="1256" height="662" alt="Screenshot 2025-09-15 160513" src="https://github.com/user-attachments/assets/3c75e582-c7d0-45cf-aa74-d95990fdc70b" />


## Partial Autocorrelation:

<img width="1266" height="658" alt="Screenshot 2025-09-15 160547" src="https://github.com/user-attachments/assets/450e9a13-4faf-4c63-bd3b-c285f05131c8" />


## SIMULATED ARMA(2,2) PROCESS:

<img width="1248" height="652" alt="Screenshot 2025-09-15 160612" src="https://github.com/user-attachments/assets/e26664b7-6c77-4f99-b012-12096692a8e9" />



## Autocorrelation:

<img width="1242" height="644" alt="Screenshot 2025-09-15 160643" src="https://github.com/user-attachments/assets/1da60fe5-1964-4b84-9fc9-a55ab66ba1aa" />

## Partial Autocorrelation:

<img width="1265" height="649" alt="Screenshot 2025-09-15 160808" src="https://github.com/user-attachments/assets/42185c8b-58bc-4749-8372-0da43c92b133" />


# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
