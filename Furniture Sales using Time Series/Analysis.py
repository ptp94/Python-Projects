# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:05:00 2020

@author: ptpar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

data = pd.read_csv("C:\\Users\\ptpar\\OneDrive\\Documents\\Projects\\Python\\Furniture Sales using Time Series\\Super_Store_data.csv", encoding='cp1252')

#---- EDA ----
data.head()
data.info()
data.describe()
data.isnull().sum()

#Predicting Sales using Past Sales Data only
data['Order Date'].min(), data['Order Date'].max()

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 
        'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code',
        'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name',
        'Quantity', 'Discount', 'Profit']

data_4ts = data.drop(columns=cols, axis=1)

data_4ts = data_4ts.sort_values('Order Date')

#Making the time stamps into date-time objects
data_4ts['Order Date'] = pd.to_datetime(data_4ts['Order Date'])

#Setting the date as the index
data_4ts.set_index('Order Date', inplace=True)

data_4ts.index

#Resampling Monthly and taking mean sales of the month
y = data_4ts['Sales'].resample('MS').mean()

#Looking at sales in the year 2017 
y['2017':]
#Plotting
y.plot(figsize=(10, 10))
plt.ylabel("Sales")
plt.title("Sales (2014-17)")
plt.show()

#There is seasonality present in the time series. The sales are
#always low at the beginning of the year and high towards the end.

'''
Checking Stationarity
We will use Augemented Dickey-Fuller (ADF) statistic as it is one of the
more widely used statistical test for checking stationarity in a time-series.

ADF uses an autoregressive (AR) model and optimises an information criterion
across different lag values.

Ho (null): 
    Time Series can be represented by a unit root (non-stationary)
    so there is some time-dependent structure.
    
H1 (alternative):
    The time is series is stationary
'''
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
print('ADF Statistic: {}'.format(result[0]))
print('P-value: {}'.format(result[1]))
for key,values in result[4].items():
    print("{}: {:.3f}".format(key, values))
    
'''
ADF Statistic is -6.96 which is less than the value of -3.578 (1%).
Therefore we reject the Ho (null) hypothesis with the significance level
less than 1&. As a result, we can conclude that our time series is stationary.

The p-value is way below the threshold of (0.05), hence the null hypothesis
is rejected. '''

#==== Decomposing ====
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(y)

plt.plot(y, label="Original")
plt.legend(loc='best')

#Trend
trend = decomposition.trend
plt.show()
plt.plot(trend, label="Trend")
plt.legend(loc='best')

#Seasonal
season = decomposition.seasonal
plt.show()
plt.plot(season, label="Seasonal")
plt.legend(loc="best")

#Residuals
residuals = decomposition.resid
plt.show()
plt.plot(residuals, label="Residuals")
plt.legend(loc="best")

#Forecast using ARIMA(p, d, q)
import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

print("Combinations of Seasonal ARIMA (SARIMAX)")
print("SARIMAX: {} x {}".format(pdq[1], seasonal_pdq[1]))
print("SARIMAX: {} x {}".format(pdq[1], seasonal_pdq[2]))
print("SARIMAX: {} x {}".format(pdq[2], seasonal_pdq[3]))
print("SARIMAX: {} x {}".format(pdq[2], seasonal_pdq[4]))

#---- Parameter Selection for ARIMA ----
from pylab import rcParams
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order=param, 
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            
            if results.aic < 300:
                print("ARIMA{} x {}12 - AIC:{}".format(param, param_seasonal,
                      results.aic))
            
        except:
            continue

#Fitting the ARIMA model 
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1,1,1),
                                seasonal_order=(1,1,0,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(10,10))
plt.show()
                                        
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'),
                              dynamic=False)

pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label="One-step forward Forecast", 
                         alpha=.7, figsize=(12, 12))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.2)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
plt.title("One-year forward Forecast")
plt.legend()
plt.show()

#RMSE
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print("The RMSE of Forecast: {}".format(round(np.sqrt(mse), 2)))

'''
Our model was able to forecast average daily sales in the test set within
76.88 of the real sales. '''


pred_uc = results.get_forecast(steps=13)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label="Forecast")
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.25)

ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
print(pred_ci)
plt.legend()
plt.title("Forecast")
plt.show()


