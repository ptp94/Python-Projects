# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:03:15 2020

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

data.describe()
data.isnull().sum()

cols = ['Order Date', 'Ship Mode', 'Segment', 'City', 'State', 'Region', 
                                    'Sub-Category', 'Sales', 'Quantity', 
                                    'Discount', 'Profit']

data1 = data.loc[:, cols]

del cols

data1.corr()

data_corr = data1.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(data_corr, vmax = 0.8,
                      linewidths=0.01, square=True, annot=True, 
                      cmap="YlOrRd",
                      linecolor="white", )
plt.title("Correlation between features") 
plt.show()

del data_corr

y = data1.loc[:, 'Sales']
X = data1.loc[:, data1.columns != "Sales"]

del data1

X['Order Date'] = pd.to_datetime(X['Order Date'])
X.info()

#Encoding the Dataset's categorical variables
dum_cols = ['Ship Mode', 'Segment', 'City', 'State', 'Region', 'Sub-Category']
X1 = pd.get_dummies(X, columns=dum_cols, drop_first=True)
del dum_cols

#Applying OLS to our standard dataset
from statsmodels.api import OLS

X_array = np.array(X1)
y_array = np.array(y)

ols = OLS(y_array, X_array[:,1:].astype(float)).fit()

#Looking at Summary to check for Adj.R2 and Columns which have pvalue > 0.05
ols.summary()



#Too many columns so conduct Backward Elimination to improve model performance
def backwardElimination(x, y, SL):
    '''
    x := The input array
    y := dependant Variable 
    SL := Significance Level
    
    Function conducts backward elimination to optimise the OLS model
    '''
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues.astype(float))
        #adjR = regressor_OLS.rsquared_adj.astype(float)
        if (maxVar > SL):
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x

#Significance Level at 0.05 (5%)
SL = 0.05
X_input = X_array[:,1:].astype(float)
X_BE = backwardElimination(X_input, y_array, SL)

#Convert to Dataframe for Correlation Check
X_BE_df = pd.DataFrame(data=X_BE)
X_BE_corr = X_BE_df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(X_BE_corr[(X_BE_corr >= 0.5) | (X_BE_corr <= -0.5)],
                      vmax = 0.8,
                      linewidths=0.01, square=True, annot=True, 
                      cmap="YlOrRd",
                      linecolor="white", )
plt.title("Correlation between features") 
plt.show()

#Checking which row/column to delete
def collinear_cols(corr_matrix):
    '''
    corr_matrix := Add the matrix to find the collinear rows and column
    '''
    for i in range(28):
        for j in range(28):
            a = corr_matrix[i][j]
            if i == j:
                continue
            elif (a >= 0.5) | (a <= -0.5):
                print(i, j)
    
collinear_cols(X_BE_corr)

#Deleting 7th column since it has all 0s except one 1 value
X_BE = np.delete(X_BE, 7, 1)

del X_BE_df

#Creating a DataFrame to 
X_BE_df = pd.DataFrame(X_BE)

X2 = pd.concat([X['Order Date'], X_BE_df], axis=1) 

X3 = np.array(X2)
X3.shape

#Importing function to convert time-series into supervised learning dataset.
from series_to_supervised import series_to_supervised

#Going to apply a shift of 7 days and predict the 8th day (Weekly)
dataset_final = series_to_supervised(X3, n_in=7)





        