# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:15:33 2018

@author: Preet

There are several factors that influence the price a buyer is willing to 
pay for a house. Some are apparent and obvious and some are not. 
The challenge is to learn a relationship  between the important features and 
the price and use it to predict the prices of a new set of houses.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the Dataset
data = pd.read_csv("train.csv")

#Data Preparation for datatypes
data.info()

data.shape

#Statistical Summary
data.describe(include=['object'])

data.describe(include=['int64'])

#As we are predicting the Price, splitting the variable.
target = data['SalePrice']

target.head()

import seaborn as sns
sns.distplot(target, hist=True)

#Data left-skewed, so log-transform to make it Normally distributed
target_logtrans = np.log(target)

sns.distplot(target_logtrans)

plt.rcParams['figure.figsize'] = (12, 6)
prices = pd.DataFrame({"Sale Price": data["SalePrice"], 
                       "Log-Sale Price": target_logtrans})
prices.hist() #Data looks more Normally Distributed now

#Dropping The target var from dataset
data_backup = data #creating a backup - just in case!
data = data.drop(columns = ["SalePrice"])
data.head()

#---- Feature Engineering ----
#We know that MSSubClass is a class not an integer
data['MSSubClass'] = data['MSSubClass'].apply(str)

#The overall condition OverallCond is also a categorical var
data['OverallCond'] = data['OverallCond'].astype(str)

#Same with Year and month sold
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)

#Aggregating the Square Footage 
data['TotalSqft'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

#Dropping above columns and the Id since we no longer need them
data = data.drop(columns=['Id','TotalBsmtSF','1stFlrSF','2ndFlrSF'])

#Splitting Data into Categorical and Numerical Variables 
data.columns.values
cat_cols = [col for col in data.columns.values if data[col].dtype == 'object']
data_cat = data[cat_cols]

#Numerical Data = Data - Categorical Vars
data_num = data.drop(cat_cols, axis = 1)

data_num.describe()

data_cat.head()

#Skewness: Finding out if data lacks symmetry.
#We need to reduce the skewness if there is any to less for skewness coeff
# greater than 0.75

data_num.hist(figsize=(17,20), bins=50, xlabelsize=8, ylabelsize=8);

from scipy.stats import skew
data_num_skew = data_num.apply(lambda x: skew(x.dropna()))
data_num_skew = data_num_skew[data_num_skew > 0.75]

data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])

data_num_skew

data_num.hist(figsize=(15,18), bins=50, xlabelsize=8, ylabelsize=8);

#Mean Nomralisation: Need to change the scale
data_num = ((data_num - data_num.mean())/(data_num.max() - data_num.min()))
data_num.describe()

data_num.hist(figsize=(15,18), bins=50, xlabelsize=8, ylabelsize=8);

#---- Missing Data Handling ----
null_Houseprice = data.isnull().sum()
null_Houseprice = null_Houseprice[null_Houseprice > 0]
null_Houseprice.sort_values(inplace=True)
null_Houseprice.plot.bar()

#Printing total numbers and percentage of missing data
total_missing = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing, percent], axis=1, keys=['Total','Percent'])
missing_data.head()

#Treating the Missing Data
#If no. of values > 260, we drop them, else fill them with the median
data_len = data_num.shape[0]

for col in data_num.columns.values:
    missing_values = data_num[col].isnull().sum()
    print("{} - missing values: {}, ({:.2f}%)".format(col, missing_values,
          missing_values/data_len * 100))
    
    if missing_values > 260:
        print("Dropping Col: {}".format(col))
        data_num = data_num.drop(col, axis = 1)
    
    else:
        print("Filling missing with median in col: {}".format(col))
        data_num = data_num.fillna(data_num[col].median())


#Categorical Variables

data_len = data_cat.shape[0]

for col in data_cat.columns.values:
    missing_values = data_cat[col].isnull().sum()
    
    if missing_values > 50:
        print("Dropping Column: {}".format(col))
        data_cat.drop(col, axis = 1)
    
    else:
        pass

#Checking
data_cat.describe()


#Dummy Encoding for Categorical Variables
data_cat.columns

data_cat_dummies = pd.get_dummies(data_cat, drop_first=True)
data_cat_dummies.head()

#Checking the No. of Numerical and Categorical Features
print("Numerical Features: " + str(len(data_num.columns)))
print("Categorical Features: " + str(len(data_cat_dummies.columns)))

#Concatenating the data
newdata = pd.concat([data_num, data_cat_dummies], axis = 1)

# ---- Exploratory Data Analysis ----

#sns.factorplot("Fireplaces", "SalePrice", data=data_backup, hue="FireplaceQu");

#If Fireplace is missing that means house does not have one
FireplaceQu = data_backup["FireplaceQu"].fillna('None')
pd.crosstab(data_backup.Fireplaces, data_backup.FireplaceQu)

#We can see that price increases with the quality of the house
sns.barplot(data_backup.OverallQual, data_backup.SalePrice)

#Plot for MSZoning
labels = data_backup["MSZoning"].unique()
sizes = data_backup["MSZoning"].value_counts().values
explode = [0.1,0,0,0,0]
percent = 100*sizes/sizes.sum()
labels = ['{0} - {1:1.1f}%'.format(i, j) for i, j in zip(labels, percent)]

colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral', 'blue']
patches, texts = plt.pie(sizes, colors=colors, explode=explode, shadow=True,
                         startangle=90)

plt.legend(patches, labels, loc="best")
plt.title("Zoning Classification")
plt.show() #Most Properties are in a Low Density Zone

#Now compare the zones with the sale price of the properties
sns.violinplot(data_backup.MSZoning, data_backup["SalePrice"])
plt.title("Zones wrt Sale Price")
plt.xlabel("MSZoning")
plt.ylabel("Sale Price");

#Visualise Sale Price per sqft.
price_sqft = data_backup['SalePrice']/data_backup['GrLivArea']
plt.hist(price_sqft, color="blue")
plt.title("Sale Price per Square Foot")
plt.xlabel("Price/sqft")
plt.ylabel("Number of Sales"); #Most sales happened in 110-150 price/sqft range

#Visualise Price/sqft against the age of the property
age_property = data_backup['YrSold'] - data_backup['YearBuilt']
sns.scatterplot(age_property, price_sqft)
plt.ylabel("Price per square foot ($)")
plt.xlabel("Age of the Property (years)")
plt.show() #The older the house, the lower the price of it per sqft.

#Visualise the Aircon and Heating arrangements
sns.stripplot(x="HeatingQC", y="SalePrice", data=data_backup, hue="CentralAir",
              jitter=True, dodge=True)
plt.title("Sale Price vs. Heating Quality")
plt.show() #House with aircon increases the price of the property

#Full bathroom
sns.boxplot(data_backup["FullBath"], data_backup["SalePrice"])
plt.title("Sale Price vs. Full Bathrooms")
plt.show() #The greater the number, the higher the average price of the property


#---Correlation---
data_num.corr()

#Visualising only those with high multicollinearity
correlation = data_num.corr()

mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
plt.figure(figsize=(30,30))
sns.heatmap(correlation[(correlation >= 0.5) | (correlation <= -0.5)],
                        mask=mask,
                        cmap='YlGnBu', vmax = 1.0, vmin = -1.0, linewidths=0.1,
                        annot=True, annot_kws={"size":8}, square=True)
plt.title("Correlation between features")
plt.show()

#---- Linear Regression Modelling ----

#Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(newdata, target_logtrans,
                                                    test_size=0.3,
                                                    random_state=0)

print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

#Building the model
import statsmodels.api as sm

model1 = sm.OLS(y_train, X_train).fit()

#Summarising the model
model1.summary()

#Function for the root mean square error
def rmse(predictions, target):
    
    differences = predictions - target
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    
    return rmse_val

cols = ['Model', 'R-Squared', 'Adj. R-squared', 'RMSE']
model_reports = pd.DataFrame(columns = cols)
prediction1 = model1.predict(X_test)

tmp1 = pd.Series({'Model': 'Base Linear Regression Model',
                  'R-Squared': model1.rsquared,
                  'Adj. R-squared': model1.rsquared_adj,
                  'RMSE': rmse(prediction1, y_test)})

model1_report = model_reports.append(tmp1, ignore_index=True)
model1_report

#Building OLS model with a constant
df_constant = sm.add_constant(newdata)

X_train1, X_test1, y_train1, y_test1 = train_test_split(df_constant,
                                                        target_logtrans,
                                                        test_size=0.3,
                                                        random_state=0)

model2 = sm.OLS(y_train1, X_train1).fit()

model2.summary2()

#Predicting the model on test data
predictions2 = model2.predict(X_test1)

tmp2 = pd.Series({'Model': 'Linear Regression Model with Constant',
                  'R-Squared': model2.rsquared,
                  'Adj. R-squared': model2.rsquared_adj,
                  'RMSE': rmse(predictions2, y_test1)})
    
model2_report = model_reports.append(tmp2, ignore_index = True)
model2_report

#--- Handling Multicollinearity ----

#Variance Inflation Factor (VIF) = 1 / (1 - R^2)
#Statistics to look out for
# VIF = 1: Not Correlated
# 1 <= VIF <= 5: Moderately Correlated
# VIF > 5: Highly Correlated

print("\nVariance Inflation Factor")
colnames = X_train1.columns
for i in np.arange(0, len(colnames)):
    xvars = list(colnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(X_train1[yvar], (X_train1[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print(yvar, round(vif, 3))

#Removing VIF over 100
vif_100 = ['MSSubClass_20','MSSubClass_60','RoofStyle_Gable','RoofStyle_Hip',
           'RoofMatl_CompShg','Exterior1st_MetalSd','Exterior1st_VinylSd',
           'Exterior2nd_VinylSd','GarageQual_TA','GarageCond_TA']

keep_factors = [x for x in X_train1 if x not in vif_100]
print(keep_factors)
X_train2 = X_train1[keep_factors]
X_train2.head()

model3 = sm.OLS(y_train1, X_train2).fit()
model3.summary()

X_test2 = X_test1[keep_factors]
X_test2.head()

predictions3 = model3.predict(X_test2)

tmp3 = pd.Series({'Model': 'LRM after removing VIF above 100',
                  'R-Squared': model3.rsquared,
                  'Adj. R-squared': model3.rsquared_adj,
                  'RMSE': rmse(predictions3, y_test1)})

model3_report = model_reports.append(tmp3, ignore_index = True)
model3_report

#Removing variables with VIF>=10
print("\nVariance Inflation Factor")
colnames = X_train2.columns
for i in np.arange(0, len(colnames)):
    xvars = list(colnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(X_train2[yvar], (X_train2[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print(yvar, round(vif, 3))
    

VIF_10 = ['MSSubClass_20','MSSubClass_60','MSSubClass_90','YearBuilt',
          'MasVnrArea','BsmtFinSF1','BsmtFinSF2','GrLivArea', 'GarageYrBlt',
          'MiscVal','TotalSF','MSSubClass_190','MSSubClass_45',
          'Neighborhood_Gilbert','Neighborhood_IDOTRR','MSSubClass_50',
          'MSSubClass_80', 'MSZoning_FV','MSZoning_RL','MSZoning_RM',
          'Neighborhood_BrkSide','Neighborhood_CollgCr','Neighborhood_Edwards',
          'Neighborhood_NAmes','Neighborhood_OldTown','Neighborhood_Sawyer',
          'Neighborhood_Somerst','Condition2_Norm','HouseStyle_1.5Unf',
          'HouseStyle_2Story','HouseStyle_SLvl','Neighborhood_NWAmes', 
          'Condition2_Feedr','BldgType_2fmCon','Foundation_PConc',
          'KitchenQual_TA', 'HouseStyle_SFoyer','MasVnrType_BrkFace',
          'HouseStyle_1Story','Exterior1st_CemntBd','Exterior1st_HdBoard',
          'Exterior1st_Plywood','Exterior1st_Wd Sdng','Exterior2nd_CmentBd',
          'Exterior2nd_HdBoard','Exterior2nd_Plywood', 'Exterior2nd_Wd Sdng',
          'MasVnrType_None','MasVnrType_Stone', 'ExterQual_Gd','ExterQual_TA',
          'ExterCond_Fa','ExterCond_Gd','ExterCond_TA','BsmtQual_TA',
          'BsmtFinType1_Unf','BsmtFinType2_Unf','Heating_GasA','Heating_GasW',
          'Heating_Grav','GarageType_BuiltIn','SaleType_New',
          'SaleCondition_Partial','GarageType_Attchd','GarageType_Detchd',
          'MiscFeature_Shed','Functional_Typ']
    
keep_factors = [x for x in X_train2 if x not in VIF_10]
X_train3 = X_train2[keep_factors]
keep_factors = [x for x in X_test2 if x not in VIF_10]
X_test3 = X_test2[keep_factors]

model4 = sm.OLS(y_train1, X_train3).fit()

model4.summary()

predictions4 = model4.predict(X_test3)

tmp4 = pd.Series({'Model': 'LRM after removing VIF above 10',
                  'R-Squared': model4.rsquared,
                  'Adj. R-squared': model4.rsquared_adj,
                  'RMSE': rmse(predictions4, y_test1)})

model4_report = model_reports.append(tmp4, ignore_index = True)
model4_report

#Removing Variables with VIF > 5
print ("\nVariance Inflation Factor")
cnames = X_train3.columns
for i in np.arange(0,len(cnames)):
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(X_train3[yvar],(X_train3[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print (yvar,round(vif,3))

VIF_5 = ['LotArea','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea',
         'PoolArea','MSSubClass_75','RoofStyle_Shed','BsmtCond_TA',
         'FireplaceQu_TA','PoolQC_Gd' ,'Condition1_Norm','MoSold_6','MoSold_7']

to_keep = [x for x in X_train3 if x not in VIF_5]
print(to_keep)
X_train4 = X_train3[to_keep]
X_train4.head()

#Building the Model
model5 = sm.OLS(y_train1,X_train4).fit()
model5.summary()

to_keep = [x for x in X_test3 if x not in VIF_5]
X_test4 = X_test3[to_keep]
X_test4.head()

predictions5 = model5.predict(X_test4)

tmp5 = pd.Series({'Model': 'LRM after removing VIF above 5',
                  'R-Squared': model5.rsquared,
                  'Adj. R-squared': model5.rsquared_adj,
                  'RMSE': rmse(predictions5, y_test1)})

model5_report = model_reports.append(tmp5, ignore_index = True)
model5_report

#Removing Variable based on Insignificant Variables using P-value
X = X_train4
y = y_train1

def feature_selection(X, Y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(Y, sm.add_constant((X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(Y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = feature_selection(X, y)

print('resulting features:')
print(result)

df_train= X_train4.filter(['const', 'OverallQual', 'LotFrontage', 'FullBath',
                           'BsmtFullBath', 'CentralAir_Y', 'HalfBath', 
                           'Condition2_PosN', 'Neighborhood_Crawfor', 
                           'LotConfig_CulDSac', 'WoodDeckSF', 'YearRemodAdd', 
                           'Neighborhood_ClearCr', 'Exterior1st_BrkFace', 
                           'ScreenPorch', 'Neighborhood_NridgHt', 
                           'Neighborhood_NoRidge', 'OverallCond_3', 
                           'MSSubClass_30', 'SaleType_WD', 'OverallCond_5', 
                           'BsmtExposure_Gd', 'Neighborhood_StoneBr', 
                           'Functional_Maj2', 'Exterior2nd_Wd Shng', 
                           'FireplaceQu_Gd', 'Exterior1st_BrkComm', 
                           'MSSubClass_160', 'Alley_Pave', 'OpenPorchSF', 
                           'PavedDrive_Y', 'BedroomAbvGr', 'OverallCond_4', 
                           'Heating_OthW', 'Neighborhood_Timber', 
                           'SaleCondition_Normal', 'SaleType_ConLI', 
                           'YrSold_2010', 'BsmtUnfSF', 'LotShape_IR2', 
                           'GarageQual_Fa', 'Utilities_NoSeWa', 'BsmtHalfBath', 
                           'OverallCond_8', 'SaleType_ConLw', 'Fence_MnPrv', 
                           'Fence_GdWo', 'RoofMatl_WdShngl', 'HeatingQC_TA', 
                           'Exterior2nd_Brk Cmn', 'RoofStyle_Gambrel'])

df_test= X_test4.filter(['const', 'OverallQual', 'LotFrontage', 'FullBath', 
                         'BsmtFullBath', 'CentralAir_Y', 'HalfBath', 
                         'Condition2_PosN', 'Neighborhood_Crawfor', 
                         'LotConfig_CulDSac', 'WoodDeckSF', 'YearRemodAdd', 
                         'Neighborhood_ClearCr', 'Exterior1st_BrkFace', 
                         'ScreenPorch', 'Neighborhood_NridgHt', 
                         'Neighborhood_NoRidge', 'OverallCond_3', 
                         'MSSubClass_30', 'SaleType_WD', 'OverallCond_5', 
                         'BsmtExposure_Gd', 'Neighborhood_StoneBr', 
                         'Functional_Maj2', 'Exterior2nd_Wd Shng', 
                         'FireplaceQu_Gd', 'Exterior1st_BrkComm', 
                         'MSSubClass_160', 'Alley_Pave', 'OpenPorchSF', 
                         'PavedDrive_Y', 'BedroomAbvGr', 'OverallCond_4', 
                         'Heating_OthW', 'Neighborhood_Timber', 
                         'SaleCondition_Normal', 'SaleType_ConLI', 
                         'YrSold_2010', 'BsmtUnfSF', 'LotShape_IR2', 
                         'GarageQual_Fa', 'Utilities_NoSeWa', 'BsmtHalfBath', 
                         'OverallCond_8', 'SaleType_ConLw', 'Fence_MnPrv', 
                         'Fence_GdWo', 'RoofMatl_WdShngl', 'HeatingQC_TA', 
                         'Exterior2nd_Brk Cmn', 'RoofStyle_Gambrel'])

df_train.isna().sum().sum(), df_test.isna().sum().sum()

#Building Model after removing insignificant vars using p-value
model6 = sm.OLS(y_train1, df_train).fit()
model6.summary()

#Predicting on Test Data
predictions6 = model6.predict(df_test)

tmp6 = pd.Series({'Model': 'LRM after removing insignificant variables',
                  'R-Squared': model6.rsquared,
                  'Adj. R-squared': model6.rsquared_adj,
                  'RMSE': rmse(predictions6, y_test)})

model6_report = model_reports.append(tmp6, ignore_index = True)
model6_report


#--- Multiple Interactions ---
target = pd.DataFrame(y_train1, columns=['SalePrice'])
data = pd.concat([X_train4, target], axis=1)

# Building Linear Regression model using OLS 
import statsmodels.formula.api as smf
interaction = smf.ols(formula= 'SalePrice ~ OverallQual * YearRemodAdd *  BsmtFullBath', 
                      data = data).fit()
# Note the Swap of X and Y 
interaction.summary()

#---- Visualisation ---- 

'''
1. Residual Plot: Scatterplot of fitted values against residuals
   with a locally weighted scatterplot smoothing (lowess) regression line 
   showing any apparent trend. '''

plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

# fitted values (need a constant term for intercept)
model_fitted_y = model6.fittedvalues

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'SalePrice', data=data, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')


'''
2. QQ Plot: 
    To visualise if residuals are Normally distributed. This plots the z-score
    residuals against the theoretical normal quantiles. 
    
    We are looking for anything outlying far from the digaonal line. '''
    

res = model6.resid
import scipy.stats as stats
fig = sm.qqplot(res, stats.t, fit=True, line='45')
plt.title('Normal Q-Q')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Standardized Residuals');
plt.show() 

'''
3. Scale-Location Plot:
    A residual plot showing their spread, used to assess heteroscedasticity.'''

# normalized residuals
model_norm_residuals = model6.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

plot_lm_3 = plt.figure(3)
plot_lm_3.set_figheight(8)
plot_lm_3.set_figwidth(12)
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

#---- Final Comparison with the Models we built ----
# Comparison of various model
cols = ["Model", "R-Squared", "Adj. R-squared", "RMSE"]
class_model = pd.DataFrame(columns = cols)
class_model = class_model.append([model1_report, model2_report, model3_report,
                                  model4_report, model5_report, model6_report])
class_model

# --- Conclusion ----
''' 
We will choose the model 6 with insignificant variables removed using p-values
with R-Squared: 0.88 and RMSE of 0.1784 as it will perform better with new data
because it makes the fewest assumptions possible out of all models selected.'''

