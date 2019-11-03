# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:58:25 2019

@author: Preet
"""

#Importing the usual DS libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Data
#data_raw = pd.read_csv("Enter File Path here\German_Credit_data.csv")
data_raw = pd.read_csv("German_Credit_data.csv")

data_raw.head()

#--- Data Preparation ---
data_raw.shape

data_raw.describe()
data_raw.info() #No Missing Values - Woop!

#--- Exploratory Data Analysis ---
data_raw.hist(figsize=(15, 15)) 
#Age_in_Years and Credit Amount are not Normally Distrbuted

#Number of levels in each variable
data_raw.nunique()

#Plotting the Distribution for the Age
#Split data into good and poor
data_gd = data_raw[data_raw["Creditability"] == 1]
data_poor = data_raw[data_raw["Creditability"] == 0]

#Compare the two
fig, ax =plt.subplots(nrows = 2, figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = .8)

graph1 = sns.distplot(data_gd["Age_in_years"], ax=ax[0], color="g")
graph1 = sns.distplot(data_poor["Age_in_years"], ax=ax[0], color="r")
graph1.set_title("Age Distribution", fontsize=15)
graph1.set_xlabel("Age")
graph1.set_ylabel("Frequency")

graph2 = sns.countplot(x="Age_in_years", data=data_raw, palette="hls",
                       ax=ax[1], hue="Creditability")

graph2.set_title("Age Counting by Creditability", fontsize = 15)
graph2.set_xlabel("Age")
graph2.set_ylabel("Count")
plt.show()

#Deriving the Distribution of the Class
print(data_raw.groupby("Creditability").size())
#30% belongs to 0 credibility and 70% to credibility level 1

#Confirming the above
total_len = len(data_raw["Creditability"])
percentage_labels = (data_raw["Creditability"].value_counts()/total_len)*100
percentage_labels

#Visualise for sakes of looking at it.
sns.set()
sns.countplot(data_raw["Creditability"]).set_title("Data Distribution")
ax = plt.gca()
for patch in ax.patches:
    height = patch.get_height()
    ax.text(patch.get_x() + patch.get_width()/2.0, 
            height + 2, '{:.2f}%'.format((height/total_len)*100), 
            fontsize = 12, ha = 'center', va = 'bottom')

sns.set(font_scale=1.5)
ax.set_xlabel("Labels for Creditability attribute")
ax.set_ylabel("Number of Records")
plt.show()

#The Usual Correlation check
data_raw.corr()

data_corr = data_raw.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(data_corr[(data_corr >=0.5) | (data_corr <= -0.5)], vmax = 0.8,
                      linewidths=0.01, square=True, annot=True, cmap="YlGnBu",
                      linecolor="white")
plt.title("Correlation between features") 
plt.show()
#Credit Amount and Duration_of_Credit_Month are highly correlated.
#We will treat the multicolinear variables later.

#--- Logistic Regression ---
#We try and apply this since target variable is Binary.

predictor = data_raw.iloc[:, data_raw.columns != "Creditability"]
target = data_raw.iloc[:, data_raw.columns == "Creditability"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictor,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=0)

#Check for equal splits 
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)

#Building the Model (finally!)
import statsmodels.api as sm
#Build Logit Model
logit = sm.Logit(y_train, X_train)
#Fit the model
model1 = logit.fit()
#Print the summary
model1.summary()

#Predict values using Test data
y_pred = model1.predict(X_test)

pred_df = pd.DataFrame(y_pred)
pred_df.head()

#Assigning the class 0 and 1 based on a threshold of 0.5
pred_df["Predicted_Class"] = np.where(pred_df[0] >= 0.5, 1, 0)
pred_df.head()

#Check for Model Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred_df["Predicted_Class"])) #74% Accuracy (not bad)

#--- Evaluating the Model ----
#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, pred_df["Predicted_Class"]).ravel()
print(confusion_mat)
#[37 63 15 185]: 37 and 185 are correct predictions and 63 and 15 being incorrect.


#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_df["Predicted_Class"]))

#ROC Curve: A plot of True Positive Rate vs. False Positive Rate.
y_pred_prob = model1.predict(X_test)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

#Plot ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, pred_df["Predicted_Class"]) #Area under curve: 0.6475

#Cohen's Kappa Score
#A metric that compares an Observed accuracy with an Expected Accuracy.

cols = ["Model", "R-Squared", "ROC Score", "Precision Score", "Recall Score",
        "Accuracy Score", "Kappa Score"]

model_report = pd.DataFrame(columns=cols)

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, cohen_kappa_score

temp1 = ({"Model": "Logistic Regression Base Model",
          "R-Squared": model1.prsquared,
          "ROC Score": roc_auc_score(y_test, pred_df["Predicted_Class"]),
          "Precision Score": precision_score(y_test, pred_df["Predicted_Class"]),
          "Recall Score": recall_score(y_test, pred_df["Predicted_Class"]),
          "Accuracy Score": accuracy_score(y_test, pred_df["Predicted_Class"]),
          "Kappa Score": cohen_kappa_score(y_test, pred_df["Predicted_Class"])})
    
    
model1_report = model_report.append(temp1, ignore_index=True)
model1_report


#---- Now model post removing the Credit_Amount Variable ----
#Since it had multicollinearity
X_train.columns
X_train.shape
X_train1 = X_train.drop(columns = ["Credit_Amount"])
X_train1.shape

X_test.columns
X_test1 = X_test.drop(columns = ["Credit_Amount"])
X_test.shape
X_test1.shape

#Build the model
logit2 = sm.Logit(y_train, X_train1)
model2 = logit2.fit()
model2.summary()

#Predict the model
y_pred2 = model2.predict(X_test1)

pred_df2 = pd.DataFrame(y_pred2)
pred_df2.head()

pred_df2["Predicted_Class"] = np.where(pred_df2[0] >= 0.5, 1, 0) 
pred_df2.head()
pred_df2.tail()

#Confusion Matrix
confusion_mat = confusion_matrix(y_test, pred_df2["Predicted_Class"]).ravel()
confusion_mat #39,184: Correct Predictions and 61,16: Incorrect

print(classification_report(y_test, pred_df2["Predicted_Class"]))
#Since the F1 score tells the accuracy of the classifier in classifying the
#data points in that particular class, our accuracy of the model is 72%

#ROC Curve
y_pred_prob = model2.predict(X_test1)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

#Plotting the Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

temp2 = ({"Model": "LR model post removing Credit_Amount variable",
          "R-Squared": model2.prsquared,
          "ROC Score": roc_auc_score(y_test, pred_df2["Predicted_Class"]),
          "Precision Score": precision_score(y_test, pred_df2["Predicted_Class"]),
          "Recall Score": recall_score(y_test, pred_df2["Predicted_Class"]),
          "Accuracy Score": accuracy_score(y_test, pred_df2["Predicted_Class"]),
          "Kappa Score": cohen_kappa_score(y_test, pred_df2["Predicted_Class"])})

model2_report = model_report.append(temp2, ignore_index=True)
model2_report

#---- Building Model post removing "Duration_of_Credit_Month" variable ----
#This also had multicollinearity with Credit_Amount

X_train.columns
X_train2 = X_train.drop(columns = "Duration_of_Credit_month")
X_test2 = X_test.drop(columns = "Duration_of_Credit_month")

#Check
print("X_train shape: ", X_train.shape)
print("X_train2 shape: ", X_train2.shape)
print("X_test shape:", X_test.shape)
print("X_test2 shape: ", X_test2.shape)
#Successful


#Build the Model
logit3 = sm.Logit(y_train, X_train2)
model3 = logit3.fit()
model3.summary()

#Predicting model on test data
y_pred3 = model3.predict(X_test2)
pred_df3 = pd.DataFrame(y_pred3)
pred_df3.head()

#Assigning 0 and 1 based on a threshold of 0.5
pred_df3["Predicted_Class"] = np.where(pred_df3[0] >= 0.5, 1, 0)
pred_df3.head()

#Evaluating the Model
confusion_mat = confusion_matrix(y_test, pred_df3["Predicted_Class"])
confusion_mat #38 and 186 Correct Predictions, 62 and 14 Incorrect Predictions.

#F1-Score
print(classification_report(y_test, pred_df3["Predicted_Class"]))
#72% Accuracy on Weigted Average

#ROC Curve for this model
y_pred_prob = model3.predict(X_test2)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

#Plotting the Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Duration of Credit Month Variable removed")
plt.show()

#Capturing all Metrics
temp3 = ({"Model": "LR model after Removing Duration of Credit",
          "R-Squared": model3.prsquared,
          "ROC Score": roc_auc_score(y_test, pred_df3["Predicted_Class"]),
          "Precision Score": precision_score(y_test, pred_df3["Predicted_Class"]),
          "Recall Score": recall_score(y_test, pred_df3["Predicted_Class"]),
          "Accuracy Score": accuracy_score(y_test, pred_df3["Predicted_Class"]),
          "Kappa Score": cohen_kappa_score(y_test, pred_df3["Predicted_Class"])})

model3_report = model_report.append(temp3, ignore_index=True)
model3_report

#Building final Dataframe consisting of all models built
model_LogReg = pd.DataFrame(columns = cols) #Using the cols we defined earlier
model_LogReg = model_LogReg.append([model1_report, model2_report,
                                    model3_report], ignore_index=True)

model_LogReg
'''
We can see that though there is no difference in ROC Score between 
LR without Credit_Amount and Duration_of_Credit_month variable, the model did
marginally improve from the base model with the variables included.  

'''
#--- Treating Multicollinearity and Building a Logistic Regression Model ---

#We will use the Variance Inflation Factor (VIF)
print("\nVIF: Removing Vars > VIF of 5")
col_names = X_train2.columns
for i in np.arange(0, len(col_names)):
    x_var = list(col_names)
    y_var = x_var.pop(i)
    mod = sm.OLS(X_train2[y_var], (X_train2[x_var]))
    result = mod.fit()
    vif = 1/(1 - result.rsquared)
    print(y_var, round(vif, 4))
    
#Removing Variables > VIF of 10
#Columns picked from the output of above
VIF10 = ["Years_of_Present_Employment", "Sex_&_Marital_Status", "Age_in_years",
         "Concurrent_Credits", "Housing", "Occupation", "No_of_dependents",
         "Telephone", "Foreign_Worker"]

keep_var = [x for x in X_train2 if x not in VIF10]

X_train_VIF = X_train2[keep_var]
print(X_train_VIF.shape) #700x10

keep_var_test = [x for x in X_test2 if x not in VIF10]
X_test_VIF = X_test2[keep_var_test]
print(X_test_VIF.shape) #300x10 

#Building the Model
logit4 = sm.Logit(y_train, X_train_VIF)

model4 = logit4.fit()
model4.summary()

y_pred4 = model4.predict(X_test_VIF)

pred_df4 = pd.DataFrame(y_pred4)
pred_df4.head()

#Assigning 0 and 1 with a threshold of 0.5
pred_df4["Predicted_Class"] = np.where(pred_df4[0] >= 0.5, 1, 0)
pred_df4.head()

#Check Accuracy of our model
print(round(accuracy_score(y_test, pred_df4["Predicted_Class"]), 3) * 100)#75.7% 
















