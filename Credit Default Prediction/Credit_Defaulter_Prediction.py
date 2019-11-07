# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:09:45 2019

@author: Preet

The dataset is taken from the TW banking system.
Goal: Predict defaulters with certain known parameters.
"""
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

#Importing the Dataset

#data = pd.read_csv("Path_here\\creditcarddefault.csv")
data = pd.read_csv("creditcarddefault.csv")

#Some exploration
data.describe() 
data.info() #Zero null values

data.shape #30,000 x 25

data.head()

#Our target is in the last column so we need to segregate it.
#Also we do not need the ID column.

X = data.iloc[:, 1:24].values
y = data.iloc[:, 24].values

X.shape #(30000, 23)
y.shape #(30000, )

#Dataset is clean so split into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state = 111)

#Check for split
print(X_train.shape)
print(X_test.shape)

print("\n", y_train.shape)
print(y_test.shape)
#Successful

#Our Data has various scales of different ranges: Regularize the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Check
X_train[:2]

#Build our Model
import keras
from keras.models import Sequential
from keras.layers import Dense

NN_model = Sequential()

#We have 23 input dims therefore to calculate no. of nodes:
#No. of nodes = (No. of Input Notes + 1)/2 = 23+1/2 = 12 nodes
#This is to have an optimum error
#We will have random weights therefore have a uniformly distributed random weights.
#As a result our initialisation will happen with uniformly random weights

#1st Hidden Layer
NN_model.add(Dense(input_dim = 23, output_dim = 12, init = "uniform",
                   activation="relu"))

#2nd Hidden Layer
#As input dim is the output dim from previous layer, we need not specify it here.
NN_model.add(Dense(output_dim = 12, init = "uniform", activation="relu"))

#Output Layer
#Since we have a binary target var, we will use a sigmoid activation
NN_model.add(Dense(output_dim = 1, init="uniform", activation="sigmoid"))


#Compile the NN.
#Again since our target is binary, we will use a binary_crossentropy
#for loss function

NN_model.compile(optimizer="sgd", loss="binary_crossentropy",
                 metrics=["accuracy"])

#Fit our model and run it for 100 epochs
NN_model.fit(X_train, y_train, batch_size=10, epochs=100)


#Predicting
y_pred = NN_model.predict(X_test)

pred_vals = (y_pred > 0.5)

#Creating a Confusion Matrix
target_names = [ 'no', 'yes']
# code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred_vals)

plt.figure()
plot_confusion_matrix(cm, classes=target_names, normalize=False)
plt.show()

#Evaluating our model
score = NN_model.evaluate(X_test, y_test)
print("\nAccuracy: {:.2f}%".format(score[1]*100))







