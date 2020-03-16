import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# Importing data set
dataset = pd.read_csv('Churn_Modelling.csv')
dataset_variables = dataset.iloc[:, 3:-1].values
dataset_results = dataset.iloc[:, -1].values

# Encoding categorical data
labelencoder_dataset_variables = LabelEncoder()
dataset_variables[:, 1] = labelencoder_dataset_variables.fit_transform(dataset_variables[:, 1])
labelencoder_dataset_variables = LabelEncoder()
dataset_variables[:, 2] = labelencoder_dataset_variables.fit_transform(dataset_variables[:, 2])

# Creating a column transformer => for state 
ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), [1])],  # the column numbers I want to apply this to
    remainder='passthrough') # This leaves the rest of my columns in place

dataset_variables = ct.fit_transform(dataset_variables)

# Removing the dummy variable trap for countries 
dataset_variables  = dataset_variables[:, 1:]

# Splitting the data to create train the model, 80: 20 rule, training set contains 80% of the data, 20% to test the data 
dataset_variables_train, dataset_variables_test, dataset_results_train, dataset_results_test = train_test_split(dataset_variables, dataset_results, test_size= 0.2, random_state =0)

# Using feature scaling for variable dependence - so a specific variable is not highly dependent on the results
sc = StandardScaler()
# Fitting the standardized value for the trained anf test values
dataset_variables_train = sc.fit_transform(dataset_variables_train)
dataset_variables_test = sc.transform(dataset_variables_test)

# Initialise the ANN :)
classifier = Sequential()

# Adding input layer and the first hidden layer, for 6 layers (len(arr)// 2), with softmax graph, with an input of 11 values - dataset_variables
classifier.add(Dense(6, activation='softmax', input_dim= 11))

# Adding our second hidden layer - second hidden layer with a softmax graph
classifier.add(Dense(6, activation='softmax'))

# Adding the ouput layer - output variable with a sigmoid variable, showing the percentage/ likelihood someone will leave or stay with the bank
classifier.add(Dense(1, activation='sigmoid'))

# Compling the ANN - 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set - Testing the ANN, by 100 epochs 
classifier.fit(dataset_variables_train, dataset_results_train, batch_size=10, nb_epoch=100)

# Predicting the Test set results
dataset_results_pred = classifier.predict(dataset_variables_test)

# Presenting the predicted variables for the test results 
dataset_results_pred = [True if (i > .5) else False for i in dataset_results_pred]