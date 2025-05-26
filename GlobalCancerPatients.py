# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:50:03 2025

@author: pchri
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn as skl

from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder

#survival analysis
from lifelines import CoxPHFitter

cancer = pd.read_csv(r'C:\Users\pchri\Downloads\global_cancer_patients_2015_2024.csv')
print(cancer)

#converting cancer stages to numbers
le = LabelEncoder()
cancer['Cancer_Stage'] = le.fit_transform(cancer['Cancer_Stage'])
cancer['Cancer_Stage']

del cancer['Patient_ID']
del cancer['Gender']
del cancer['Country_Region']
del cancer['Cancer_Type']

df = pd.DataFrame(cancer)

cph = CoxPHFitter()
cph.fit(df, 'Cancer_Stage', event_col='Target_Severity_Score')
cph.print_summary()
cph.predict_median(df)

print(cancer.shape)
print(cancer)

#descriptions
print(cancer.describe())

#class distribution
print(cancer.groupby('Cancer_Stage').size())

#Split-out validation dataset
X = cancer.drop('Target_Severity_Score', axis=1)
Y = cancer['Target_Severity_Score']
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=(seed))

lr =  LinearRegression()
m1 = lr.fit(X_train, Y_train)
a = m1.score(X_validation, Y_validation)

dtr = DecisionTreeRegressor()
m2 = dtr.fit(X_train, Y_train)
b = m2.score(X_validation, Y_validation)

knr =  KNeighborsRegressor()
m3 = knr.fit(X_train, Y_train)
c = m3.score(X_validation, Y_validation)

svr = SVR()
m4 = svr.fit(X_train, Y_train)
d = m4.score(X_validation, Y_validation)

rf =  RandomForestRegressor()
m5 = rf.fit(X_train, Y_train)
e = m5.score(X_validation, Y_validation)

gb =  GradientBoostingRegressor()
m6 = gb.fit(X_train, Y_train)
f = m6.score(X_validation, Y_validation)

et =  ExtraTreesRegressor()
m7 = et.fit(X_train, Y_train)
g = m7.score(X_validation, Y_validation)

ab =  AdaBoostRegressor()
m8 = ab.fit(X_train, Y_train)
h = m8.score(X_validation, Y_validation)

vector = [a,b,c,d,e,f,g,h]
print(vector)

predictions = m1.predict(X_validation)
print(predictions)
