# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:06:53 2016

@author: JingGuo
"""
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import plot_utils
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss

## reading data
print 'Reading data ...'
bdata = load_boston()
df = pd.DataFrame(data = bdata.data, columns = bdata.feature_names)
X = df.values
y = bdata.target

## spliting data into train, validation, test set
from sklearn.cross_validation import train_test_split
X1, X_test, y1, y_test = train_test_split(X,y,test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X1,y1,test_size=0.5)


# append a column of ones to matrix X
XX_train = np.vstack([np.ones((X_train.shape[0],)),X_train.T]).T
XX_val = np.vstack([np.ones((X_val.shape[0],)),X_val.T]).T
XX_test = np.vstack([np.ones((X_test.shape[0],)),X_test.T]).T

## find the theta on the train set using gradient descent methods when lamda =0
reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
theta_opt0 = reglinear_reg1.train(XX_train,y_train,reg=0.0,num_iters=5000)
print 'the theta is ', theta_opt0

## find the best achievable error on the test set using the theta computed above when lamda =0
error_test = (np.dot(np.array([np.dot(XX_test,theta_opt0)-y_test]),np.array([np.dot(XX_test,theta_opt0)-y_test]).T)/(2*len(y_test)))[0,0]
print 'the best achievable error on the test set is ', error_test