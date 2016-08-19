from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils


#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    mu = np.array([np.mean(X,axis=0)])
    sigma = np.array([np.std(X,axis=0)])
    #X_norm = (X-np.array([np.ones(X.shape[0])]).T.dot(mu))/(sigma)
    X_norm = (X - mu) / sigma
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################
    reglinear_reg = RegularizedLinearReg_SquaredLoss()  
    for i in range(1,num_examples+1):       
        theta1 = reglinear_reg.train(X[0:i],y[0:i],reg,num_iters=1000)         
        error_train[i-1] = (np.dot(np.array([np.dot(X[0:i],theta1)-y[0:i]]),np.array([np.dot(X[0:i],theta1)-y[0:i]]).T)/(2*len(y[0:i])))[0,0]
        error_val[i-1] = (np.dot(np.array([np.dot(Xval,theta1)-yval]),np.array([np.dot(Xval,theta1)-yval]).T)/(2*len(yval)))[0,0]
    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):
  
    reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = np.zeros((len(reg_vec),))
    error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################  
    reglinear_reg = RegularizedLinearReg_SquaredLoss()       
    for i in range(len(reg_vec)):
        theta_local = reglinear_reg.train(X,y,reg = reg_vec[i],num_iters=1000)
        error_train[i] = (np.dot(np.array([np.dot(X,theta_local)-y]),np.array([np.dot(X,theta_local)-y]).T)/(2*len(y)))[0,0]
        error_val[i] = (np.dot(np.array([np.dot(Xval,theta_local)-yval]),np.array([np.dot(Xval,theta_local)-yval]).T)/(2*len(yval)))[0,0]    
    return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################
    reglinear_reg2 = RegularizedLinearReg_SquaredLoss()  
    for i in range(2,num_examples+1): 
        for j in range(50):
            index_rd_train = np.array(random.sample(range(0,len(y)),i))  
            index_rd_val = np.array(random.sample(range(0,len(yval)),i)) 
            
            theta1 = reglinear_reg2.train(X[index_rd_train],y[index_rd_train],reg,num_iters=1000)  
                  
            error_train[i-1] += (np.dot(np.array([np.dot(X[index_rd_train],theta1)-y[index_rd_train]]),np.array([np.dot(X[index_rd_train],theta1)-y[index_rd_train]]).T)/(2*50*len(y[index_rd_train])))[0,0]
            error_val[i-1] += (np.dot(np.array([np.dot(Xval[index_rd_val],theta1)-yval[index_rd_val]]),np.array([np.dot(Xval[index_rd_val],theta1)-yval[index_rd_val]]).T)/(2*50*len(yval[index_rd_val])))[0,0]
    
   
        
    
    
#    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval









