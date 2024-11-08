import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression

def MSE(y_true, y_pred): #same with loss
    return mean_squared_error(y_true, y_pred)


def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def R2_adjusted(y_true, y_pred, y_train, X_train): # Adjusted R-Squared
    return 1 - (1-R2(y_true, y_pred)) * (len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)  
    

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def PERSON(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def P_VALUE(y_true, y_pred):
    return pearsonr(y_true, y_pred)[1]

def C_INDEX(y_true, y_pred):
    return concordance_index(y_true, y_pred)


def SD(y_true, y_pred):
    y_pred = y_pred.reshape((-1,1))
    lr = LinearRegression().fit(y_pred,y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))
