''' Import all the packages and libraries required for the code'''
import os, sys
import random
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import Dense, Embedding, Reshape
from keras.layers import LSTM, RNN
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras import backend as K
from datetime import datetime
from dateutil import parser
import re
import scipy
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy.ndimage.interpolation import shift
from sklearn.metrics import r2_score
from keras.models import model_from_json
import collections 
from tqdm import tqdm
from numpy.linalg import LinAlgError

import warnings
import itertools
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
from pandas.plotting import autocorrelation_plot

''' Change the path of python to current directory '''
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)

''' Read the input data'''
df = pd.read_excel("Data.xlsx")
df['Timeindex'].min(), df['Timeindex'].max() 
df_use = df[['Timeindex','Variable','Log_Variable']]
df_use.dtypes

# Check and fill in the missing values
missing_val_df = df_use.isnull().sum()
missing_val_df = df_use.isna().sum()
df_use = df_use.interpolate()
missing_val_df = df_use.isna().sum()

''' Visualizing Time Series Data'''
plt.figure()
plt.plot(df_use['Timeindex'][0:24*20],df_use['Variable'][0:24*20])
plt.show()

''' Decomposing the series'''
decomposition = sm.tsa.seasonal_decompose(df_use['Variable'], model='additive', freq=1)

fig = decomposition.plot()
plt.show()

''' Decomposing the log transformed time series'''
decomposition_log = sm.tsa.seasonal_decompose(df_use['Log_Variable'], model='additive', freq=1)

fig = decomposition_log.plot()
plt.show()


# check the stationarity of the original time series
result = adfuller(df_use['Variable'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

'''Manually Configure ARIMA Models'''
diff_series = df_use['Variable']
    
plt.figure()
plt.subplot(211)
plot_acf(df_use['Variable'][0:24*100],lags=np.arange(100), ax=plt.gca())
plt.subplot(212)
plot_pacf(df_use['Variable'][0:24*100],lags=np.arange(100), ax=plt.gca())
plt.show()


numlags = 24 # This is used for the LSTM model later on

# Split the data into train and test set
train_size = int(len(df_use) * 0.80)
train, test = df_use[0:(train_size+numlags)], df_use[train_size:]

scaler = StandardScaler()
train_std = scaler.fit_transform(train['Variable'].values.reshape(-1,1))
test_std = scaler.transform(test['Variable'].values.reshape(-1,1))



# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order, interval=24):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.80)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in test.index:
        # difference data
#        diff = difference(history, interval)
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
#        yhat = inverse_difference(history, yhat, interval)
        predictions.append(yhat)
        history.append(test.ix[t,0])
  
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
#   AIC = len(test) * np.log(mse/ len(test)) +  2*(arima_order[0]+arima_order[2]+1)
    rmse = sqrt(mse)
    return rmse


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s RMSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    
# evaluate parameters
p_values = range(0, 5)
d_values = range(0, 1)
q_values = range(1, 5)

warnings.filterwarnings("ignore")
#evaluate_models(df_use['Variable'][0:24*100], p_values, d_values, q_values)

best_arima_order = (4,0,3)

## Predict Using The Best ARIMA model
mod = sm.tsa.statespace.SARIMAX(train['Variable'],
                                order=(4, 0, 3),
                                seasonal_order=(0, 0, 0, 0),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Set the dynamic = False if you don't want use the predictions as inputs
pred = results.get_prediction(start=14078, end=17568, dynamic=True)
pred_ci = pred.conf_int()

plt.figure()
ax = df_use.loc[:,'Variable'].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Time')
ax.set_ylabel('Variable')
plt.legend()
plt.show()


## Deep Learning Based Model and Prediction
def preprocess_data_deeplearning(data, numlags=24):
    data_x_list = []
    data_y_list = []
    for i in range((len(data)-numlags)):
        data_x_list.append([data[(i):((i)+numlags)]])
        data_y_list.append([data[((i)+numlags)]])
    data_x = np.array(data_x_list[0:len(data_x_list)])
    data_y = np.array(data_y_list[0:len(data_y_list)])
    data_x_ml = data_x.reshape((data_x.shape[0], 4, -1,1))
    data_y = data_y.reshape((data_y.shape[0],))
    return data_x_ml, data_y
        
train_x, train_y = preprocess_data_deeplearning(train_std)
test_x, test_y = preprocess_data_deeplearning(test_std)

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )




#define statefule LSTM model
model_name = 'Oct24'
batch_size=1
train_x_lstm = train_x.reshape((len(train_x),24,1))
K.clear_session()
tf.reset_default_graph()
if os.path.exists("{}.json".format(model_name)) and os.path.exists("{}.h5".format(model_name)):
    # load json and create model
    json_file = open("{}.json".format(model_name), 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("{}.h5".format(model_name))
    print("Loaded model {} from disk".format(model_name))    
else:
    model = Sequential()
    model.add(LSTM(64, batch_input_shape=(batch_size, 24, 1), return_sequences=False, stateful=True))
    #model.add(LSTM(64, return_sequences=False, stateful=True))
    model.add(Dense(1))
    #adam_opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.3, amsgrad=False)
    model.compile(optimizer='adam', loss='mse', metrics=[r2_keras])
    # fit model
    callbacks = [ EarlyStopping(patience=50, verbose=1),
                  ReduceLROnPlateau(patience=5, verbose=1),
                  ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
            ]
    for i in range(200):
        print(i)
        model.fit(train_x_lstm, train_y, epochs=1, verbose=1, batch_size=1, validation_split=0.02, callbacks = callbacks,shuffle=False)
        model.reset_states()
    
    model_json = model.to_json()
    model_name = 'Oct24'
    with open("{}.json".format(model_name), "w") as json_file:
                    json_file.write(model_json)
    model.save_weights("{}.h5".format(model_name))

# Predict from the trained model in dynamic mode (i.e. use the model precdiction as input for test set)  
y_train_std_list = []
for i in range(len(train_x)):
    y_train_std = model.predict(train_x_lstm[i,:,:].reshape(1,24,1)).reshape(-1,)
    y_train_std_list.append(y_train_std)
    
num_pred = len(test_y)
train_x_std_last = train_x[(len(train_x)-1),:,:,:].reshape((1,24,1))
train_x_reshape = train_x_std_last.reshape((-1,))
train_x_reshape_shift = shift(train_x_reshape, -1, cval=np.NaN)
y_pred_std_list = []
x_pred_std_list = []
for i in range(num_pred+1):
    y_std = model.predict(train_x_std_last)
    x_pred_std_list.append(train_x_std_last)
    y_pred_std_list.append(y_std)
    train_x_reshape = train_x_std_last.reshape((-1,))
    train_x_reshape_shift = shift(train_x_reshape, -1, cval=y_std)
    train_x_std_last = train_x_reshape_shift.reshape((-1,24,1))
model.reset_states()


# Non Dynamic Prediction
num_pred = len(test_y)
train_x_std_last = train_x[(len(train_x)-1),:,:,:].reshape((1,24,1))
train_x_reshape = train_x_std_last.reshape((-1,))
train_x_reshape_shift = shift(train_x_reshape, -1, cval=np.NaN)
y_pred_std_list_nd = []
x_pred_std_list = []
for i in range(num_pred+1):
    y_std_nd = model.predict(train_x_std_last)
    x_pred_std_list.append(train_x_std_last)
    y_pred_std_list_nd.append(y_std_nd)
    train_x_reshape = train_x_std_last.reshape((-1,))
    if i==0:
        train_x_reshape_shift = shift(train_x_reshape, -1, cval=y_std_nd)
    else:
        train_x_reshape_shift = shift(train_x_reshape, -1, cval=test_y[i-1])
    train_x_std_last = train_x_reshape_shift.reshape((-1,24,1))
model.reset_states()

y_pred = np.array(y_pred_std_list[1:len(y_pred_std_list)]).reshape(-1,)
y_pred_nd = np.array(y_pred_std_list_nd[1:len(y_pred_std_list_nd)]).reshape(-1,)
r2_test = r2_score(test_y,y_pred)

y_train_std = np.array(y_train_std_list).reshape((-1,))
y_train_real = scaler.inverse_transform(y_train_std)
y_pred_real = scaler.inverse_transform(y_pred)
y_pred_real_nd = scaler.inverse_transform(y_pred_nd)
test_y_real = scaler.inverse_transform(test_y)

plt.figure()
plt.plot(train['Timeindex'][0:(len(train)-numlags)], train['Variable'][0:(len(train)-numlags)], 'r', label='Training_Observed')
plt.plot(train['Timeindex'][0:(len(train)-numlags)], y_train_real, 'b', label='Training_Predicted')
plt.plot(test['Timeindex'][0:(len(test)-numlags)], test['Variable'][0:(len(test)-numlags)], 'r', label = 'Test_Observed')
plt.plot(test['Timeindex'][0:(len(test)-numlags)], y_pred_real, 'k', label = 'Test_Predicted')
plt.legend()
plt.show()

plt.figure()
plt.plot(train['Timeindex'][0:(len(train)-numlags)], train['Variable'][0:(len(train)-numlags)], 'r', label='Training_Observed')
plt.plot(train['Timeindex'][0:(len(train)-numlags)], y_train_real, 'b', label='Training_Predicted')
plt.plot(test['Timeindex'][0:(len(test)-numlags)], test['Variable'][0:(len(test)-numlags)], 'r', label = 'Test_Observed')
plt.plot(test['Timeindex'][0:(len(test)-numlags)], y_pred_real_nd, 'k', label = 'Test_Predicted')
plt.legend()
plt.show()


# Analyze forecast errors
residuals = list(test_y-y_pred_nd)
residuals = pd.DataFrame(residuals)
print(residuals.describe())
# plot
plt.figure()
plt.subplot(211)
plot_acf(residuals,lags=np.arange(10), ax=plt.gca())
plt.subplot(212)
plot_pacf(residuals,lags=np.arange(10), ax=plt.gca())
plt.show()

# plot
plt.figure()
plt.subplot(211)
residuals.hist(ax=plt.gca())
plt.subplot(212)
residuals.plot(kind='kde', ax=plt.gca())
plt.show()
