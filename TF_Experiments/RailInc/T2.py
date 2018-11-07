import numpy as np
from matplotlib import pyplot
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

dataset=pd.read_csv(r'C:\Users\sparamas\Downloads\flight_delays_train.csv',header=0)
dataset.columns=['DepTime','Interchange1','Interchange2','DestinationTime']
values=dataset.values
pyplot.plot(dataset)
encoder=LabelEncoder()
values[:,3]=encoder.fit_transform(values[:,3])
print(values[:,3])
values=values.astype('float32')
scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(values)

# print(scaled)

train=scaled[:round(len(values)*0.9),:]
# print(train)
test=scaled[:round(len(values)*0.1):,:]
# print(test)

x_train,y_train=train[:,:-1],train[:,-1]
# print(y_train)
x_test,y_test=test[:,:-1],test[:,-1]
print(y_test)

x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],1))

model = Sequential()
model.add(LSTM(50, input_shape=(3,1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(x_train, y_train, epochs=1, batch_size=72, validation_data=(x_test, y_test), verbose=1, shuffle=False)
score=model.evaluate(x_test,y_test)
print("accuracy:",score[0])
pred=model.predict(x_test)



# regr=linear_model.LinearRegression()
# regr.fit(x_train,y_train)
# pred=regr.predict(x_test)

# print('Coefficients: \n', regr.coef_)
# The mean squared error

# print('pred', pred[1])
# print('y',y_test[0])