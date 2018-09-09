import numpy as np
import matplotlib.pyplot as plt
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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
col_names=['DepTime','Distance','Interchange1time','DestinationTime']
features=pd.read_csv(r'C:\Users\sparamas\Downloads\flight_delays_train.csv',header=0,names=col_names)
df=pd.DataFrame(features)

# print(df.values[0])

def load_data(features,seq_len):
    no_of_features=len(features.columns)
    print(no_of_features)
    data=features.as_matrix()
    # print(data)
    sequence_length=seq_len+1
    # print(sequence_length)
    result=[]
    # print(result)
    for index in range(len(data)-sequence_length):
        result.append(data[index: index+sequence_length])
    result=np.array(result)
    # print(result.shape)
    row=round(0.9*result.shape[0])
    # print(row)
    train=result[:int(row),:]
    # print(train.shape)
    x_train=train[:,-1,:3]
    # print(x_train)
    y_train=train[:,-1,-1]
    # print(y_train)
    x_test=result[int(row):,-1,:3]
    # print(x_test)
    y_test=result[int(row):,-1][-1]
    # print(y_train)
    return [x_train,y_train,x_test,y_test]
window=5

x_train,y_train,x_test,y_test=load_data(features,window)
# x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
model=Sequential()
print(len(y_train))
print(x_train.shape[0])
print(y_train.shape[0])
model.add(LSTM(256,input_shape=(3,1)))
model.add(Dense(1))
model.compile(optimizer='ADAM',loss='mse',metrics=['acc'])
x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],1))
history= model.fit(x_train,y_train,epochs=5,shuffle=False)
pred=model.predict(x_test)
score=model.evaluate(x_train,y_train)
print("accuracy:",score[1])
print(pred)
