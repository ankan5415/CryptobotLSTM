# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pymysql
import pprint
import matplotlib.pyplot as plt

#host = 
#port = 
#user = 
#password =
#database = 

#con = pymysql.connect(
    #host=host,
    #port=int(port),
    #user=user,
    #passwd=password,
    #db=database,
    #charset='')




trainset = con.cursor()


trainset.execute("SELECT * FROM Trades.tx WHERE Symbol='ETHBTC' AND time >= '2019-10-14 02:23:30.437000' LIMIT 0, 40000;") 


            
rows = trainset.fetchall()

inputData = {}
inputXvals = []



for row in rows:
    inputXvals.append(row[2])
    if row[2] in inputData:
        inputData[row[2]] += row[3]
        
    else: 
        inputData[row[2]] = row[3]

   
testset = con.cursor()
testset.execute("SELECT * FROM Trades.tx WHERE Symbol='ETHBTC' AND time >= '2019-10-14 02:23:30.437000' LIMIT 40000, 80;") 
vals = testset.fetchall()

testdata = []
for val in vals:
    testdata.append(val[2])

realset = con.cursor()
realset.execute("SELECT * FROM Trades.tx WHERE Symbol='ETHBTC' AND time >= '2019-10-14 02:23:30.437000' LIMIT 40060, 20;") 
val1s = realset.fetchall()

    
realdata = []

for val1 in val1s:
    
    realdata.append(val1[2])

realdata = np.array(realdata)

    





con.close()

    

# Create data
x = inputData.keys()
y = inputData.values()
colors = (255, 255, 255)



def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(x))
    plt.bar(x, y, width = 0.0000005)
    
    plt.xlabel('Price', fontsize=10)
    plt.ylabel('Volume', fontsize=10)
    plt.title('Ethereum - Bitcoin')
    ax = plt.gca()
    ax.set_ylim([0, 2500])
    plt.show()
    
#plot_bar_x()
    
    
    
    
    
    
#--------------------Data Formatting-----------------------#

trainingSet = inputXvals



trainingSet = np.array(inputXvals, dtype='f')

from sklearn import preprocessing

trainingSet = trainingSet.reshape(-1, 1)



training_set = trainingSet
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
for i in range(60, len(trainingSet)):
    X_train.append(training_set_scaled[i-60:i, 0])

X_train = np.array(X_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))




# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    
regressor.fit(X_train, y_train, epochs = 100, batch_size = 300)





inputs = np.array(testdata, dtype = 'f')
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i - 60 : i, 0])
X_test = np.array(X_test)



X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

realdata = np.array(realdata, dtype = 'f')



# Visualising the results
plt.plot(realdata, color = 'red', label = 'Real Cost')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Crypto Price')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('ETHBTC Price')
plt.legend()
plt.show()



percent_error = 0
average_error = 0

k = 0
while k < len(realdata):
    average_error += abs(predicted_stock_price - realdata) / realdata
    
    k += 1
    
average_error = average_error/k
    
print(average_error)


import os
model_json = regressor.to_json()
with open('modelsave.json', 'w') as save_json:
    save_json.write(model_json)
regressor.save_weights("model.h5")
print("saved model to disk!")

