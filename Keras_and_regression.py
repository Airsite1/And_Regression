import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt  
from keras.models import Sequential
from keras.layers import Dense, Activation

#Training Data
infile = open('../And_operator.txt')
x=[]
y=[]
for line in infile:
   arr=line.split(',')
   x.append([int(arr[0]),int(arr[1])])
   y.append (int(arr[2]))
print(y,x)
y=np.array(y,dtype='float32')
x=np.array(x,dtype='float32')

#Build Model
model = Sequential()
model.add(Dense(3,input_shape=(2,)))
model.add(Activation("linear"))
model.add(Dense(2))
model.add(Activation("linear"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

#Compiling
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#Training
model.fit(x, y, epochs=100, batch_size=4, verbose=0)

#Test
model.evaluate(x,y, batch_size=4)

model.summary()

