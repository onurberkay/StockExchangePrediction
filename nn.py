
import os 
os.environ["DML_VISIBLE_DEVICES"] = "0" 
import sklearn
print (sklearn.__version__)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import add_dummy_feature

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np
import yfinance as yf

import tensorflow as tf 
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU

from tensorflow.keras import regularizers


#tf.debugging.set_log_device_placement(True)
def ma(now,i,hist):
    total = 0
    for t in range(now-i,now):
        total += hist["Close"][t]
    total/=i
    return total

stocks = ["ENJSA.IS","CEMTS.IS","TOASO.IS","IPEKE.IS","TTRAK.IS","SASA.IS",
          "TTKOM.IS","EREGL.IS","OTKAR.IS","TUPRS.IS","EGSER.IS","SISE.IS",
          "GARAN.IS","ARCLK.IS","BRISA.IS","CCOLA.IS","BIMAS.IS","DOHOL.IS",
          "MAKTK.IS","SAHOL.IS","ZOREN.IS","KOZAL.IS","TAVHL.IS","YATAS.IS",
          "ODAS.IS","KRDMD.IS","YKBNK.IS","VESTL.IS","THYAO.IS","FROTO.IS"
          ]
stockLast = []


X = []
y = []
for stock in stocks:
    print(stock)
    msft = yf.Ticker(stock)
    hist = msft.history(period="10y",interval="1d")
    print(hist)
    temp = []
    for i in range(len(hist["Close"])-10,len(hist["Close"])-1):
        if hist["Close"][i-1]==0:
            temp.append(0)
        else :
            temp.append((hist["Close"][i]-hist["Close"][i-1])/hist["Close"][i-1])
        
        if hist["Volume"][i-1]==0:
            temp.append(0)
        else :
            temp.append((hist["Volume"][i]-hist["Volume"][i-1])/hist["Volume"][i-1])
        

        
        
    stockLast.append(temp)

    
    for t in range(1000,len(hist["Close"])-11):
        temp = []
        for i in range(1,10):
            if hist["Close"][t+i-1]==0:
                temp.append(0)
            else :
                temp.append((hist["Close"][t+i]-hist["Close"][t+i-1])/hist["Close"][t+i-1])
            
            if hist["Volume"][t+i-1]==0:
                temp.append(0)
            else :
                temp.append((hist["Volume"][t+i]-hist["Volume"][t+i-1])/hist["Volume"][t+i-1])

        yTemp = (hist["Close"][t+10]-hist["Close"][t+9])/hist["Close"][t+9]
        
        y.append(yTemp)
        X.append(temp)
batch_size = 30

print(len(X))
X= np.array(X)
y = np.array(y)
#X = add_dummy_feature(X)
X = X.astype('float32')
y = y.astype('float32')
#y = y.reshape(len(y),1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle=False,random_state=1)
X_train = X_train[0:batch_size*((int)(len(X_train)/batch_size))]
y_train = y_train[0:batch_size*((int)(len(X_train)/batch_size))]

X_test = X_test[0:batch_size*((int)(len(X_test)/batch_size))]
y_test = y_test[0:batch_size*((int)(len(X_test)/batch_size))]

print(len(X_train))

scaler.fit(X_train)  
X_train = scaler.transform(X_train) 
X_train = X_train.reshape(len(X_train),18,1)

print(len(X_train))
# apply same transformation to test data
X_test = scaler.transform(X_test) 
X_test = X_test.reshape(len(X_test),18,1)

#regr =MLPRegressor(verbose=True,shuffle=True,batch_size=1000,tol=0.00000000001,alpha=0.0001,beta_2=0.999999999,n_iter_no_change=10000,activation="tanh",learning_rate_init=0.00001,learning_rate="adaptive",warm_start=False,solver ="adam",hidden_layer_sizes=(1000,1000,1000),random_state=1, max_iter=50000,max_fun=150000).fit(X_train, y_train)

#network = Input(199) >>Relu(50) >> Relu(50) >> Relu(50) >>Relu(1)
#regr = algorithms.IRPROPPlus(network,verbose=True,increase_factor=1.2,decrease_factor=0.9,shuffle_data=True,minstep=0.000000001,step=0.001,error= "mse")

#regr.train(X_train, y_train)
look_back = 18
model = Sequential()
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, batch_input_shape=(batch_size, look_back,1), stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001),return_sequences=True, stateful=True))
model.add(GRU(36,kernel_regularizer=regularizers.l2(0.0000000000001)))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000000001),metrics=['accuracy'])
for i in range(50):
	model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
	model.reset_states()



preds = model.predict(X_test,batch_size=30)
model.reset_states()

score =0
count =0
scoreB =0
countB=0
scoreC =0
countC=0
scoreD =0
countD=0
for r in range(len(y_test)):
    if abs(preds[r])>0.03:
        count+=1
        if abs(y_test[r]-preds[r])<0.02 :
            score +=1
    if preds[r]>0.01:
        countB+=1
        if y_test[r]>0.01 :
            scoreB +=1
    if preds[r]>0.02:
        countC+=1
        if y_test[r]>0.02 :
            scoreC +=1
    if preds[r]>0.03:
        countD+=1
        if y_test[r]>0.03 :
            scoreD +=1
if count !=0:
    print("Score 0.02 SAPMA 0.02>")
    print(score/count)


print("ScoreB 0.01> 0.01>")
print(countB)
if countB!=0:
    print(scoreB/countB)


print("ScoreC 0.02> 0.02>")
print(countC)
if countC!=0:
    print(scoreC/countC)

print("ScoreD 0.03> 0.03>")
print(countD)
if countD!=0:
    print(scoreD/countD)
#stockLast = add_dummy_feature(stockLast)
lastPredict = model.predict(stockLast,batch_size=30)
print("lastValues")

print("predcount")
print(len(preds))
print(len(y_test))
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(preds, label='predict')
pyplot.plot(y_test, label='real')
pyplot.legend()
pyplot.show()
for i in range(len(lastPredict)):
    print(stocks[i])
    print(lastPredict[i])

