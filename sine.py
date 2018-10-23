import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

### Generate training points



### process y values between 0 and 1
### x = +-2pi = +-6.28318530718 


### process output [0,1] to [-1,1] values
### y = +-1 





### Network

def build_model():
	model = keras.Sequential([
	    keras.layers.Dense(100, activation=tf.nn.relu),
	    keras.layers.Dense(100, activation=tf.nn.relu),
	    keras.layers.Dense(1)
	])

	#model.compile(optimizer=tf.train.AdamOptimizer(), 
    #          loss='sparse_categorical_crossentropy',
    #          metrics=['accuracy'])

	optimizer = tf.train.RMSPropOptimizer(0.001)

	model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae'])
	return model;

### Execute

## line
##
#########################################

linex = np.linspace(-1,0.99,300)
liney = linex

# Shuffle the training set
order = np.argsort(np.random.random(liney.shape))
linex = linex[order]
liney = liney[order]

squish_linex = (linex+1)/2
squish_liney = squish_linex

squish_linex = np.reshape(squish_linex,(-1,1))
squish_liney = np.reshape(squish_liney,(-1,1))

plt.plot(linex, liney, 'g-')

#print(squish_linex)
#print(squish_liney)

model = build_model()
model.fit(squish_linex, squish_liney, epochs=10) #probar con mas o menos epochs


test_linex = np.linspace(-1,0.99,10)
test_liney = test_linex
squish_test_linex = (test_linex+1)/2
squish_test_liney = squish_test_linex

squish_test_linex = np.reshape(squish_test_linex,(-1,1))
squish_test_liney = np.reshape(squish_test_liney,(-1,1))

[loss, mae] = model.evaluate(squish_test_linex, squish_test_liney, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(squish_test_linex).flatten()*2-1
plt.scatter(test_linex,test_predictions)


## sine
##
#########################################


b = 5
x = np.linspace(-2*np.pi, 1.99*np.pi, 4000) #genera los puntos de prueba, aqui genera 4000
y = np.sin(x+b)
plt.plot(x, y)
plt.axis('tight')


# Shuffle the training set
order = np.argsort(np.random.random(y.shape))
x = x[order]
y = y[order]

squish_x = (x+np.pi*2)/(np.pi*4)
squish_y = ((y+1)/2)

squish_x = np.reshape(squish_x,(-1,1))
squish_y = np.reshape(squish_y,(-1,1))

# build model
model = build_model()
model.fit(squish_x, squish_y, epochs=100) #probar con mas o menos epochs


# test data
testx = np.linspace(-2*np.pi, 1.99*np.pi, 30)
testy = np.sin(testx+b)
squish_testx = (testx+np.pi*2)/(np.pi*4)
squish_testy = ((testy+1)/2)

squish_testx = np.reshape(squish_testx,(-1,1))
squish_testy = np.reshape(squish_testy,(-1,1))

[loss, mae] = model.evaluate(squish_testx, squish_testy, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(squish_testx).flatten()*2-1
plt.scatter(testx,test_predictions)

#print(np.split(x,[50])[0])
#print(np.split(squish_x,[50])[0])
#print(x)
#print(squish_x)
#print(y)
#print(squish_y)
#print((x[3]+np.pi*2)/(np.pi*4))
#print(squish_x[3])

#plt.xlabel('sample(n)')
#plt.ylabel('voltage(V)')
plt.show()