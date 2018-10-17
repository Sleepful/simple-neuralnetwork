import numpy as np
from matplotlib import pyplot as plt

#################
### Variables ###
#################
coordLim = 100  # Define the limit of the graphic
plt.isinteractive()
pointsX = []
pointsY = []
linePointsX = []
linePointsY = []

####################################
# receive the beginning and the    #
# destiny of the line and print it #
####################################
def printLine(cord1, cord2, lineCord1, lineCord2):
    #pointx = np.linspace(cord1[0], cord2[0])
    #pointy = np.linspace(cord1[1], cord2[1])
    plt.axhline(0, color="red")    #draw a red line that divide X axis
    plt.axvline(0, color="red")    #draw a red line that divide Y axis

    plt.xlim(-coordLim,coordLim)
    plt.ylim(-coordLim,coordLim)
    plt.plot(cord1, cord2, "go", lineCord1, lineCord2, "r-")
    plt.show()      #use .draw() when time pause is enabled

    #plt.pause(0.5) #tiempo de espera para que muestre la imagen
    #plt.close()    #cierra la gráfica

class TrainingGrounds:
    def __init__(self, m, b):
        self.m          = m #between 0 y 100
        self.b          = b #between 0 y 1
        self.generate_new_dot()
        self.classify_dot()

    def generate_new_dot(self):
        self.dot = np.random.rand(2)*100
        self.classify_dot()

    def classify_dot(self):
        if(0>self.m*self.dot[0]+self.b-self.dot[1]):
            self.C=1
        else:
            self.C=0
        return self.C

class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=0.5, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0

    def predict(self, X):
        x = np.insert(X, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for _ in range(self.epochs):
            y = self.predict(X)
            x = np.insert(X, 0, 1)
            e = d - y
            self.W = self.W + self.lr * e * x



if __name__ == "__main__":
    m = 1
    b = 2
    trainer = TrainingGrounds(m,b)
    x = trainer.dot
    y = trainer.C

    percept = Perceptron(2)

    for i in range(1000):
        trainer.generate_new_dot()

        X = trainer.dot
        d = trainer.C

        percept.fit(X, d)


    for i in range(20):
        trainer.generate_new_dot()

        X = trainer.dot
        d = trainer.C

        pointsX.append(int(X[0]))
        pointsY.append(int(X[1]))

        y = m*pointsX[i]+b      #finds the points in Y axis of the dividing line

        linePointsX.append(m*int(X[0]))
        linePointsY.append(y)

        p = percept.predict(X)

        print("Point ", pointsX[i], ",", pointsY[i], " -- predict ", p, " answer ", d)

    printLine(pointsX, pointsY, linePointsX, linePointsY)   #print the graphic with all elements