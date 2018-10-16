import numpy as np

def sigmoid(x):
    return round(1.0/(1+ np.exp(-x)),5)

def sigmoid_derivative(x):
    return x * (1.0 - x)

class TrainingGrounds:
    def __init__(self, m, b):
        self.m          = m #entre -10 y 10
        self.b          = b #entre 0 y 1
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
        p = percept.predict(X)
        print("predict ", p, " answer ", d)
