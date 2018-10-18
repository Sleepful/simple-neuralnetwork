import numpy as np
from ann import *
from matplotlib import pyplot as plt

#################
### Variables ###
#################
coordLim = 10 #Define the limit of the graphic
plt.isinteractive()
points = [(1,5), (4,-3), (7,-5)]

#################
###  Program  ###
#################

####################################
# receive the beginning and the    #
# destiny of the line and print it #
####################################
def printLine(cord1, cord2):
        print(cord1[0], "...", cord2[0])
        pointx = np.linspace(cord1[0], cord2[0])
        pointy = np.linspace(cord1[1], cord2[1])
        plt.axhline(0, color="red")
        plt.axvline(0, color="red")
        plt.xlim(-coordLim, coordLim)
        plt.ylim(-coordLim, coordLim)
        plt.plot([1,5], [5,7], 'r--', [1,4,7], [5,-3,-5], 'gD')
        plt.show()
        #plt.pause(0.5)
        #plt.close()

printLine((-1, 3), (5, 7))
    
'''
for i in range(10):
    for j in range(10):
        printLine((i, j), (i+10, j+10))
'''
