import numpy as np

def sigmoid(z):
    '''Returns the sigmoid function for parameter z'''
    return 1/(1 + np.exp(-z))

def relu(z):
    ''' Returns the relu function for paramter z'''
    return np.maximum(0, z)