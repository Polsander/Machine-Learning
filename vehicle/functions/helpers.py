import numpy as np

def sigmoid(z):
    '''Returns the sigmoid function for parameter z'''
    return 1/(1 + np.exp(-z))



def relu(z):
    ''' Returns the relu function for paramter z'''
    return np.maximum(0, z)


def softmax(z):
    ''' Returns the softmax function for parameter z'''
    z_stabilize = z - np.max(z, axis=1, keepdims=True) # Stabilize/normalize the results by subtracting all with max Z to avoid exponent explosion
    exp_z = np.exp(z_stabilize) # Get the exponents for every example (m, n) shape
    row_sums = np.sum(exp_z, axis=1, keepdims=True) # summate each row (m, n) shape
    return exp_z/row_sums



def normalize(value, max, min):
    '''Returns a normalized value'''
    normalized = (value - min)/(max - min)

    return normalized



def calculate_a(X, W, b, type):
    '''
    X is an (m, n) shaped array
    W is an (n, v) shaped array - where v is the number of neurons
    b is a 1-D array (v) - Where v is the number of neurons
    '''
    if type != "sigmoid" and type != "relu" and type != "softmax":
        raise ValueError("Must provide valid type as sigmoid or relu")

    Z = np.matmul(X, W) + b
    A = None

    if type == "sigmoid":
        A = sigmoid(Z)
    elif type == "relu":
        A= relu(Z)
    elif type == "softmax":
        A = softmax(Z)

    return A



def gradient_last_layer(A, Y, X):
    m = A.shape[0]
    dLdZ = A - Y # change in Z
    dLdW = (1/m) * np.matmul(X.T, dLdZ)
    dLdb = (1/m) * np.sum(dLdZ, axis=0)  # Keeps shape (1,)

    return (dLdW, dLdb, dLdZ)




def gradient_hidden_layer(dLdZ_next, W_next, A_curr, A_prev, activation): # dLdZ, W2, A1, X
    '''
    Params:
    dLdZ_next Gradient from the next layer (m, v)
    W_next Weights of the next layer (n, v)
    A_curr Activation of current layer (m, n)
    A_prev Inputs to current layer (m, p)

    Returns:
    dLdW (p, n)
    dLdb (n,)
    '''
    

    # A_curr = sigmoid(Z_curr)
    if activation == "sigmoid":
        dAdZ_curr = A_curr * (1 - A_curr)
    elif activation == "relu":
        dAdZ_curr = (A_curr > 0).astype(float)
    else:
        raise ValueError("Unsupported activation function")
    
    dLdA_curr = np.matmul(dLdZ_next, W_next.T)
    dLdZ_curr = dLdA_curr * dAdZ_curr

    m = A_prev.shape[0]
    dLdW = np.matmul(A_prev.T, dLdZ_curr) / m
    dLdb = np.sum(dLdZ_curr, axis=0) / m # Axis is important here because we want the bias per neuron

    return (dLdW, dLdb, dLdZ_curr)




def update_weights_biases(W, dLdW, b, dLdb, alpha):

    W = W - alpha* dLdW
    b = b - alpha * dLdb

    return W, b


def random_2D_array(rows, cols, min, max) -> np.array:
    '''
    Generates a uniform random matrix or 2D numpy array
    '''

    return np.random.uniform(low=min, high=max, size=(rows, cols))