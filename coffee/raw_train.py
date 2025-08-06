'''
This script here is an example for me to learn forward propagation and backward propagation manually just from using numpy.
This is supposedly going to help me train and understand machine learning neural networks.
Later, I will use tensorflow in a seperate file.
'''

import numpy as np
import matplotlib.pyplot as plt
from helpers import sigmoid, relu, normalize

'''
Generally speaking we are sending data X (time and temperature) into a neuron.
We can define any number of neurons, but just so we do not overfit, we are choosing to do 3 neurons in layer 1, and 1 neuron in layer 2.

The first layer (Hidden layer with 3 Neurons):
Each of th e3 neurons does:
z = W1 * X1 + W2 * X2 + b (for every row)
a = sigmoid(z)

This must be all vectorized - so therefore for an input of X of shape (200, 2)
A_1 will give a shape of (200, 3) -> 200 examples, with 3 activations each

Note: Weights and biases are different and random for each neuron when they begin training
The function below can be ran for hwoever many time necessary (n layers) before we need to do back propagation and check results
'''

def calculate_a(X, W, b, type):
    '''
    X is an (m, n) shaped array
    W is an (n, v) shaped array - where v is the number of neurons
    b is a 1-D array (v) - Where v is the number of neurons
    '''
    if type != "sigmoid" and type != "relu":
        raise ValueError("Must provide valid type as sigmoid or relu")

    Z = np.dot(X, W) + b
    A = None

    if type == "sigmoid":
        A = sigmoid(Z)
    else:
        A= relu(Z)

    return A


def loss_fn(A, Y):
    # Loss = -1/m sum(y*log(A) + (1-y)*log(1-a))

    m = A.shape[0]

    loss = -1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    # print("Loss:", loss)

    return loss


'''
After the loss function is calculated, we need to compute the gradients and know which direction to go.

dL/dw and dL/db are the partial derivatives of the loss function with respect to W weights and b Biases.
This is only applicable for the last final layer. The hidden layers will require back propagation.

dL/dw = (a-y) * x (for a single example)
dL/db = (a-y) (for a single example )
'''

def gradient_last_layer(A, Y, X):
    m = A.shape[0]
    dLdZ = A - Y # change in Z
    dLdW = (1/m) * np.dot(X.T, dLdZ)
    dLdb = (1/m) * np.sum(dLdZ, axis=0)  # Keeps shape (1,)

    return (dLdW, dLdb, dLdZ)

'''
This is probably the most complex function, and where it becomes compatible with back propagation.
If we have 2 layers, (which means we have matrices W1 and W2), then we are computing dLdW2 and dLdb2.

So we take the derivative of the loss function as we did before, but this time with respect to W1 (or the hidden layer weight)
This leaves us having to do a massive chain rule - which results in the following:

dL/dW1 = dL/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1

Or in a moregeneral notation:

dL/dW^(l) = a_k^(l-1) * sigma'(z_j^(l)) * dL/da_j^(l)

So now we begin calculating each term in the gradient hidden layer function below to obtain dL/dW1

'''

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
        dA_dZ_curr = A_curr * (1 - A_curr)
    elif activation == "relu":
        dA_dZ_curr = (A_curr > 0).astype(float)
    else:
        raise ValueError("Unsupported activation function")
    
    dL_dA_curr = np.dot(dLdZ_next, W_next.T)
    dL_dZ_curr = dL_dA_curr * dA_dZ_curr

    m = A_prev.shape[0]
    dLdW = np.dot(A_prev.T, dL_dZ_curr) / m
    dLdb = np.sum(dL_dZ_curr, axis=0) / m # Axis is important here because we want the bias per neuron

    return (dLdW, dLdb, dL_dZ_curr)


'''
Finally we write a function that is able to update the given weights. WHere alpha is the learning rate.
'''


def update_weights_biases(W, dLdW, b, dLdb, alpha):

    W = W - alpha* dLdW
    b = b - alpha * dLdb

    return W, b


'''
For inference :)
'''

def predict(X, W1, b1, W2, b2, W3, b3):
    A1 = calculate_a(X, W1, b1, type="relu")
    A2 = calculate_a(A1, W2, b2, type="relu")
    A3 = calculate_a(A2, W3, b3, type="sigmoid")

    print("A3 probability", A3)
    if (A3 >= 0.5):
        return "Good coffee"
    else:
        return "Bad coffee"

def showPlot(steps, loss):

    plt.xlabel("Steps")
    plt.ylabel("loss")

    plt.plot(steps, loss)
    plt.show()






if __name__ == "__main__":
    
    # Define Inputs and initial variables

    X = np.load("/home/olivererdmann/Documents/code/ml_learn/coffee/data/data_X.npy")
    Y = np.load("/home/olivererdmann/Documents/code/ml_learn/coffee/data/data_Y.npy")
    Y = Y.reshape(-1, 1)

    epochs = 12200
    learning_rate = 5e-2
    loss_arr = []
    steps_arr = []


    W1 = np.array([[0.07, 0.2, 0.03, -0.1, 0.9], [0.4, -0.1, 0.04, -0.9, 0.1]])
    b1 = np.array([0, 0, 0, 0, 0])

    W2 = np.array([[0.17, -0.2, -0.03, 0.7, 0.7], [0.08, -0.1, -0.04, 0.4, 0.8], [-0.02, 0.5, 0.5, -0.8, -0.9], [0.4, 0.2, 0.7, -0.7, -0.12], [-0.12, 0.34, -0.5, -0.8, 0.11]])
    b2 = np.array([0, 0, 0, 0, 0])

    W3 = np.array([[0.06], [0.4], [0.3], [-0.4], [0.4]])
    b3 = np.array([0.0])

    # Now we run our loop based on how many epochs we choose :)
    for i in range(epochs):

        #Forward propagationS
        A1 = calculate_a(X, W1, b1, type="relu")
        A2 = calculate_a(A1, W2, b2, type="relu")
        A3 = calculate_a(A2, W3, b3, type="sigmoid")

        loss = loss_fn(A3, Y)
        # print("loss:", loss, "epoch:", i)
        loss_arr.append(loss)
        steps_arr.append(i)

        #Backward propagation
        dLdW3, dLdb3, dLdZ3 = gradient_last_layer(A3, Y, A2) #Output activation, Solution, Input to last layer

        dLdW2, dLdb2, dLdZ2 = gradient_hidden_layer(dLdZ3, W3, A2, A1, activation="relu") # Change of L wrt to Z in next layer, Weights in next layer, Output activation, Input
        dLdW1, dLdb1, dLdZ1 = gradient_hidden_layer(dLdZ2, W2, A1, X, activation="relu")


        #Update weights and biases
        W1, b1 = update_weights_biases(W1, dLdW1, b1, dLdb1, learning_rate)
        W2, b2 = update_weights_biases(W2, dLdW2, b2, dLdb2, learning_rate)
        W3, b3 = update_weights_biases(W3, dLdW3, b3, dLdb3, learning_rate)

        # Once the weights and biases are updated, we loop back and go again
    
    print("Final training loss:", loss)
    print("Updated weights:\n", W1, "\n", W2, "\n")
    print("Updated biases:\n", b1, "\n", b2)

    # showPlot(steps_arr, loss_arr)
    print("\nInference: \n", "Enter 'q' to exit")
    max_temp = 300
    min_temp = 160
    max_time = 21
    min_time = 7

    while True:

        temp = input("Enter a temperature: ")
        if temp == 'q': break
        time = input("Enter a time: ")
        if time == 'q': break

        try:
            temp = float(temp)
            time = float(time)
            # Normalize
            temp_norm = normalize(temp, max_temp, min_temp)
            time_norm = normalize(time, max_time, min_time)
        except ValueError:
            print("Temperature or Time is not a valid numeric type")
            continue

        

        print("time = ", time, " mins")
        print("temp = ", temp, " degrees")
        X_new = np.array([[time_norm, temp_norm]]) # MUST NORMALIZE!!!

        res = predict(X_new, W1, b1, W2, b2, W3, b3)
        print(res)
