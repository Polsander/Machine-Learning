import numpy as np
import matplotlib.pyplot as plt
import pickle

from functions.helpers import calculate_a, gradient_last_layer, gradient_hidden_layer, update_weights_biases, random_2D_array
from functions.plots import show_raw_training

def loss_fn(A, Y):
    m = Y.shape[0]
    epsilon = 1e-15
    single_loss = np.sum(Y * np.log(A + epsilon), axis=1) # Add epsilon to avoid log(0)
    loss = -1/m * np.sum(single_loss)

    return loss

def accuracy_fn(A, Y):
    A_indeces = np.argmax(A, axis=1)
    Y_indeces = np.argmax(Y, axis=1)

    correct = np.sum((A_indeces == Y_indeces))

    accuracy = correct/Y.shape[0] * 100

    return accuracy


def predict(image_X, W1, b1, W2, b2, W3, b3, W4, b4):

    A1 = calculate_a(image_X, W1, b1, type="relu")
    A2 = calculate_a(A1, W2, b2, type="relu")
    A3 = calculate_a(A2, W3, b3, type="relu")
    A4 = calculate_a(A3, W4, b4, type="softmax")

    index = int(np.argmax(A4))

    if index == 0:
        print("Car!")
    elif index == 1:
        print("Bike!")
    elif index == 2:
        print("Tank!")

    print("A4 probabilities:", A4)

    #Reshape the image tested
    original_image = image_X.reshape(64, 64) * 255
    plt.imshow(original_image, cmap="gray")
    plt.show()



if __name__ == "__main__":

    X = np.load("/home/olivererdmann/Documents/code/ml_learn/vehicle/data/X_vehicles.npy")
    Y = np.load("/home/olivererdmann/Documents/code/ml_learn/vehicle/data/Y_vehicles.npy")

    # Need to generate my initial weights and biases
    # Doing a 3 layer neural network - 64 neurons / 64 neurons / 3 neurons - 8384 parameters
    min_value = -0.3
    max_value = 0.3

    np.random.seed(23)
    # seed 23 took me to 72% accuracy at 0.7 loss
    # 452 took to 77% accuracy at 0.5 loss (40k epochs)
    # 1111 took to 75% accuracy at 0.6 loss (70k epochs)

    W1 = random_2D_array(4096, 98, min_value, max_value) # 4096 input features, 98 neurons
    W2 = random_2D_array(98, 128, min_value, max_value) # 98 hidden features, 128 neurons
    W3 = random_2D_array(128, 64, min_value, max_value) # 128 hidden features, 64 neurons
    W4 = random_2D_array(64, 3, min_value, max_value) # 64 hidden features, 3 neurons

    b1 = np.zeros(98)
    b2 = np.zeros(128)
    b3 = np.zeros(64)
    b4 = np.zeros(3)


    # Some constants
    alpha = 9e-3
    epochs = 1600
    loss_metric = []
    steps_metric = []

    # Commence training
    for i in range(epochs):

        #Forward prop
        A1 = calculate_a(X, W1, b1, type="relu")
        A2 = calculate_a(A1, W2, b2, type="relu")
        A3 = calculate_a(A2, W3, b3, type="relu")
        A4 = calculate_a(A3, W4, b4, type="softmax")

        
        # Calculate loss
        loss = loss_fn(A4, Y)
        accuracy = accuracy_fn(A4, Y)
        if (i + 1) % 10 == 0:
            print("loss:", loss, "accuracy:", accuracy, "steps:", i + 1)
        loss_metric.append(loss)
        steps_metric.append(i+1)

        #Back prop
        dLdW4, dLdb4, dLdZ4 = gradient_last_layer(A4, Y, A3)
        dLdW3, dLdb3, dLdZ3 = gradient_hidden_layer(dLdZ4, W4, A3, A2, activation="relu")
        dLdW2, dLdb2, dLdZ2 = gradient_hidden_layer(dLdZ3, W3, A2, A1, activation="relu")
        dLdW1, dLdb1, dLdZ1 = gradient_hidden_layer(dLdZ2, W2, A1, X, activation="relu")

        #Update weights
        W4, b4 = update_weights_biases(W4, dLdW4, b4, dLdb4, alpha)
        W3, b3 = update_weights_biases(W3, dLdW3, b3, dLdb3, alpha)
        W2, b2 = update_weights_biases(W2, dLdW2, b2, dLdb2, alpha)
        W1, b1 = update_weights_biases(W1, dLdW1, b1, dLdb1, alpha)


    
    print("Final training loss:", loss)
    # print("Updated weights:\n", W1, "\n", W2, "\n")
    # print("Updated biases:\n", b1, "\n", b2)

    show_raw_training(steps_metric, loss_metric)


    # Instantiate and test
    with open('/home/olivererdmann/Documents/code/ml_learn/vehicle/data/test_dict.pkl', 'rb') as f:
        test_dict = pickle.load(f)
    
    print("choose tank1 - tank5, car1 - car5, and bike1 - bike5 to choose an image to test.")
    print("Enter 'q' to exit\n")

    while True:
        inpt = input("Which image would you like to test?: ")

        if inpt == 'q': break

        try:
            image_to_test = test_dict[inpt].reshape(1, -1)
            predict(image_to_test, W1, b1, W2, b2, W3, b3, W4, b4)

        except Exception as e:
            print(e)
            print("Something went wrong, try again or press 'q' to exit")
            continue
