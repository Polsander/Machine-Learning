import numpy as np
import matplotlib.pyplot as plt
import random

def generate_coffee_dataset(num_samples, normalize=False):

    data_X = np.zeros((num_samples, 2))
    data_Y = np.zeros((num_samples))

    for i in range(num_samples):

        temp = random.randint(160, 300)
        time = random.randint(7, 21)

        if (9 <= time <= 12) and (220 <= temp <= 282):
            coffee = 1
        elif (12 < time <= 14) and (210 <= temp <= 260):
            coffee = 1
        elif (14 < time <= 18) and (190 <= temp <= 250):
            coffee = 1
        else:
            coffee = 0
        
        if normalize:
            time = (time - 7)/(21 - 7) # (time - lowest time)/ (highest time - lowest time)
            temp = (temp - 160)/(300 - 160)
        
        data_X[i][0] = time
        data_X[i][1] = temp

        data_Y[i] = coffee
    
    return (data_X, data_Y)


def plot_data(X, Y):
    good_coffee = Y == 1
    bad_coffee = Y == 0

    plt.scatter(X[good_coffee, 0], X[good_coffee, 1], marker="o", color="green", label="Good", s=100, edgecolor="black")
    plt.scatter(X[bad_coffee, 0], X[bad_coffee, 1], marker="x", color="red", label="Bad", s=100)

    plt.xlabel("Roasting time (min)")
    plt.ylabel("Temperature Celsius")
    plt.title("Coffee Roasting Outcomes")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, Y = generate_coffee_dataset(500, True)
    plot_data(X, Y)

    #Export
    np.save("data_X.npy", X)
    np.save("data_Y.npy", Y)
