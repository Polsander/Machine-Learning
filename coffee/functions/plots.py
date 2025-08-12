import matplotlib.pyplot as plt

def tensor_training_curve(train_loss, validation_loss, epochs):

    plt.xlabel("training steps")
    plt.ylabel("loss")

    plt.plot(epochs, train_loss)
    plt.plot(epochs, validation_loss)

    plt.show()