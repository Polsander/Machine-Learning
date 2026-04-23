import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unsupervised_learning import scale
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.utils import shuffle
from neural_network_template import Neural_Network
import torch



class Kernel_SVR:
    def __init__(self):
        self.model = None

    def fit(self, X_scaled, Y_scaled):
        self.model = SVR(gamma=0.85, kernel="rbf")
        self.model.fit(X_scaled, Y_scaled.flatten())

    def predict(self, X_scaled):
        y_pred = self.model.predict(X_scaled)
        return y_pred

    def test_and_plot(self, X_scaled, Y_scaled):

        Y_pred = self.predict(X_scaled)
        # Parity plot
        plt.scatter(Y_scaled.flatten(), Y_pred.flatten(), label="SVR model")
        plt.xlabel("True Y")
        plt.ylabel("Predicted Y")
        plt.title("Parity Plot - Kernel SVR")
        plt.plot([Y_scaled.min(), Y_scaled.max()], [Y_scaled.min(), Y_scaled.max()], "k--", label="ideal")
        plt.legend()
        plt.show()


class Gaussian_Regression:
    def __init__(self):
        self.model = None

    def fit(self, X_scaled, Y_scaled):
        self.model = GaussianProcessRegressor(kernel=Matern(), random_state=73)
        self.model.fit(X_scaled, Y_scaled)

    def predict(self, X_scaled):
        Y_pred = self.model.predict(X_scaled)
        return Y_pred

    def test_and_plot(self, X_scaled, Y_scaled):
        Y_pred = self.predict(X_scaled)

        # Parity plot
        plt.scatter(Y_scaled.flatten(), Y_pred.flatten(), label="GP model")
        plt.xlabel("True Y")
        plt.ylabel("Predicted Y")
        plt.title("Parity Plot - GP")
        plt.plot([Y_scaled.min(), Y_scaled.max()], [Y_scaled.min(), Y_scaled.max()], "k--", label="ideal")
        plt.legend()
        plt.show()


class Regression_Dense_NN:
    def __init__(self, input_size, layer_sizes):
        self.model = Neural_Network(input_size, layer_sizes)
        self.epochs = []
        self.validation_losses = []
        self.training_losses = []
    
    def fit(self, X_scaled_train, Y_scaled_train, epochs = 100, lr=0.001, X_test = None, Y_test = None):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        X_tensor, Y_tensor = self.model.convert_to_tensors(X_scaled_train, Y_scaled_train)
        X_test_tensor, Y_test_tensor = None, None
        if X_test is not None and Y_test is not None:
            X_test_tensor, Y_test_tensor = self.model.convert_to_tensors(X_test, Y_test)

        for epoch in range(epochs):
            # === Forward propogation ===
            self.model.train() # Set model to training mode
            outputs = self.model(X_tensor)
            loss = criterion(outputs, Y_tensor)
            self.training_losses.append(loss.item())

            # === Back propogation ===
            optimizer.zero_grad() # clear old gradients
            loss.backward() # back propogate
            optimizer.step() # update weights

            # === Validation ===
            if X_test_tensor is not None and Y_test_tensor is not None:
                self.model.eval() # turn on evaluation mode (turns off dropout/batchnorm)
                with torch.no_grad():
                    validation_outputs = self.model(X_test_tensor)
                    validation_loss = criterion(validation_outputs, Y_test_tensor)
                    self.validation_losses.append(validation_loss.item())

            self.epochs.append(epoch+1)
        
    def plot_training_curve(self):

        plt.plot(self.epochs, self.training_losses, label="training loss")
        plt.plot(self.epochs, self.validation_losses, label="validation loss")
        plt.xlabel("epochs")
        plt.ylabel("MSE loss")
        plt.title("NN Training and Validation Loss")
        plt.legend()
        plt.show()
    
    def predict(self, X):

        self.model.eval()
        with torch.no_grad():
            X_tensor, _ = self.model.convert_to_tensors(X, np.array([]))
            Y_pred = self.model(X_tensor)
            return Y_pred.numpy()
    
    def parity_plot(self, X, Y):

        Y_pred = self.predict(X)
        # Parity plot
        plt.scatter(Y.flatten(), Y_pred.flatten(), label="NN model")
        plt.xlabel("True Y")
        plt.ylabel("Predicted Y")
        plt.title("Parity Plot - NN")
        plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], "k--", label="ideal")
        plt.legend()
        plt.show()



def run_supervised_learning(Xraw, Yraw, Xraw_test, Yraw_test):
    X, x_scaler = scale(Xraw)
    Y, y_scaler = scale(Yraw)
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.5, shuffle=True
    # )
    X_train = X
    Y_train = Y

    X_test = x_scaler.transform(Xraw_test)
    Y_test = y_scaler.transform(Yraw_test)

    # Try Kernel SVR
    kernel_svr_model = Kernel_SVR()
    kernel_svr_model.fit(X_train, Y_train)
    kernel_svr_model.test_and_plot(X_test, Y_test)
    Y_pred = kernel_svr_model.predict(X_test)
    print("SSE KERNEL SVR:      ", np.sum((Y_pred - Y_test.ravel())**2))
    print("MSE KERNEL SVR:      ", np.mean((Y_pred - Y_test.ravel())**2))

    # Try Gaussian Regression (slow and maybe impractical? -  given the amount of data we have used so far)
    X_train_sub, Y_train_sub = shuffle(X_train, Y_train, random_state=42)
    n_samples = int(len(X_train_sub) * 0.2)
    X_train_sub = X_train_sub[:n_samples]
    Y_train_sub = Y_train_sub[:n_samples]
    GP_model = Gaussian_Regression()
    GP_model.fit(X_train_sub, Y_train_sub)
    GP_model.test_and_plot(X_test, Y_test)
    Y_pred = GP_model.predict(X_test)
    print("\nSSE Gaussian:      ", np.sum((Y_pred.flatten() - Y_test.flatten())**2))
    print("MSE Gaussian:        ", np.mean((Y_pred.flatten() - Y_test.flatten())**2))

    # Try Neural Network
    NN = Regression_Dense_NN(len(X_train[0]), [30, 40, 15, 1])
    NN.fit(X_train, Y_train, epochs=150, lr=0.01, X_test=X_test, Y_test=Y_test)
    NN.plot_training_curve()
    NN.parity_plot(X_test, Y_test)
    Y_pred = NN.predict(X_test)
    print("\nSSE Neural Net:      ", np.sum((Y_pred.flatten() - Y_test.flatten())**2))
    print("MSE Neural Net:        ", np.mean((Y_pred.flatten() - Y_test.flatten())**2))

    # Run KFOLD Cross Validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    svr_sse, svr_mse = np.zeros(k), np.zeros(k)
    gp_sse, gp_mse = np.zeros(k), np.zeros(k)
    nn_sse, nn_mse = np.zeros(k), np.zeros(k)

    print(f"\nStarting Cross Validation for {k} runs...")

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        print("iteration ", i + 1)
        X_train, Y_train = X[train_idx, : ], Y[train_idx, : ]
        # X_test, Y_test = X[test_idx,: ], Y[test_idx,: ]

        # instantiate models
        svr_model = Kernel_SVR()
        gp_model = Gaussian_Regression()
        nn_model = Regression_Dense_NN(len(X_train[0]), [30,40,15,1])

        # Fit
        svr_model.fit(X_train, Y_train)

        n_samples = int(len(X_train) * 0.2)
        X_train_sub = X_train[:n_samples]
        Y_train_sub = Y_train[:n_samples]
        gp_model.fit(X_train_sub, Y_train_sub)

        nn_model.fit(X_train, Y_train, epochs=40, lr=0.01)

        # Predict
        Y_svr = y_scaler.inverse_transform(svr_model.predict(X_test).reshape(-1,1))
        Y_gp = y_scaler.inverse_transform(gp_model.predict(X_test).reshape(-1,1))
        Y_nn = y_scaler.inverse_transform(nn_model.predict(X_test))

        # Calculate SSE's and MSE's
        svr_sse[i] = np.sum((Y_svr.ravel() - Yraw_test.ravel())**2)
        svr_mse[i] = np.mean((Y_svr.ravel() - Yraw_test.ravel())**2)

        gp_sse[i] = np.sum((Y_gp.ravel() - Yraw_test.ravel())**2)
        gp_mse[i] = np.mean((Y_gp.ravel() - Yraw_test.ravel())**2)

        nn_sse[i] = np.sum((Y_nn.ravel() - Yraw_test.ravel())**2)
        nn_mse[i] = np.mean((Y_nn.ravel() - Yraw_test.ravel())**2)
    
    print("Support Vector Regression:       ",     "MSE:   ", f"{np.mean(svr_mse):.3f}      ", "Std MSE:    ", f"{np.std(svr_mse):.3f}" )
    print("Gaussian Regression:             ",     "MSE:   ", f"{np.mean(gp_mse):.3f}       ", "Std MSE:    ", f"{np.std(gp_mse):.3f}")
    print("Neural Network:                  ",     "MSE:   ", f"{np.mean(nn_mse):.3f}       ", "Std MSE:    ", f"{np.std(nn_mse):.3f}" )


    


if __name__ == "__main__":
    X_blue1 = np.load("data/blue/matrices/X_blue1.npy")
    X_blue2 = np.load("data/blue/matrices/X_blue2.npy")
    X_blue3 = np.load("data/blue/matrices/X_blue3.npy")
    X_blue4 = np.load("data/blue/matrices/X_blue4.npy")
    Y_blue1 = np.load("data/blue/matrices/Y_blue1.npy").reshape(-1, 1)
    Y_blue2 = np.load("data/blue/matrices/Y_blue2.npy").reshape(-1, 1)
    Y_blue3 = np.load("data/blue/matrices/Y_blue3.npy").reshape(-1, 1)
    Y_blue4 = np.load("data/blue/matrices/Y_blue4.npy").reshape(-1, 1)

    X_green1 = np.load("data/green/matrices/X_green1.npy")
    X_green2 = np.load("data/green/matrices/X_green2.npy")
    X_green3 = np.load("data/green/matrices/X_green3.npy")
    X_green4 = np.load("data/green/matrices/X_green4.npy")
    Y_green1 = np.load("data/green/matrices/Y_green1.npy").reshape(-1, 1)
    Y_green2 = np.load("data/green/matrices/Y_green2.npy").reshape(-1, 1)
    Y_green3 = np.load("data/green/matrices/Y_green3.npy").reshape(-1, 1)
    Y_green4 = np.load("data/green/matrices/Y_green4.npy").reshape(-1, 1)

    X_yellow1 = np.load("data/yellow/matrices/X_yellow1.npy")
    X_yellow2 = np.load("data/yellow/matrices/X_yellow2.npy")
    X_yellow3 = np.load("data/yellow/matrices/X_yellow3.npy")
    Y_yellow1 = np.load("data/yellow/matrices/Y_yellow1.npy").reshape(-1, 1)
    Y_yellow2 = np.load("data/yellow/matrices/Y_yellow2.npy").reshape(-1, 1)
    Y_yellow3 = np.load("data/yellow/matrices/Y_yellow3.npy").reshape(-1, 1)

    X_blue_test = np.load("data/blue/matrices/X_blue_test.npy")
    X_green_test = np.load("data/green/matrices/X_green_test.npy")
    X_yellow_test = np.load("data/yellow/matrices/X_yellow_test.npy")

    Y_blue_test = np.load("data/blue/matrices/Y_blue_test.npy").reshape(-1,1)
    Y_green_test = np.load("data/green/matrices/Y_green_test.npy").reshape(-1,1)
    Y_yellow_test = np.load("data/yellow/matrices/Y_yellow_test.npy").reshape(-1,1)

    X_blue = np.vstack([X_blue1, X_blue2, X_blue3, X_blue4])
    X_green = np.vstack([X_green1, X_green2, X_green3, X_green4])
    X_yellow = np.vstack([X_yellow1, X_yellow2, X_yellow3])

    Y_blue = np.vstack([Y_blue1, Y_blue2, Y_blue3, Y_blue4])
    Y_green = np.vstack([Y_green1, Y_green2, Y_green3, Y_green4])
    Y_yellow = np.vstack([Y_yellow1, Y_yellow2, Y_yellow3])

    X = np.vstack([X_green])
    Y = np.vstack(np.concatenate([Y_green]))

    df = pd.DataFrame(X, columns=["L", "A", "B", "L_var", "A_var", "B_var"])
    df.insert(3, "L^2", df["L"] ** 2)
    df.insert(4, "A^2", df["A"] ** 2)
    df.insert(5, "B^2", df["B"] ** 2)
    df.insert(6, "LA", df["L"] * df["A"])
    df.insert(7, "LB", df["L"] * df["B"])
    df.insert(8, "AB", df["A"] * df["B"])

    X = df.to_numpy()

    X_test = np.concatenate([X_green_test])
    Y_test = np.vstack(np.concatenate([Y_green_test]))

    df_test = pd.DataFrame(X_test, columns=["L", "A", "B", "L_var", "A_var", "B_var"])
    df_test.insert(3, "L^2", df_test["L"] ** 2)
    df_test.insert(4, "A^2", df_test["A"] ** 2)
    df_test.insert(5, "B^2", df_test["B"] ** 2)
    df_test.insert(6, "LA", df_test["L"] * df_test["A"])
    df_test.insert(7, "LB", df_test["L"] * df_test["B"])
    df_test.insert(8, "AB", df_test["A"] * df_test["B"])

    X_test = df_test.to_numpy()


    # Run functions
    run_supervised_learning(X, Y, X_test, Y_test)
