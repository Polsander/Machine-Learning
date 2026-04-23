# Unsupervised learning script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import KernelPCA



def scale(matrix):
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)

    return scaled_matrix, scaler

def PCA(Xraw, Yraw):
    X, x_scaler = scale(Xraw)
    Y, y_scaler = scale(Yraw)

    # Conduct SVD
    # No redundant features!
    def SVD_rank(X_sub, tol):
        return np.sum(np.linalg.svd(X_sub, full_matrices=False)[1] > tol)
    
    for i in range(1, len(X[0])+1):
        print(f'rank(X[:,1:{i}]) = ', SVD_rank(X[:,:i], tol=1e-1))
    # PCA 3D
    U,S,Vt = np.linalg.svd(X, full_matrices=False) # Rank X is the number of independent columns
    W = Vt.T[:,:3]

    T = X @ W
    gamma = np.linalg.solve(T.T @ T, T.T @ Y)
    print(np.shape(T))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(T[:,0], T[:,1], T[:,2], c=Y, cmap="viridis", s=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D Reduction")
    plt.colorbar(scatter)

    plt.show()

    #Predict with model and parity plot
    Y_pred = T @ gamma
    plt.scatter(Y_pred, Y, label="PCA model")
    plt.plot([-2,1.5], [-2,1.5], linestyle="dotted", label="ideal")
    plt.title("Parity Plot - PCA")
    plt.xlabel("Y actual")
    plt.ylabel("Y predicted")
    plt.legend()
    plt.show()

    # Parameter certainty

    return W, gamma

def PCA_kernel(Xraw, Yraw, plot = False):
    X, x_scaler = scale(Xraw)
    Y, y_scaler = scale(Yraw)

    kPCA_transformer = KernelPCA(n_components=7, kernel='rbf')

    X = np.column_stack((np.ones(len(X)), X))
    breakpoint()
    T = kPCA_transformer.fit_transform(X)
    print(T.shape)
    gamma = np.linalg.solve(T.T @ T, T.T @ Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if plot:
        scatter = ax.scatter(T[:,0], T[:,1], T[:,2], c=Y, cmap="viridis", s=1)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("Kernel PCA 3D Reduction")
        plt.colorbar(scatter)

        plt.show()

        #Predict with model and parity plot
        Y_pred = T @ gamma
        plt.scatter(Y_pred, Y, label="Kernel PCA model")
        plt.plot([-2,1.5], [-2,1.5], linestyle="dotted", label="ideal")
        plt.title("Parity Plot - PCA")
        plt.xlabel("Y actual")
        plt.ylabel("Y predicted")
        plt.legend()
        plt.show()


def PLS(Xraw, Yraw):
    X, x_scaler = scale(Xraw)
    Y, y_scaler = scale(Yraw)

    # PLS 3D
    X_tilde = X.copy()
    k = 7
    W_pls = np.zeros((np.shape(X_tilde)[1], k))
    for i in range(k):
        U, S, Vt = np.linalg.svd(X_tilde.T @ Y, full_matrices=False)
        ui = U[:, 0]
        W_pls[:, i] = ui
        Xu = np.vstack(X_tilde @ ui)
        I = np.eye(len(X_tilde))
        X_tilde = (I - ((Xu @ Xu.T)/(Xu.T @ Xu))) @ X_tilde
    
    T = X @ W_pls
    gamma = np.linalg.solve(T.T @ T, T.T @ Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(T[:,0], T[:,1], T[:,2], c=Y, cmap="viridis", s=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PLS 3D Reduction")
    plt.colorbar(scatter)

    plt.show()

    Y_pred = T @ gamma
    plt.scatter(Y_pred, Y, label="PLS model")
    plt.plot([-2,1.5], [-2,1.5], linestyle="dotted", label="ideal")
    plt.title("Parity Plot - PLS")
    plt.xlabel("Y actual")
    plt.ylabel("Y predicted")
    plt.legend()
    plt.show()

    # Uncertainty


    return W_pls, gamma

class PCA_op():
    def __init__(self):
        self.W = None
        self.gamma = None

    def fit(self, X, Y, k=3):

        # PCA 3D
        U,S,Vt = np.linalg.svd(X, full_matrices=False) # Rank X is the number of independent columns
        W = Vt.T[:,:k]
        T = X @ W
        T_ = np.hstack((np.ones((len(T), 1)), T))
        gamma = np.linalg.solve(T_.T @ T_, T_.T @ Y)
        self.W = W
        self.gamma = gamma

    def predict(self, X):
        T = X @ self.W
        T_ = np.hstack((np.ones((len(T), 1)), T))
        Y_pred = T_ @ self.gamma
        return Y_pred

class PLS_op():
    def __init__(self):
        self.W = None
        self.gamma = None
        self.x_scaler = None
        self.y_scaler = None

    def fit(self, X, Y, k=3):

        # PLS 3D
        X_tilde = X.copy()
        W_pls = np.zeros((np.shape(X_tilde)[1], k))
        for i in range(k):
            U, S, Vt = np.linalg.svd(X_tilde.T @ Y, full_matrices=False)
            ui = U[:, 0]
            W_pls[:, i] = ui
            Xu = np.vstack(X_tilde @ ui)
            I = np.eye(len(X_tilde))
            X_tilde = (I - ((Xu @ Xu.T)/(Xu.T @ Xu))) @ X_tilde
        
        T = X @ W_pls
        # Add intercept
        T_ = np.hstack((np.ones((len(T), 1)), T))
        gamma = np.linalg.solve(T_.T @ T_, T_.T @ Y)
        self.W = W_pls
        self.gamma = gamma

    def predict(self, X):
        T = X @ self.W
        T_ = np.hstack((np.ones((len(T), 1)), T))
        Y_pred = T_ @ self.gamma
        return Y_pred

class LinRegression():
    def __init__(self):
        self.theta = None

    def fit(self, X, Y):
        X_ = np.hstack((np.ones((len(X), 1)), X))
        theta = np.linalg.solve(X_.T @ X_, X_.T @ Y)
        self.theta = theta
    
    def predict(self, X):
        X_ = np.hstack((np.ones((len(X), 1)), X))
        Y_pred = X_ @ self.theta
        return Y_pred
    



def cross_validate(Xraw, Yraw, k):
    kf = KFold(n_splits=k, shuffle=True)

    mse_pca = np.zeros(k)
    mse_pls = np.zeros(k)
    mse_lin = np.zeros(k)

    X, x_scaler = scale(Xraw)
    Y, y_scaler = scale(Yraw)

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Instantiate all models
        pca_model = PCA_op()
        pls_model = PLS_op()
        lin_model = LinRegression()

        # Fit models
        pca_model.fit(X_train, Y_train, k=7)
        pls_model.fit(X_train, Y_train, k=7)
        lin_model.fit(X_train, Y_train)

        # Test models
        Y_pca = pca_model.predict(X_test)
        Y_pls = pls_model.predict(X_test)
        Y_lin = lin_model.predict(X_test)

        #MSE calculations
        mse_pca[i] = np.mean((Y_test - Y_pca)**2)
        mse_pls[i] = np.mean((Y_test - Y_pls)**2)
        mse_lin[i] = np.mean((Y_test - Y_lin)**2)
        df1 = pd.DataFrame({"Y_pca": Y_pca.flatten(), "Y_test": Y_test.flatten()})
        df2 = pd.DataFrame({"Y_pls": Y_pls.flatten(), "Y_test": Y_test.flatten()})
        df3 = pd.DataFrame({"Y_lin": Y_lin.flatten(), "Y_test": Y_test.flatten()})

    print("*** Average MSE for each model ****")
    print("MSE PCA model:       ", np.mean(mse_pca))
    print("MSE PLS model:       ", np.mean(mse_pls))
    print("MSE LIN model:       ", np.mean(mse_lin))

if __name__ == "__main__":
    X_blue1 = np.load("data/blue/matrices/X_blue1.npy")
    X_blue2 = np.load("data/blue/matrices/X_blue2.npy")
    X_blue3 = np.load("data/blue/matrices/X_blue3.npy")
    Y_blue1 = np.load("data/blue/matrices/Y_blue1.npy").reshape(-1,1)
    Y_blue2 = np.load("data/blue/matrices/Y_blue2.npy").reshape(-1,1)
    Y_blue3 = np.load("data/blue/matrices/Y_blue3.npy").reshape(-1,1)

    X_green1 = np.load("data/green/matrices/X_green1.npy")
    X_green2 = np.load("data/green/matrices/X_green2.npy")
    X_green3 = np.load("data/green/matrices/X_green3.npy")
    Y_green1 = np.load("data/green/matrices/Y_green1.npy").reshape(-1,1)
    Y_green2 = np.load("data/green/matrices/Y_green2.npy").reshape(-1,1)
    Y_green3 = np.load("data/green/matrices/Y_green3.npy").reshape(-1,1)

    X_yellow1 = np.load("data/yellow/matrices/X_yellow1.npy")
    X_yellow2 = np.load("data/yellow/matrices/X_yellow2.npy")
    Y_yellow1 = np.load("data/yellow/matrices/Y_yellow1.npy").reshape(-1,1)
    Y_yellow2 = np.load("data/yellow/matrices/Y_yellow2.npy").reshape(-1,1)

    X_blue = np.vstack([X_blue1, X_blue2, X_blue3])
    X_green = np.vstack([X_green1, X_green2, X_green3])
    X_yellow = np.vstack([X_yellow1, X_yellow2])

    Y_blue = np.vstack([Y_blue1, Y_blue2, Y_blue3])
    Y_green = np.vstack([Y_green1, Y_green2, Y_green3])
    Y_yellow = np.vstack([Y_yellow1, Y_yellow2])

    X = np.vstack([X_blue, X_green, X_yellow])
    Y = np.vstack(np.concatenate([Y_blue, Y_green, Y_yellow]))

    df = pd.DataFrame(X, columns=["L", "A", "B", "L_var", "A_var", "B_var"])
    df.insert(3, "L^2", df["L"]**2)
    df.insert(4, "A^2", df["A"]**2)
    df.insert(5, "B^2", df["B"]**2)
    df.insert(6, "LA", df["L"] * df["A"])
    df.insert(7, "LB", df["L"] * df["B"])
    df.insert(8, "AB", df["A"] * df["B"])

    X = df.to_numpy()
    # Question is if we should possibly merge these together in one matrix or keep them all separate?
    # Also may need more than 3 dimensions
    # cross_validate(X, Y, 10)

    PCA(X, Y)
    PLS(X, Y)
    PCA_kernel(X, Y, plot=True)
