import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def scale(matrix):
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)

    return scaled_matrix, scaler

def PLS_visualize(Xraw, Yraw):
    X, x_scaler = scale(Xraw)
    # Y, y_scaler = scale(Yraw)

    # PLS 3D
    X_tilde = X.copy()
    k = 7
    W_pls = np.zeros((np.shape(X_tilde)[1], k))
    for i in range(k):
        U, S, Vt = np.linalg.svd(X_tilde.T @ Yraw, full_matrices=False)
        ui = U[:, 0]
        W_pls[:, i] = ui
        Xu = np.vstack(X_tilde @ ui)
        I = np.eye(len(X_tilde))
        X_tilde = (I - ((Xu @ Xu.T)/(Xu.T @ Xu))) @ X_tilde
    
    T = X @ W_pls
    gamma = np.linalg.solve(T.T @ T, T.T @ Yraw)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create custom colormap for classes: Blue=0, Green=1, Yellow=2
    colors = ['blue', 'green', 'yellow']
    cmap = ListedColormap(colors)
    
    # Create norm to map class values to colormap
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    scatter = ax.scatter(T[:,0], T[:,1], T[:,2], c=Yraw, cmap=cmap, norm=norm, s=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PLS 3D Reduction")
    plt.colorbar(scatter)

    plt.show()

    return W_pls, gamma

def PCA_visualize(Xraw, Yraw):

    # Split train test
    X, x_scaler = scale(Xraw)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Yraw, test_size=0.33)

    pca_model = PCA(n_components=3, random_state=42)
    T = pca_model.fit_transform(X)
    W = pca_model.components_
    gamma = np.linalg.solve(T.T @ T, T.T @ Yraw)

    colors = ['blue', 'green', 'yellow']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    # Create figure with 2D and 3D subplots
    fig = plt.figure(figsize=(14, 5))
    
    # 2D plot (PC1 vs PC2)
    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(T[:,0], T[:,1], c=Yraw, cmap=cmap, norm=norm, s=20)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("PCA 2D Reduction (PC1 vs PC2)")
    plt.colorbar(scatter1, ax=ax1, label="Class")
    
    # 3D plot (PC1 vs PC2 vs PC3)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(T[:,0], T[:,1], T[:,2], c=Yraw, cmap=cmap, norm=norm, s=1)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("PCA 3D Reduction")
    plt.colorbar(scatter2, ax=ax2, label="Class")

    plt.tight_layout()
    plt.show()


def PCA_neutral(Xraw, Yraw):

    # Split train test
    X, x_scaler = scale(Xraw)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Yraw, test_size=0.33)

    pca_model = PCA(n_components=3, random_state=42)
    T = pca_model.fit_transform(X)
    W = pca_model.components_
    gamma = np.linalg.solve(T.T @ T, T.T @ Yraw)

    colors = ['grey', 'blue', 'green', 'yellow']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    # Create figure with 2D and 3D subplots
    fig = plt.figure(figsize=(14, 5))
    
    # 2D plot (PC1 vs PC2)
    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(T[:,0], T[:,1], c=Yraw, cmap=cmap, norm=norm, s=20)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("PCA 2D Reduction (neutral)")
    plt.colorbar(scatter1, ax=ax1, label="Class")
    
    # 3D plot (PC1 vs PC2 vs PC3)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(T[:,0], T[:,1], T[:,2], c=Yraw, cmap=cmap, norm=norm, s=1)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("PCA 3D Reduction (neutral)")
    plt.colorbar(scatter2, ax=ax2, label="Class")

    plt.tight_layout()
    plt.show()


def SVD_decomp(X):
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    print(S)





if __name__ == "__main__":
    X_blue1 = np.load("data/blue/matrices/X_blue1.npy")
    X_blue2 = np.load("data/blue/matrices/X_blue2.npy")
    X_blue3 = np.load("data/blue/matrices/X_blue3.npy")
    X_blue_test = np.load("data/blue/matrices/X_blue_test.npy")

    X_green1 = np.load("data/green/matrices/X_green1.npy")
    X_green2 = np.load("data/green/matrices/X_green2.npy")
    X_green3 = np.load("data/green/matrices/X_green3.npy")
    X_green_test = np.load("data/green/matrices/X_green_test.npy")


    X_yellow1 = np.load("data/yellow/matrices/X_yellow1.npy")
    X_yellow2 = np.load("data/yellow/matrices/X_yellow2.npy")
    X_yellow_test = np.load("data/yellow/matrices/X_yellow_test.npy")

    X_blue = np.vstack([X_blue1, X_blue2, X_blue3])
    X_green = np.vstack([X_green1, X_green2, X_green3])
    X_yellow = np.vstack([X_yellow1, X_yellow2])

    Y_blue = np.vstack(np.ones(len(X_blue)) * 0)
    Y_green = np.vstack(np.ones(len(X_green)) * 1)
    Y_yellow = np.vstack(np.ones(len(X_yellow)) * 2)

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
    
    SVD_decomp(X)
    # PLS_visualize(X, Y)
    PCA_visualize(X, Y)

    # Try introducing a neutral-zone    
    Y_neutral_blue1 = np.vstack(np.ones(300)* 0)
    Y_blue1 = np.vstack(np.ones(len(X_blue1[300:])) * 1)
    Y_neutral_blue2 = np.vstack(np.ones(300)* 0)
    Y_blue2 = np.vstack(np.ones(len(X_blue2[300:])) * 1)
    Y_neutral_blue3 = np.vstack(np.ones(550)* 0)
    Y_blue3 = np.vstack(np.ones(len(X_blue3[550:])) * 1)

    Y_blue = np.concatenate([Y_neutral_blue1, Y_blue1, Y_neutral_blue2, Y_blue2, Y_neutral_blue3, Y_blue3])

    Y_neutral_green1 = np.vstack(np.ones(90)* 0)
    Y_green1 = np.vstack(np.ones(len(X_green1[90:])) * 2)
    Y_neutral_green2 = np.vstack(np.ones(200)* 0)
    Y_green2 = np.vstack(np.ones(len(X_green2[200:])) * 2)
    Y_neutral_green3 = np.vstack(np.ones(200)* 0)
    Y_green3 = np.vstack(np.ones(len(X_green3[200:])) * 2)

    Y_green = np.concatenate([Y_neutral_green1, Y_green1, Y_neutral_green2, Y_green2, Y_neutral_green3, Y_green3])

    Y_neutral_yellow1 = np.vstack(np.ones(150)* 0)
    Y_yellow1 = np.vstack(np.ones(len(X_yellow1[150:])) * 3)
    Y_neutral_yellow2 = np.vstack(np.ones(150)* 0)
    Y_yellow2 = np.vstack(np.ones(len(X_yellow2[150:])) * 3)

    Y_yellow = np.concatenate([Y_neutral_yellow1, Y_yellow1, Y_neutral_yellow2, Y_yellow2])


    Y = np.vstack(np.concatenate([Y_blue, Y_green, Y_yellow]))

    PCA_neutral(X, Y)

    np.save(f"data/classification/X_classifier.npy", X)
    np.save(f"data/classification/Y_classifier.npy", Y)


    # Create labeled testing data as well with neutral zone
    X_blue_test = np.load("data/blue/matrices/X_blue_test.npy")
    X_green_test = np.load("data/green/matrices/X_green_test.npy")
    X_yellow_test = np.load("data/yellow/matrices/X_yellow_test.npy")

    Y_test_neutral_blue1= np.vstack(np.ones(456)* 0)
    Y_test_blue1 = np.vstack(np.ones(len(X_blue_test[456:])) * 1)
    Y_blue_test = np.concatenate([Y_test_neutral_blue1, Y_test_blue1])

    Y_test_neutral_green1= np.vstack(np.ones(330)* 0)
    Y_test_green1 = np.vstack(np.ones(len(X_green_test[330:])) * 2)
    Y_green_test = np.concatenate([Y_test_neutral_green1, Y_test_green1])

    Y_test_neutral_yellow1= np.vstack(np.ones(180)* 0)
    Y_test_yellow1 = np.vstack(np.ones(len(X_yellow_test[180:])) * 3)
    Y_yellow_test = np.concatenate([Y_test_neutral_yellow1, Y_test_yellow1])

    X_test = np.concatenate([X_blue_test, X_green_test, X_yellow_test])
    Y_test = np.vstack(np.concatenate([Y_blue_test, Y_green_test, Y_yellow_test]))

    df_test = pd.DataFrame(X_test, columns=["L", "A", "B", "L_var", "A_var", "B_var"])
    df_test.insert(3, "L^2", df_test["L"]**2)
    df_test.insert(4, "A^2", df_test["A"]**2)
    df_test.insert(5, "B^2", df_test["B"]**2)
    df_test.insert(6, "LA", df_test["L"] * df_test["A"])
    df_test.insert(7, "LB", df_test["L"] * df_test["B"])
    df_test.insert(8, "AB", df_test["A"] * df_test["B"])

    PCA_neutral(X_test, Y_test)

    X_test = df_test.to_numpy()
    np.save(f"data/classification/X_test_classifier.npy", X_test)
    np.save(f"data/classification/Y_test_classifier.npy", Y_test)