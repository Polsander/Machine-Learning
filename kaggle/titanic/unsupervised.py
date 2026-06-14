import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

def PCA_vis(X, Y):
    
    pca_model2 = PCA(n_components=2)
    pca_model3 = PCA(n_components=3)
    T2 = pca_model2.fit_transform(X)
    T3 = pca_model3.fit_transform(X)

    fig = plt.figure(figsize=(14, 5))
    cmap="PiYG"

    # 2D plot (PC1 vs PC2)
    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(T2[:,0], T2[:,1], c=Y, cmap=cmap, s=20)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("PCA 2D Reduction (PC1 vs PC2)")
    plt.colorbar(scatter1, ax=ax1, label="Class")
    
    # 3D plot (PC1 vs PC2 vs PC3)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(T3[:,0], T3[:,1], T3[:,2], c=Y, cmap=cmap, s=1)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("PCA 3D Reduction")
    plt.colorbar(scatter2, ax=ax2, label="Class")

    plt.tight_layout()
    plt.show()


def PLS_visualize(X,Y):
    # PLS 3D
    X_tilde = X.copy()
    Y = np.vstack(Y)
    breakpoint()

    k = 9
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
    scatter = ax.scatter(T[:,0], T[:,1], T[:,2], c=Y, cmap="PiYG", s=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PLS 3D Reduction")
    plt.colorbar(scatter)

    plt.show()


if __name__ == "__main__":

    Xraw = np.load("matrices/train/X.npy", allow_pickle=True)
    Yraw = np.load("matrices/train/Y.npy", allow_pickle=True)

    print("Column indices are as follows:\n")
    print("1        2   3     4       5     6    7  8  9")
    print("Pclass, Sex, Age, SibSp, Parch, Fare, S, Q, C\n")
    # scaling X first (use standard scaling).
    # Consider scaling these columnds [Age, Fare]
    ct = ColumnTransformer(
    transformers=[("scaler", StandardScaler(), [3, 6])],
    remainder="passthrough"
    )

    X_scaled = ct.fit_transform(Xraw)

    PCA_vis(X_scaled, Yraw)
    PLS_visualize(X_scaled, Yraw) # partial least squares seems to be the most promising here


    
