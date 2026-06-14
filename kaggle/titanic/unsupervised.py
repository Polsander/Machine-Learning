import numpy as np

if __name__ == "__main__":

    X = np.load("matrices/train/X.npy", allow_pickle=True)
    Y = np.load("matrices/train/Y.npy", allow_pickle=True)

    print("Column indices are as follows:\n")
    print("1        2   3     4       5     6    7  8  9")
    print("Pclass, Sex, Age, SibSp, Parch, Fare, S, Q, C\n")
    # scaling X first (use standard scaling).
    # Consider scaling these columnds [Age, Fare]