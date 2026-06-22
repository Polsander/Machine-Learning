import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from unsupervised import get_PLS_space

class Logit_Reg():
    def __init__(self):
        self.model = None
    def fit(self, X, Y):
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, Y)
        self.model = model
    def predict(self, X):
        Y_pred = self.model.predict(X)
        return Y_pred
    def plot_confusion_matrix(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred, labels=[1,0])
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["survived", "died"])
        disp.plot()
        plt.title("Confusion Matrix Display")
        plt.show()
    def get_metrics(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='binary')
        recall = recall_score(Y_test, Y_pred, average='binary')
        f1 = f1_score(Y_test, Y_pred, average='binary')
        return (accuracy, precision, recall, f1)

class Desc_Tree():
    def __init__(self):
        self.model = None
    def fit(self, n_trees, max_depth, X, Y):
        model = RandomForestClassifier(n_trees, max_depth=max_depth, random_state=42)
        model.fit(X, Y)
        self.model = model
    def predict(self, X):
        Y_pred = self.model.predict(X)
        return Y_pred
    def plot_confusion_matrix(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred, labels=[1,0])
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["survived", "died"])
        disp.plot()
        plt.title("Confusion Matrix Display - Random Forest")
        plt.show()
    def get_metrics(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='binary')
        recall = recall_score(Y_test, Y_pred, average='binary')
        f1 = f1_score(Y_test, Y_pred, average='binary')
        return (accuracy, precision, recall, f1)

class SVM_Cls():
    def __init__(self):
        self.model = None
    def fit(self, X, Y, C=1, gamma="scale", tol=0.001):
        model = SVC(C=C, kernel="rbf", gamma=gamma, tol=tol, random_state=42)
        model.fit(X, Y)
        self.model = model
    def predict(self, X):
        Y_pred = self.model.predict(X)
        return Y_pred
    def plot_confusion_matrix(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred)
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["survived", "died"])
        disp.plot()
        plt.show()
    def get_metrics(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='binary')
        recall = recall_score(Y_test, Y_pred, average='binary')
        f1 = f1_score(Y_test, Y_pred, average='binary')
        return (accuracy, precision, recall, f1)


def cross_validation(K, Xraw, Yraw):
    pass


if __name__ == "__main__":
    Xraw = np.load("matrices/train/X.npy", allow_pickle=True)
    Yraw = np.load("matrices/train/Y.npy", allow_pickle=True)


    ct = ColumnTransformer(
    transformers=[("scaler", StandardScaler(), [3, 6])],
        remainder="passthrough"
    )
    X_scaled = ct.fit_transform(Xraw)
    X_pls = get_PLS_space(X_scaled, Yraw, k=7)

    # Begin supervised learning - try on X_scaled and X_pls and see which one works better

    print("====== For Raw Data ======")
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Yraw, test_size=0.33)

    log_reg = Logit_Reg()
    log_reg.fit(X_train, Y_train)
    Y_pred = log_reg.predict(X_test)
    log_reg.plot_confusion_matrix(Y_pred, Y_test)
    acc, pre, rec, f1 = log_reg.get_metrics(Y_pred, Y_test)
    print("Logistic Regression:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")

    rand_for = Desc_Tree()
    rand_for.fit(80, 6, X_train, Y_train)
    Y_pred = rand_for.predict(X_test)
    rand_for.plot_confusion_matrix(Y_pred, Y_test)
    acc, pre, rec, f1 = rand_for.get_metrics(Y_pred, Y_test)
    print("Random Forest:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")

    svm = SVM_Cls()
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_test)
    svm.plot_confusion_matrix(Y_pred, Y_test)
    acc, pre, rec, f1 = svm.get_metrics(Y_pred, Y_test)
    print("Support Vector Classifer:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")


    print("\n====== For PLS Data ======")
    X_train, X_test, Y_train, Y_test = train_test_split(X_pls, Yraw, test_size=0.33)

    log_reg = Logit_Reg()
    log_reg.fit(X_train, Y_train)
    Y_pred = log_reg.predict(X_test)
    log_reg.plot_confusion_matrix(Y_pred, Y_test)
    acc, pre, rec, f1 = log_reg.get_metrics(Y_pred, Y_test)
    print("Logistic Regression:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")

    rand_for = Desc_Tree()
    rand_for.fit(80, 6, X_train, Y_train)
    Y_pred = rand_for.predict(X_test)
    rand_for.plot_confusion_matrix(Y_pred, Y_test)
    acc, pre, rec, f1 = rand_for.get_metrics(Y_pred, Y_test)
    print("Random Forest:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")

    svm = SVM_Cls()
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_test)
    svm.plot_confusion_matrix(Y_pred, Y_test)
    acc, pre, rec, f1 = svm.get_metrics(Y_pred, Y_test)
    print("Support Vector Classifer:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")
