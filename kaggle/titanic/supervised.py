import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from unsupervised import get_PLS_space
import sys

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
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    log_scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    randfor_scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    svm_scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}


    for i, (train_idx, test_idx) in enumerate(kf.split(Xraw, Yraw)):
        print("iteration:", i)

        ct = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), [3, 6])],
            remainder="passthrough"
        )

        X_cross_train = ct.fit_transform(Xraw[train_idx])
        X_cross_test = ct.transform(Xraw[test_idx])
        Y_cross_train, Y_cross_test = Yraw[train_idx], Yraw[test_idx]

        # Instantiate models
        log_reg = Logit_Reg()
        rand_for = Desc_Tree()
        svm = SVM_Cls()

        #fit
        log_reg.fit(X_cross_train, Y_cross_train)
        rand_for.fit(n_trees=80, max_depth=6, X=X_cross_train, Y=Y_cross_train)
        svm.fit(X_cross_train, Y_cross_train, C=0.7, gamma=0.15)

        # Predict
        Y_log = log_reg.predict(X_cross_test)
        Y_for = rand_for.predict(X_cross_test)
        Y_svm = svm.predict(X_cross_test)

        # calculate scores
        logit_score = log_reg.get_metrics(Y_log, Y_cross_test)
        for_score = rand_for.get_metrics(Y_for, Y_cross_test)
        svm_score = svm.get_metrics(Y_svm, Y_cross_test)

        for j, metric in enumerate(['acc', 'prec', 'rec', 'f1']):
            log_scores[metric].append(logit_score[j])
            randfor_scores[metric].append(for_score[j])
            svm_scores[metric].append(svm_score[j])
        
    print("====== Cross Validation Results =======")
    print("Logistic Regression:\n",
           f"Accuracy: {np.mean(log_scores['acc']):.3f}\n",
           f"Precision: {np.mean(log_scores['prec']):.3f}\n",
           f"Recall: {np.mean(log_scores['rec']):.3f}\n",
           f"f1 score: {np.mean(log_scores['f1']):.3f}\n")
    
    print("Random Forest:\n",
           f"Accuracy: {np.mean(randfor_scores['acc']):.3f}\n",
           f"Precision: {np.mean(randfor_scores['prec']):.3f}\n",
           f"Recall: {np.mean(randfor_scores['rec']):.3f}\n",
           f"f1 score: {np.mean(randfor_scores['f1']):.3f}\n")
    
    print("Support Vector Classifier:\n",
           f"Accuracy: {np.mean(svm_scores['acc']):.3f}\n",
           f"Precision: {np.mean(svm_scores['prec']):.3f}\n",
           f"Recall: {np.mean(svm_scores['rec']):.3f}\n",
           f"f1 score: {np.mean(svm_scores['f1']):.3f}\n")



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


    # print("\n====== For PLS Data ======")
    # X_train, X_test, Y_train, Y_test = train_test_split(X_pls, Yraw, test_size=0.33)

    # log_reg = Logit_Reg()
    # log_reg.fit(X_train, Y_train)
    # Y_pred = log_reg.predict(X_test)
    # log_reg.plot_confusion_matrix(Y_pred, Y_test)
    # acc, pre, rec, f1 = log_reg.get_metrics(Y_pred, Y_test)
    # print("Logistic Regression:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")

    # rand_for = Desc_Tree()
    # rand_for.fit(80, 6, X_train, Y_train)
    # Y_pred = rand_for.predict(X_test)
    # rand_for.plot_confusion_matrix(Y_pred, Y_test)
    # acc, pre, rec, f1 = rand_for.get_metrics(Y_pred, Y_test)
    # print("Random Forest:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")

    # svm = SVM_Cls()
    # svm.fit(X_train, Y_train)
    # Y_pred = svm.predict(X_test)
    # svm.plot_confusion_matrix(Y_pred, Y_test)
    # acc, pre, rec, f1 = svm.get_metrics(Y_pred, Y_test)
    # print("Support Vector Classifer:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")

    cross_validation(K=10, Xraw=Xraw, Yraw=Yraw)
    print("Random forest is working the best here")

