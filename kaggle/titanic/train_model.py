import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys

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


if __name__ == "__main__":

    user_input = input("Are you ready to continue to run the defined code to use your model to predict and save the submission file? (y/n)")
    if user_input.lower() != 'y':
        sys.exit(1)
    

    Xraw = np.load("matrices/train/X.npy", allow_pickle=True)
    Yraw = np.load("matrices/train/Y.npy", allow_pickle=True)


    ct = ColumnTransformer(
    transformers=[("scaler", StandardScaler(), [3, 6])],
        remainder="passthrough"
    )
    X_train, X_test, Y_train, Y_test = train_test_split(Xraw, Yraw, test_size=0.33)

    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)


    Random_Forest_Classifier = Desc_Tree()

    Random_Forest_Classifier.fit(n_trees=80, max_depth=6, X=X_train, Y=Y_train)
    Y_pred = Random_Forest_Classifier.predict(X_test)
    Random_Forest_Classifier.plot_confusion_matrix(Y_pred, Y_test)
    acc, pre, rec, f1 = Random_Forest_Classifier.get_metrics(Y_pred, Y_test)
    print("Random Forest:\n", f"Accuracy: {acc:.3f}\n", f"Precision: {pre:.3f}\n", f"Recall: {rec:.3f}\n", f"f1 score: {f1:.3f}\n")
    

    user_input = input("Generate submission? (y/n)")
    if user_input.lower() != 'y':
        sys.exit(1)
    
    new_X = np.load("matrices/test/X.npy", allow_pickle=True)
    ids = np.load("matrices/test/ids.npy", allow_pickle=True)

    X_test = ct.transform(new_X)
    Y_pred = Random_Forest_Classifier.predict(X_test)
    df = pd.DataFrame()

    df['PassengerId'] = ids.ravel()
    df['Survived'] = Y_pred.ravel()
    df.to_csv('submission_OE.csv', index=False)
