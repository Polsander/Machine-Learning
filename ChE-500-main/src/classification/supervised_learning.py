import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from unsupervised_learning import scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from neural_network_template import Neural_Network
from xgboost import XGBClassifier
import torch



class Linear_SVM():
    def __init__(self):
        self.model = None
    def fit(self, X_scaled, Y_scaled):
        self.model = SVC(C=1, kernel='linear', random_state=42)
        self.model.fit(X_scaled, Y_scaled.ravel())
    def predict(self, X_scaled):
        Y_pred = self.model.predict(X_scaled)
        return Y_pred
    def show_metrics(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3])    
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["Neutral", "Blue", "Green", "Yellow"])
        disp.plot()
        plt.title("Linear SVM Confusion Matrix")
        plt.show()
    
    def get_scores(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    @staticmethod
    def show_decision_boundary(X_train, X_test, Y_train, Y_test, dim=2):
        # dim = 2 or 3
        pca_transformation = PCA(n_components=dim)
        T_train = pca_transformation.fit_transform(X_train)
        T_test = pca_transformation.transform(X_test)
        model = SVC(C=1, kernel="linear", random_state=42)
        model.fit(T_train, Y_train.ravel())

        plot_decision_boundary((T_train, T_test), (Y_train, Y_test), dim, model)

class Kernel_SVM():
    def __init__(self):
        self.model = None
    def fit(self, X_scaled, Y_scaled):
        self.model = SVC(C=1, kernel='rbf', gamma=0.32, random_state=42)
        self.model.fit(X_scaled, Y_scaled.ravel())
    def predict(self, X_scaled):
        Y_pred = self.model.predict(X_scaled)
        return Y_pred
    def show_metrics(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3])    
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["Neutral", "Blue", "Green", "Yellow"])
        disp.plot()
        plt.title("Kernel SVM Confusion Matrix")
        plt.show()
    
    def get_scores(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    @staticmethod
    def show_decision_boundary(X_train, X_test, Y_train, Y_test, dim=2):
        # dim = 2 or 3
        pca_transformation = PCA(n_components=dim)
        T_train = pca_transformation.fit_transform(X_train)
        T_test = pca_transformation.transform(X_test)
        model = SVC(C=1, kernel="rbf", gamma=0.32)
        model.fit(T_train, Y_train.ravel())

        plot_decision_boundary((T_train, T_test), (Y_train, Y_test), dim, model)
    

class Forest_XGB():
    def __init__(self, n_trees, max_depth, learning_rate, random_state=42):
        self.model = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
    def fit(self, X_scaled, Y_train):
        gb_model = XGBClassifier(
            booster = 'gbtree',
            n_estimators = self.n_trees,
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            random_state = self.random_state
        )
        gb_model.fit(X_scaled, Y_train.ravel())
        self.model = gb_model

    def predict(self, X_scaled):
        y_pred = self.model.predict(X_scaled)
        return y_pred
    
    def show_metrics(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3])    
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["Neutral", "Blue", "Green", "Yellow"])
        disp.plot()
        plt.title("Random Forest XGB Confusion Matrix")
        plt.show()
    
    def get_scores(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def show_decision_boundary(self, X_train, X_test, Y_train, Y_test, dim=2):
        pca_transformation = PCA(n_components=dim)
        T_train = pca_transformation.fit_transform(X_train)
        T_test = pca_transformation.transform(X_test)
        gb_model = XGBClassifier(
            booster = 'gbtree',
            n_estimators = self.n_trees,
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            random_state = self.random_state
        )
        gb_model.fit(T_train, Y_train.ravel())
        plot_decision_boundary((T_train, T_test), (Y_train, Y_test), dim, gb_model)


class Random_Forest():
    def __init__(self, n_trees, max_depth = None):
        self.model = None
        self.n_trees = n_trees
        self.max_depth = max_depth

    def fit(self, X_scaled, Y_train):
        model = RandomForestClassifier(n_estimators=self.n_trees, max_depth=self.max_depth, random_state=11)
        model.fit(X_scaled, Y_train.ravel())
        self.model = model
    
    def predict(self, X_scaled):
        Y_predict = self.model.predict(X_scaled)
        return Y_predict
    
    def show_metrics(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3])    
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["Neutral", "Blue", "Green", "Yellow"])
        disp.plot()
        plt.title("Random Forest Confusion Matrix")
        plt.show()
    
    def get_scores(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def show_decision_boundary(self, X_train, X_test, Y_train, Y_test, dim=2):
        pca_transformation = PCA(n_components=dim)
        T_train = pca_transformation.fit_transform(X_train)
        T_test = pca_transformation.transform(X_test)
        model = RandomForestClassifier(n_estimators=self.n_trees, max_depth=self.max_depth)
        model.fit(T_train, Y_train.ravel())
        plot_decision_boundary((T_train, T_test), (Y_train, Y_test), dim, model)
        


class Neural_Network_Classifier():
    def __init__(self, input_size, layer_sizes, random_state=None):
        self.model = Neural_Network(input_size, layer_sizes)
        self.epochs = []
        self.validation_losses = []
        self.training_losses = []

        if random_state is not None:
            torch.manual_seed(random_state)

    def train(self,  X_scaled_train, Y_scaled_train, epochs = 100, lr=0.001, X_test = None, Y_test = None):
        criterion = torch.nn.CrossEntropyLoss()
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

    def predict(self, X_scaled):
        self.model.eval()
        with torch.no_grad():
            X_tensor, _ = self.model.convert_to_tensors(X_scaled, np.array([]))
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            Y_pred = torch.argmax(probs, dim=1)
            return Y_pred.numpy()
    
    def show_metrics(self, Y_pred, Y_test):
        conf_mat = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3])    
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["Neutral", "Blue", "Green", "Yellow"])
        disp.plot()
        plt.title("Neural Network Classifier")
        plt.show()
    
    def get_scores(self, Y_pred, Y_test):
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, average='weighted')
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def plot_training_curve(self):

        plt.plot(self.epochs, self.training_losses, label="training loss")
        plt.plot(self.epochs, self.validation_losses, label="validation loss")
        plt.xlabel("epochs")
        plt.ylabel("Cross Entropy loss")
        plt.title("NN Training and Validation Loss")
        plt.legend()
        plt.show()


def plot_decision_boundary(T_data, Y_actual, n_dim, model):
    T_train, T_test = T_data
    Y_train, Y_test = Y_actual
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()
    
    colors = ['grey', 'blue', 'green', 'yellow']
    unique_classes = np.unique(np.concatenate([Y_train, Y_test]))
    
    if n_dim == 2:
        # 2D Decision Boundary - combined plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot decision boundary
        label_map = {0:"netural", 1:"blue", 2:"green", 3:"yellow"}
        custom_cmap = ListedColormap(colors)
        bounds = np.arange(len(colors) + 1) - 0.5
        custom_norm = BoundaryNorm(bounds, len(colors))
        DecisionBoundaryDisplay.from_estimator(model, T_train, ax=ax, 
                                               cmap=custom_cmap, norm=custom_norm,
                                                response_method='predict', alpha=0.3)
        
        for cls in unique_classes:
            mask = Y_train == cls
            ax.scatter(T_train[mask, 0], T_train[mask, 1],
                    c=colors[int(cls)] if int(cls) < len(colors) else 'black',
                    label=f'Class {label_map[int(cls)]}', s=100, marker='o', edgecolors='k')

        # Plot test points (x marks) - single legend entry
        first = True
        for cls in unique_classes:
            mask = Y_test == cls
            ax.scatter(T_test[mask, 0], T_test[mask, 1],
                    c=colors[int(cls)] if int(cls) < len(colors) else 'black',
                    label='Test Data' if first else '_nolegend_',
                    s=100, marker='X', edgecolors='k')
            first = False
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Decision Boundary - Training (o) and Test (x) Sets")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def run_supervised_training(Xraw, Yraw, Xraw_test, Yraw_test):
    X, x_scaler = scale(Xraw)
    X_train = X
    Y_train = Yraw
    X_test = x_scaler.transform(Xraw_test)
    Y_test = Yraw_test

    #=== Try SVM ===
    linear_svm = Linear_SVM()
    linear_svm.fit(X_train, Y_train)
    linear_svm.show_decision_boundary(X_train, X_test, Y_train, Y_test, dim=2) # Estimate
    Y_pred = linear_svm.predict(X_test)
    linear_svm.show_metrics(Y_pred, Y_test)

    print("Linear SVM SSE:          ", np.sum((Y_pred - Y_test.ravel())**2))

    #=== Try Kernel SVM ===
    kernel_svm = Kernel_SVM()
    kernel_svm.fit(X_train, Y_train)
    kernel_svm.show_decision_boundary(X_train, X_test, Y_train, Y_test, dim=2)
    Y_pred = kernel_svm.predict(X_test)
    kernel_svm.show_metrics(Y_pred, Y_test)

    print("Kernel SVM SSE:          ", np.sum((Y_pred - Y_test.ravel())**2))


    # Try Random Forest XGB
    random_forest_xgb = Forest_XGB(n_trees=75, max_depth=5, learning_rate=0.01)
    random_forest_xgb.fit(X_train, Y_train)
    Y_pred = random_forest_xgb.predict(X_test)
    random_forest_xgb.show_metrics(Y_pred, Y_test)
    random_forest_xgb.show_decision_boundary(X_train, X_test, Y_train, Y_test)

    print("Random Forest XGB SSE:   ", np.sum((Y_pred - Y_test.ravel())**2))


    # Try Random Forest
    random_forest = Random_Forest(n_trees=150, max_depth=6)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.show_metrics(Y_pred, Y_test)
    random_forest.show_decision_boundary(X_train, X_test, Y_train, Y_test)

    print("Random Forest SSE:   ", np.sum((Y_pred - Y_test.ravel())**2))


    # Try Neural Network
    neural_net_class = Neural_Network_Classifier(len(X_train[0]), [15,30,20, 4], random_state=42)
    neural_net_class.train(X_train, Y_train, X_test=X_test, Y_test=Y_test, epochs=350)
    neural_net_class.plot_training_curve()
    Y_pred = neural_net_class.predict(X_test)
    neural_net_class.show_metrics(Y_pred, Y_test)

    print("Neural Network SSE:   ", np.sum((Y_pred - Y_test.ravel())**2))


    # Run KFOLD K = 10 cross validation
    print ("=== Running KFOLD K = 10 Cross Validation ===")

    k = 10
    kf = KFold(n_splits=k)

    lin_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    kern_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    xgb_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    rand_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    neural_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):
        print("iteration", i+1)

        x_cross_train, y_cross_train = X_train[train_idx, :], Y_train[train_idx, :]
        x_cross_test, y_cross_test = X_test[test_idx, :], Y_test[test_idx, :]

        # Instatiate models
        linear_svm = Linear_SVM()
        kernel_svm = Kernel_SVM()
        xgb_forest = Forest_XGB(n_trees=75, max_depth=5, learning_rate=0.01)
        random_forest = Random_Forest(n_trees=150, max_depth=6)
        neural_net = Neural_Network_Classifier(len(x_cross_train[0]), [15, 30, 20, 4], random_state=42)

        # train/fit
        linear_svm.fit(x_cross_train, y_cross_train)
        kernel_svm.fit(x_cross_train, y_cross_train)
        xgb_forest.fit(x_cross_train, y_cross_train)
        random_forest.fit(x_cross_train, y_cross_train)
        neural_net.train(x_cross_train, y_cross_train, epochs=350)

        # predict
        y_linear = linear_svm.predict(x_cross_test)
        y_kernel = kernel_svm.predict(x_cross_test)
        y_xgb_forest = xgb_forest.predict(x_cross_test)
        y_rand_forest = random_forest.predict(x_cross_test)
        y_neural = neural_net.predict(x_cross_test)

        # Calculate scores
        lin_fold_scores = linear_svm.get_scores(y_linear, y_cross_test)
        kern_fold_scores = kernel_svm.get_scores(y_kernel, y_cross_test)
        xgb_fold_scores = xgb_forest.get_scores(y_xgb_forest, y_cross_test)
        rand_fold_scores = random_forest.get_scores(y_rand_forest, y_cross_test)
        neural_fold_scores = neural_net.get_scores(y_neural, y_cross_test)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            lin_scores[metric].append(lin_fold_scores[metric])
            kern_scores[metric].append(kern_fold_scores[metric])
            xgb_scores[metric].append(xgb_fold_scores[metric])
            rand_scores[metric].append(rand_fold_scores[metric])
            neural_scores[metric].append(neural_fold_scores[metric])
    
    print("\n=== Cross-Validation Results ===")
    print(f"Linear SVM:      Accuracy: {np.mean(lin_scores['accuracy']):.3f} (±{np.std(lin_scores['accuracy']):.3f}), F1: {np.mean(lin_scores['f1']):.3f} (±{np.std(lin_scores['f1']):.3f}), Precision: {np.mean(lin_scores['precision']):.3f} (±{np.std(lin_scores['precision']):.3f}), Recall: {np.mean(lin_scores['recall']):.3f} (±{np.std(lin_scores['recall']):.3f})")
    print(f"Kernel SVM:      Accuracy: {np.mean(kern_scores['accuracy']):.3f} (±{np.std(kern_scores['accuracy']):.3f}), F1: {np.mean(kern_scores['f1']):.3f} (±{np.std(kern_scores['f1']):.3f}), Precision: {np.mean(kern_scores['precision']):.3f} (±{np.std(kern_scores['precision']):.3f}), Recall: {np.mean(kern_scores['recall']):.3f} (±{np.std(kern_scores['recall']):.3f})")
    print(f"XGB Forest:      Accuracy: {np.mean(xgb_scores['accuracy']):.3f} (±{np.std(xgb_scores['accuracy']):.3f}), F1: {np.mean(xgb_scores['f1']):.3f} (±{np.std(xgb_scores['f1']):.3f}), Precision: {np.mean(xgb_scores['precision']):.3f} (±{np.std(xgb_scores['precision']):.3f}), Recall: {np.mean(xgb_scores['recall']):.3f} (±{np.std(xgb_scores['recall']):.3f})")
    print(f"Random Forest:   Accuracy: {np.mean(rand_scores['accuracy']):.3f} (±{np.std(rand_scores['accuracy']):.3f}), F1: {np.mean(rand_scores['f1']):.3f} (±{np.std(rand_scores['f1']):.3f}), Precision: {np.mean(rand_scores['precision']):.3f} (±{np.std(rand_scores['precision']):.3f}), Recall: {np.mean(rand_scores['recall']):.3f} (±{np.std(rand_scores['recall']):.3f})")
    print(f"Neural Network:  Accuracy: {np.mean(neural_scores['accuracy']):.3f} (±{np.std(neural_scores['accuracy']):.3f}), F1: {np.mean(neural_scores['f1']):.3f} (±{np.std(neural_scores['f1']):.3f}), Precision: {np.mean(neural_scores['precision']):.3f} (±{np.std(neural_scores['precision']):.3f}), Recall: {np.mean(neural_scores['recall']):.3f} (±{np.std(neural_scores['recall']):.3f})")


if __name__ == "__main__":
    X = np.load("data/classification/X_classifier.npy")
    Y = np.load("data/classification/Y_classifier.npy")
    X_test = np.load("data/classification/X_test_classifier.npy")
    Y_test = np.load("data/classification/Y_test_classifier.npy")



    # Now we start to work our magic!
    run_supervised_training(X, Y, X_test, Y_test)