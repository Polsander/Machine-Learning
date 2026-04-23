This is the work layout for the ChE 500 project for computer vision.


# General Findings

## Classification

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Linear SVM | 0.991 ± 0.019 | 0.991 ± 0.018 | 0.991 ± 0.019 | 0.990 ± 0.020 |
| Kernel SVM | 0.990 ± 0.019 | 0.991 ± 0.017 | 0.990 ± 0.019 | 0.990 ± 0.020 |
| XGB Forest | 0.859 ± 0.304 | 0.875 ± 0.297 | 0.859 ± 0.304 | 0.848 ± 0.313 |
| Random Forest | 0.992 ± 0.019 | 0.993 ± 0.018 | 0.992 ± 0.019 | 0.992 ± 0.020 |
| Neural Network | 0.991 ± 0.015 | 0.992 ± 0.015 | 0.991 ± 0.015 | 0.991 ± 0.016 |


#### Specifications

| Model | Configuration |
|-------|---|
| Linear SVM | C = 1 |
| Kernel SVM | C = 1,  Kernel: RBF, gamma = 0.32 |
| XGB Random Forest | trees = 75, max depth = 5, learning rate = 0.01 |
| Random Forest | trees = 150, max depth = 6 |
| Neural Network | Layers: 30 → 40 → 20 → 4, epochs = 350, learning rate = 0.001 |



## Regression

### 1. For Blue Regression Model

| Model | MSE | Std MSE |
|-------|-----|---------|
| Support Vector Regression | 1.215 | 0.028 |
| Gaussian Regression | 3.467 | 0.516 |
| Neural Network | 2.861 | 1.495 |


### 2. For Green Regression Model

| Model | MSE | Std MSE |
|-------|-----|---------|
| Support Vector Regression | 1.217 | 0.044 |
| Gaussian Regression | 1.701 | 0.458 |
| Neural Network | 1.189 | 0.386 |



### 3. For Yellow Regression Model

| Model | MSE | Std MSE |
|-------|-----|---------|
| Support Vector Regression | 0.588 | 0.021 |
| Gaussian Regression | 2.253 | 0.543 |
| Neural Network | 1.437 | 0.211 |

#### Specifications:

Specifications were the same for all models for each colour application.

| Model | Configuration |
|-------|---|
| SVR | Kernel: RBF, C = 1, Gamma = 0.85, epsilon=0.1 |
| GP | Kernel: Matern, 20% of data used (sample size constraints) |
| NN | Layers: 30 → 40 → 15 → 1, epochs = 40 |
