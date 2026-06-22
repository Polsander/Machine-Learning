import numpy as np
import pandas as pd

def process_age_fare(column):

    new_col = []
    age_values = []

    for i in range(len(column)):
        if not np.isnan(column[i]):
            age_values.append(column[i])
    
    # Assign average to NaN
    mean = round(np.mean(age_values))
    print('mean', mean)
    
    for i in range(len(column)):
        if np.isnan(column[i]):
            new_col.append(mean)
        else:
            new_col.append(column[i])
    return new_col


def one_hot_encode_port(column):
    S = []
    Q = []
    C = []

    for i in range(len(column)):
        value = column[i]

        match(value):

            case("S"):
                S.append(1)
                Q.append(0)
                C.append(0)
            case("Q"):
                S.append(0)
                Q.append(1)
                C.append(0)
            case("C"):
                S.append(0)
                Q.append(0)
                C.append(1)
            case _:
                S.append(1)
                Q.append(0)
                C.append(0)
    return S, Q, C



def male_female(column):
    # make male 1 and female -1
    new_col = []
    for i in range(len(column)):
        if column[i] == "male":
            new_col.append(1)
        else:
            new_col.append(-1)
    return new_col

def process_data(pd_data: pd.DataFrame, type=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The full process data function which supports being imported/exported"""
    data_copy: pd.DataFrame = pd_data
    print("Any NaN values before processing?", data_copy.isna().values.any())

    data_copy = data_copy.drop(columns=["Ticket", "Cabin", "Name"])
    data_copy["Age"] = process_age_fare(data_copy["Age"])
    data_copy["Sex"] = male_female(data_copy["Sex"])
    data_copy["S"], data_copy["Q"], data_copy["C"] = one_hot_encode_port(data_copy["Embarked"])
    data_copy["Fare"] = process_age_fare(data_copy["Fare"])
    data_copy = data_copy.drop(columns=["Embarked"])
    data_copy["SibSp"] = data_copy["SibSp"].fillna(0)
    data_copy["Parch"] = data_copy["Parch"].fillna(0)

    print("Any NaN values after processing?", data_copy.isna().values.any())
    
    Y = None if type is not None else data_copy["Survived"].to_numpy()
    ids = data_copy["PassengerId"].to_numpy()
    X = None
    if type is None:
        X = data_copy.drop(columns=["Survived", "PassengerId"]).to_numpy()
    else:
        X = data_copy.drop(columns=["PassengerId"]).to_numpy()

    return X, Y, ids


if __name__ == "__main__":

    train_file_data = pd.read_csv("data/train.csv")
    X, Y, ids = process_data(train_file_data)

    np.save("matrices/train/X.npy", X)
    np.save("matrices/train/Y.npy", Y)
    np.save("matrices/train/ids.npy", ids)
    print("Training data processing complete and matrices saved in folders")

    test_file_data = pd.read_csv("data/test.csv")
    X, Y, ids = process_data(test_file_data, type='test')

    np.save("matrices/test/X.npy", X)
    np.save("matrices/test/ids.npy", ids)