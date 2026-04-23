import numpy as np
import matplotlib.pyplot as plt
import os
from data_processing import get_LAB_values

def generate_y_label(frame_path, base= None):
    L, A, B = get_LAB_values(frame_path)
    L, A, B = np.mean(L), np.mean(A), np.mean(B)
    if not base:
        return L, A, B
    L_base, A_base, B_base = base
    
    #Execute delta_E equation
    delta_E = np.sqrt((L - L_base)**2 + (A - A_base)**2 + (B - B_base)**2)
    # if delta_E > 60:
    #     print(frame_path)
    #     breakpoint()
    return delta_E

def generate_color_labels(X, color):
    Y = np.zeros((len(X), 1))

    for i in range(len(X)):
        Y[i] = color
    
    return Y

def generate_x_features(frame_path):
    
    L, A, B = get_LAB_values(frame_path)
    L_var, A_var, B_var = np.var(L), np.var(A), np.var(B)
    L, A, B = np.mean(L), np.mean(A), np.mean(B)

    return [L, A, B, L_var, A_var, B_var]

if __name__ == "__main__":


    colour = "green"
    number = "4"
    key = colour + number
    # folder_path = f"data/{colour}/frames{number}"
    folder_path = f"data/{colour}/test"
    base_path = folder_path + "/frame0.jpg"

    Y = []
    X = []

    base = generate_y_label(base_path)
    # Then we pretty much loop and get any frame we want lol

    for i in range(0,len(os.listdir(folder_path))):
        print("step", i)
        full_path = folder_path + f"/frame{i}.jpg"
        Y_i = generate_y_label(full_path, base)
        Y.append(Y_i)
        X_i = generate_x_features(full_path)
        X.append(X_i)
    
    Y = np.array(Y)
    X = np.array(X)

    # Save variables
    np.save(f"data/{colour}/matrices/X_{key}.npy", X)
    np.save(f"data/{colour}/matrices/Y_{key}.npy", Y)

