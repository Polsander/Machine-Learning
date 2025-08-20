import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle

def convert_image_to_array(image):
    arr = cv2.imread(image) # Reads image in BGR format
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    return arr


def convert_to_grayscale(image_arr):
    rgb_weights = [0.2989, 0.5870, 0.1140]

    gray = np.dot(image_arr[...,:3], rgb_weights)
    return gray

def resize_image(image_arr, img_size):
    img_res = cv2.resize(image_arr, (img_size, img_size))

    return img_res


def load_images_to_array(cars_path: str, bike_path: str, tank_path:str, num_samples: int, img_size: int) -> tuple[np.array, np.array]:
    
    X = np.zeros((num_samples*3, img_size**2))
    Y = np.zeros((num_samples*3, 3), dtype=int)

    sample_num = 0

    car_number = 0
    for jpeg in os.scandir(cars_path):
        if car_number > 567 : break
        arr = convert_image_to_array(jpeg)
        arr_res = resize_image(arr, img_size)
        gray_arr = convert_to_grayscale(arr_res) / 255
        X[sample_num] = gray_arr.flatten()
        Y[sample_num] = np.array([1, 0, 0])
        car_number += 1
        sample_num += 1

    bike_number = 0
    for jpeg in os.scandir(bike_path):
        if bike_number > 567 : break
        arr = convert_image_to_array(jpeg)
        arr_res = resize_image(arr, img_size)
        gray_arr = convert_to_grayscale(arr_res) / 255 # normalize
        X[sample_num] = gray_arr.flatten() 
        Y[sample_num] = np.array([0, 1, 0])
        bike_number += 1
        sample_num += 1
    
    for jpeg in os.scandir(tank_path):
        arr = convert_image_to_array(jpeg)
        arr_res = resize_image(arr, img_size)
        gray_arr = convert_to_grayscale(arr_res) / 255
        X[sample_num] = gray_arr.flatten()
        Y[sample_num] = np.array([0, 0, 1])
        sample_num += 1

    return (X, Y)


def test_images_to_dict(test_path: str, img_size: int):

    cars_path = f"{test_path}/car"
    bike_path = f"{test_path}/bike"
    tank_path = f"{test_path}/tank"

    test_dict = {}

    for i, jpeg in enumerate(os.scandir(cars_path)):

        arr = convert_image_to_array(jpeg)
        arr_res = resize_image(arr, img_size)
        gray_arr = convert_to_grayscale(arr_res) / 255

        test_dict[f"car{i + 1}"] = gray_arr.flatten()
    
    for i, jpeg in enumerate(os.scandir(bike_path)):

        arr = convert_image_to_array(jpeg)
        arr_res = resize_image(arr, img_size)
        gray_arr = convert_to_grayscale(arr_res) / 255

        test_dict[f"bike{i + 1}"] = gray_arr.flatten()
    
    for i, jpeg in enumerate(os.scandir(tank_path)):

        arr = convert_image_to_array(jpeg)
        arr_res = resize_image(arr, img_size)
        gray_arr = convert_to_grayscale(arr_res) / 255

        test_dict[f"tank{i + 1}"] = gray_arr.flatten()
    
    print(test_dict)
    return test_dict



if __name__ == "__main__":

    cars_path = "/home/olivererdmann/Documents/code/ml_learn/vehicle/data/Car"
    bike_path = "/home/olivererdmann/Documents/code/ml_learn/vehicle/data/Bike"
    tank_path = "/home/olivererdmann/Documents/code/ml_learn/vehicle/data/tanks"

    img_size = 64 # standardize/normalize the image sizes (ex. 28 will result in 28 x 28 image size)
    num_samples = 568 # Any number but must be less than 568

    X, Y = load_images_to_array(cars_path, bike_path, tank_path, num_samples, img_size)

    print(X.shape)
    print(Y.shape)
    np.save("X_vehicles.npy", X)
    np.save("Y_vehicles.npy", Y)

    # test data
    test_path = "/home/olivererdmann/Documents/code/ml_learn/vehicle/data/test"

    test_dict = test_images_to_dict(test_path, img_size)
    with open('test_dict.pkl', 'wb') as f:
        pickle.dump(test_dict, f)