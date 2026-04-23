import cv2
import matplotlib.pyplot as plt

color = "green"  # CHANGE THIS
number = "1"  # CHANGE THIS
key = color + number
vidcap = cv2.VideoCapture(f"data/{color}/{key}.mp4")  # Make sure video exists
success, image = vidcap.read()
count = 0
path = f"data/{color}/frames{number}"  # Make sure path exists
# path = f"data/{color}/test"
crops = {
    "blue1": {"x": [18, 86], "y": [10, 115]},
    "blue2": {"x": [46, 105], "y": [13, 109]},
    "blue3": {"x": [35, 99], "y": [16, 108]},
    "blue4": {"x": [58, 114], "y": [25, 98]},
    "blue_test": {"x": [62, 123], "y": [16, 93]},
    "green1": {"x": [19, None], "y": [None, None]},
    "green2": {"x": [38, 112], "y": [13, 118]},
    "green3": {"x": [40, 114], "y": [10, 123]},
    "green4": {"x": [51, 117], "y": [22, 100]},
    "green_test": {"x": [58, 116], "y": [24, 100]},
    "yellow1": {"x": [34, None], "y": [None, None]},
    "yellow2": {"x": [34, 110], "y": [13, 115]},
    "yellow3": {"x":[58,116], "y": [22, 98]},
    "yellow_test": {"x": [55, 116], "y": [23, 100]}
}
while success:
    #downsample image to 128 x 128 px
    image = cv2.resize(image, dsize=(128, 128))
    # Crop image
    x, y = crops[key]["x"], crops[key]["y"]
    image = image[x[0]:x[1], y[0]:y[1]]

    cv2.imwrite(f"{path}/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print("Read a new frame: ", success)
    count += 1
