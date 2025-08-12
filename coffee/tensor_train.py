from tensorflow.keras import layers
from functions.helpers import normalize
from functions.plots import tensor_training_curve
import tensorflow as tf
import numpy as np
import random
import os



'''
To replicate the neural network that was built with numpy from scratch, a tensorflow sequential model can be used to best represent it.
'''

keras = tf.keras


model = keras.Sequential(
    [
        layers.Dense(5, activation="relu", input_shape=(2,), name="hidden_layer_1"), # Input shape specifies how many features we are inputting (time and temperature)
        layers.Dense(5, activation="relu", name="hidden_layer_2"),
        layers.Dense(1, activation="sigmoid", name="layer_3")
    ]
)

# SET SEEDS
SEED = 587 
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# We need to give good estimations of weights so that there is a chance for convergence - the seeds preset should give us good initial guesses :)
custom_weights = [
    np.random.normal(size=(2, 5)),  # weights for layer 1
    np.zeros(5),                    # bias for layer 1
    np.random.normal(size=(5, 5)),  # weights for layer 2
    np.zeros(5),                    # bias for layer 2
    np.random.normal(size=(5, 1)),  # weights for layer 3
    np.zeros(1)                     # bias for layer 3
]

model.set_weights(custom_weights)

# The model is now built and we should now be able to call its summary and see how it is constructed
model.summary()
'''
From model summary we see 3 layers, with the hidden ones having an output shape of (None,5) and the final layer_3 having a shape of (None,1). 
# The None here specifies a dynamic batch size - which tensorflow will automatically handle once we give in our training data
Additionally layers 1 to 3 have parameters 15, 30, and 6 respectively, giving the model a total of 51 parameters that are all "trainable".
This is exactly what we expected.
'''

# Load in our training data from our export - tensorflow will automatically convert these numpy arrays into a tensor format so we don't need to worry about formatting :)
X = np.load("/home/olivererdmann/Documents/code/ml_learn/coffee/data/data_X.npy")
Y = np.load("/home/olivererdmann/Documents/code/ml_learn/coffee/data/data_Y.npy")
Y = Y.reshape(-1, 1)

# Specify training configurations (optimizer, loss, metrics)
model.compile(
    optimizer=keras.optimizers.Adam(), # This sets the learning rate as well - different optimizers to choose and experiment from - Adam is well suited
    loss=keras.losses.BinaryCrossentropy(), # Binary because we are checking 2 - if coffee is good or if coffee is bad
    metrics=[keras.metrics.BinaryAccuracy()]
)

# Set up training parameters
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, min_delta=0.001, restore_best_weights=False)

history = model.fit(
    X,
    Y,
    batch_size=32,
    epochs=650,
    validation_split=0.1, # 10% of the training data will be held back and used for validation - which will show how well our model performs on unseen data
    callbacks=[early_stop] # 
)

# Because no validation, we are not running any evaluations
history.history # Train the model and Print out the training history

loss = history.history['loss']
val_loss = history.history.get('val_loss')
epochs = range(1, len(loss) + 1)

tensor_training_curve(loss, val_loss, epochs)
# After training we can enter this while loop where we can call our predictions :)

print("Enter 'q' to exist this while loop")

while True:

    temp_input = input("Enter a temperature: ")
    if temp_input == 'q': break
    time_input = input("Enter a time: ")
    if time_input == 'q': break

    
    try:
        temp_input = float(temp_input)
        time_input = float(time_input)
        # Normalize
        max_temp = 300
        min_temp = 160
        max_time = 21
        min_time = 7
        temp_input = normalize(temp_input, max_temp, min_temp)
        time_input = normalize(time_input, max_time, min_time)
    except ValueError:
        print("Temperature or Time is not a valid numeric type")
        continue

    X_predict = np.array([[time_input, temp_input]])
    prediction = model.predict(X_predict)

    if prediction[0][0] >= 0.5:
        print(f"Good Coffee!\nProbability: {prediction[0][0]}")
    else:
        print(f"Bad Coffee :(\nProbability: {prediction[0][0]}")

    

print("Training End")