import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functions.helpers import normalize


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2,5), # Linear is the layer that defines the structure of the neurons in layer (input, output)
            nn.ReLU(), # This is the activation function applied to the layer above
            nn.Linear(5,5),
            nn.ReLU(),
            nn.Linear(5,1),
            # nn.Sigmoid() # We omit this due to rounding error - better to apply it in the loss function instead
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def predict(self, x):
        self.eval()
        logits = self(x) # this will auto pass forward
        probs = torch.sigmoid(logits)
        labels = (probs > 0.5).int()
        res = ""
        if labels.item() == 1:
            res = "good coffee"
        else:
            res = "bad coffee"
        return probs, res



loss_fn = nn.BCEWithLogitsLoss() # This applies our final sigmoid along with the loss calcultion

model = NeuralNetwork().to(device)

# Load in the data
X =  np.load("/home/olivererdmann/Documents/code/ml_learn/coffee/data/data_X.npy")
Y = np.load("/home/olivererdmann/Documents/code/ml_learn/coffee/data/data_Y.npy")

#define params
epochs = 6000
learning_rate = 5e-2

X_tensor = torch.from_numpy(X).float().to(device)  # convert to float32 and move to device
Y_tensor = torch.from_numpy(Y).float().to(device)
Y_tensor = Y_tensor.unsqueeze(1)

for i in range(epochs):

    logits = model(X_tensor)
    loss = loss_fn(logits, Y_tensor)
    print(loss.item())
    model.zero_grad() # zero the gradient so it doesn't explode when running backward pass
    loss.backward()

    #update parameters
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            param.grad.zero_()
    
# Infer model
print("type q to exit")
while True:
    temp = input("Enter a temperature: ")
    if temp == 'q': break
    time = input("Enter a time: ")
    if time == 'q': break

    print("time: ", time, " secs")
    print("temp: ", temp)

    try:
        temp = float(temp)
        time = float(time)
        # Normalize
        max_temp = 300
        min_temp = 160
        max_time = 21
        min_time = 7
        temp = normalize(temp, max_temp, min_temp)
        time = normalize(time, max_time, min_time)
    except ValueError:
        print("Temperature or Time is not a valid numeric type")
        continue
    
    x_test = torch.tensor([time, temp])
    prediction = model.predict(x_test)

    print(prediction)