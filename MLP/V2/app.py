import torch as torch
import torch.nn as nn
import numpy as np
import json
import math
import os
import sys

try:
    from ..custom_tool import get_RSI, get_VWAP #outdate
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from MLP.custom_tool import get_RSI, get_VWAP #outdate

training_set = "dataset/GOOGL_training_set.json" #2020-2024
validation_set = "dataset/GOOGL_validation_set.json" #2025
test_set = "dataset/GOOGL_test_set.json" #2026
save_to = "model/google_model.pth"

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(5,10)
        
        self.layers = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(10,10),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(10,10),
            nn.LeakyReLU(),
            nn.Linear(10,5)
        )

        self.regression = nn.Linear(5,1)

    def forward(self, x):
        x = self.input(x)
        x = self.layers(x)
        x = self.regression(x)
        return x

def changePercentage(previous, current):
    return math.log(current/previous)

def load_dataset(source):
    with open(source, "r") as data:
        dataset = json.load(data)
    
    pythonArray = []
    previous_close = None
    previous_open = None
    previous_volume = None

    RSI_indicator = np.array(get_RSI(dataset, 14))# 14 day RSI
    VWAP_indicator = np.array(get_VWAP(dataset, 14))# 1 day RSI

    #Normalize the indicator
    RSI_normalize = RSI_indicator/100

    VWAP_indicator = np.array(VWAP_indicator).flatten()  # make it [N]
    VWAP_normalize = []
    previous_vwap = None

    for i in range(len(VWAP_indicator)):
        current = VWAP_indicator[i]

        if previous_vwap is not None and previous_vwap > 0:
            vwap_return = math.log(current / previous_vwap)
        else:
            vwap_return = math.log(1)

        VWAP_normalize.append(vwap_return)
        previous_vwap = current

    VWAP_normalize = np.array(VWAP_normalize).reshape(-1, 1)

    for row in dataset:

        if previous_close is not None:

            value = [
                changePercentage(previous_close, row["Close"]),
                changePercentage(previous_open, row["Open"]),
                changePercentage(previous_volume, row["Volume"])
            ]

            previous_close = row["Close"]
            previous_open = row["Open"]
            previous_volume = row["Volume"]

            pythonArray.append(value)

        else: #skip first and initialize data
            previous_close = row["Close"]
            previous_open = row["Open"]
            previous_volume = row["Volume"]
            pythonArray.append([0, 0, 0]) #make initial state

    pythonArray = np.array(pythonArray)
    array = np.hstack((pythonArray, RSI_normalize, VWAP_normalize*100)) # add RSI, VWAP [N, 1] to pythonArray

    tensor = torch.tensor(array, dtype=torch.float32)
    return tensor

def installGPU():
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print("installed GPU")
    return device

def test(result, device):
    print("==== back tracking test (2026) ====")
    dataset = load_dataset(test_set).to(device)
    result.eval()
    
    correct_direction = 0
    total_samples = len(dataset) - 1
    
    # track the actual log returns vs predicted
    predictions = []
    actuals = []

    with torch.no_grad():
        for j in range(total_samples):
            input_data = dataset[j].unsqueeze(0) 
            target = dataset[j+1, 0].item() 

            output = result(input_data).item() 
            
            # Check Directional Accuracy
            # If both are positive or both are negative, the direction is right
            if (output > 0 and target > 0) or (output < 0 and target < 0):
                correct_direction += 1
            
            predictions.append(output)
            actuals.append(target)

    accuracy = (correct_direction / total_samples) * 100
    print(f"Directional Accuracy: {accuracy:.2f}%")
    
    # Simple sanity check: print the first 5 predictions vs actuals
    print("First 5 Results (Predicted vs Actual Log Returns):")
    for i in range(5):
        print(f"P: {predictions[i]:.4f} | A: {actuals[i]:.4f}")


def validation_test(result, device):
    dataset = load_dataset(validation_set).to(device)
    result.eval()
    loss_function = nn.MSELoss()
    validation_loss = 0

    with torch.no_grad():
        for j in range(0, len(dataset) - 1):
            input_data = dataset[j].unsqueeze(0)  # [1, features]
            target = dataset[j+1, 0].unsqueeze(0).unsqueeze(1)  # [1,1]

            output = result(input_data)
            loss = loss_function(output, target)

            validation_loss += loss.item()
    
    return validation_loss

def train():
    print("start training")
    device = installGPU()
    model = MyModel().to(device)

    #config
    loss_function = nn.MSELoss()
    learning_rate = 0.0001
    back_propagation = torch.optim.Adam(model.parameters(), learning_rate)
    epoch = 100

    dataset = load_dataset(training_set).to(device)

    model.train()
    for i in range(epoch):
        training_loss = 0
        batch_size = 16
        next = 12
        for j in range(0, len(dataset)-1, next): #make it overlap
            end = min(j+batch_size, len(dataset)-1) #prevent out of bound
            batch_input = dataset[j:end]
            batch_target = dataset[j+1:end+1, 0].unsqueeze(1) #select close as target

            output = model(batch_input)
            difference = loss_function(output, batch_target)

            back_propagation.zero_grad()
            difference.backward()
            back_propagation.step()
            
            training_loss += difference.item()
        
        if(i % 10 == 0):
            validation_loss = validation_test(model, device)
            print(f"epoch: {i} training_loss: {training_loss} validation_loss: {validation_loss}")
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'back_propagation_state': back_propagation.state_dict(),
                'training_loss': training_loss,
                'validation_loss': validation_loss
            }

            torch.save(checkpoint, "checkpoint.pth")
    
    test(model, device)
    print("Done!")
    torch.save(model.state_dict(), save_to)

if __name__ == "__main__":
    train()