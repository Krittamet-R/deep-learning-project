import torch as torch
import torch.nn as nn
import numpy as np
import json
import math
import logging
from ..custom_indicator import get_RSI

training_set = "dataset/GOOGL_training_set.json" #2020-2024
validation_set = "dataset/GOOGL_validation_set.json" #2025
test_set = "dataset/GOOGL_test_set.json" #2026
save_to = "model/google_model"

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(5,10)
        
        self.layers = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(),
            nn.Linear(10,10),
            nn.LeakyReLU(),
            nn.Linear(10,10),
            nn.LeakyReLU(),
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

def load_input():
    with open(training_set, "r") as data:
        dataset = json.load(data)
    
    pythonArray = []
    previous_close = None
    previous_open = None
    previous_volume = None

    RSI_indicator = get_RSI(dataset, 14)# 14 day RSI

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

    array = np.hstack((pythonArray, RSI_indicator)) # add RSI [N, 1] to pythonArray

    tensor = torch.tensor(array, dtype=torch.float32)
    return tensor

def installGPU():
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        logging.info("installed GPU")
    return device

def train():
    logging.info("start training")
    device = installGPU()
    model = MyModel().to(device)

    #config
    loss = nn.MSELoss()
    learning_rate = 0.0001
    back_propagation = torch.optim.Adam(model.parameters(), learning_rate)
    epoch = 200

    input = load_input()

    for i in range(epoch):
        total_loss = 0
        batch_size = 16
        next = 12
        for j in range(0, len(input)-1, next): #make it overlap
            end = min(j+batch_size, len(input)-1) #prevent out of bound
            batch_input = input[j:end].to(device)
            batch_target = input[j+1:end+1, 0].unsqueeze(1).to(device) #select close as target

            output = model(batch_input)
            difference = loss(output, batch_target)

            back_propagation.zero_grad()
            difference.backward()
            back_propagation.step()
            
            total_loss = difference.item()
        
        if(i % 10 == 0):
            print(f"epoch: {epoch} loss: {total_loss}")
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'back_propagation_state': back_propagation.state_dict(),
                'loss': total_loss
            }
        
            torch.save(checkpoint, "checkpoint.pth")
    
    logging.info("train model finish")
    torch.save(model.state_dict(), save_to)

if __name__ == "__main__":
    train()