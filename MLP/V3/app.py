import torch as torch
import torch.nn as nn
import numpy as np
import json
import os
import sys

try:
    from .. import custom_tool as tool
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import custom_tool as tool



training_set = "dataset/GOOGL_training_set.json"
validation_set = "dataset/GOOGL_validation_set.json"
test_set = "dataset/GOOGL_test_set.json"

class myModel(nn.Module):

    def __init__(self, window_size, feature_num):
        super().__init__()

        self.input_dimension = window_size * feature_num

        self.network = nn.Sequential(
            nn.Linear(self.input_dimension, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16,3)
        )
    
    def forward(self, x):

        x = x.view(x.size(0), -1)
        return self.network(x)

def installGPU():
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    print(f"install {device}")
    return device

def load_dataset(set):
    with open(set, "r") as data:
        dataset = json.load(data)

    dataset = tool.initialize_dataset(dataset)
    dataset = tool.add_RSI(dataset)
    dataset = tool.add_MA(dataset)
    dataset = tool.add_Log_Return_Close(dataset)
    dataset = tool.add_Log_Return_Volume(dataset)
    dataset = tool.add_volatility(dataset)

    dataset = tool.add_targets(dataset)

    dataset = tool.drop_high(dataset)
    dataset = tool.drop_low(dataset)
    dataset = tool.drop_open(dataset)
    dataset = tool.drop_volume(dataset)

    return dataset

def train():
    device = installGPU()
    window_size = 5
    feature = 5
    model = myModel(window_size, feature).to(device)

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.001
    back_propagation = torch.optim.Adam(model.parameters(), learning_rate)
    epoch = 50

    dataset = load_dataset(training_set)
    select = ['close_log_return', 'volume_log_return', 'RSI', 'MA', 'volatility']

    raw_x = dataset[select].values
    raw_y = dataset['Target'].values

    x_windows = []
    y_labels = []

    for i in range(len(raw_x) - window_size):
        # Slice 5 rows and flatten them into one row of 25 numbers
        window_data = raw_x[i : i + window_size].flatten() 
        x_windows.append(window_data)
        
        # The target is the label for the LAST day of that window
        y_labels.append(raw_y[i + window_size])

    features = torch.tensor(x_windows, dtype = torch.float32).to(device)
    targets = torch.tensor(y_labels, dtype = torch.float32).to(device)

    model.train()
    for i in range(epoch):
        back_propagation.zero_grad()
        outputs = model(features) # Output shape [N, 3]
        loss = loss_function(outputs, targets)
        loss.backward()
        back_propagation.step()
        
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()