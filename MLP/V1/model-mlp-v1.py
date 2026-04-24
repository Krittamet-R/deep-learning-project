import torch as torch
import torch.nn as nn
import json

source = "GOOGL_history.json"
save = "googleModelWK.pth"



class myModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hl1 = nn.Linear(5,10)
        self.hl2 = nn.Linear(10,10)
        self.hl3 = nn.Linear(10,10)
        self.hl4 = nn.Linear(10,5)
        self.hl5 = nn.Linear(5,3)
        self.last = nn.Linear(3,1)

    def forward(self, x):
        x = torch.relu(self.hl1(x))
        x = torch.relu(self.hl2(x))
        x = torch.relu(self.hl3(x))
        x = torch.relu(self.hl4(x))
        x = torch.relu(self.hl5(x))
        x = self.last(x)
        return x

def loadInput():
    with open(source, "r") as f:
        data = json.load(f)

    values = []
    for row in data:
        line = [
            row["Close"],
            row["High"],
            row["Low"],
            row["Open"],
            row["Volume"]
        ]
        values.append(line)

    raw_tensor = torch.tensor(values, dtype=torch.float32)
    v_min = raw_tensor.min(dim=0)[0]
    v_max = raw_tensor.max(dim=0)[0]
    normalized_values = (raw_tensor - v_min) / (v_max - v_min)

    return normalized_values, v_min, v_max

def train():
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print(f"using {device}")
    model = myModel().to(device)

    criterion = nn.MSELoss()
    learningRate = 0.0001
    backPropagation = torch.optim.Adam(model.parameters(), learningRate)
    epoch = 180

    normalized_data, v_min, v_max = loadInput()
    inputData = normalized_data

    for i in range(epoch+1):
        totalLoss = 0
        batchSize = 32
        for j in range(0, len(inputData)-1, batchSize):

            end = min(j + batchSize, len(inputData)-1)

            batchIn = inputData[j:end].to(device)
            batchOut = inputData[j+1 : end+1, 0].unsqueeze(1).to(device)

            output = model(batchIn)
            loss = criterion(output, batchOut)

            backPropagation.zero_grad()
            loss.backward()
            backPropagation.step()
            totalLoss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {totalLoss}")
            checkpoint = {
                'vmin': v_min,
                'vmax': v_max,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': backPropagation.state_dict(),
                'loss': totalLoss,
            }
            torch.save(checkpoint, "checkpoint.pth")

    final_package = {
        'model_state': model.state_dict(),
        'vmin': v_min,
        'vmax': v_max
    }

    torch.save(final_package, save)
    print("Model saved")    

def using():
    checkpoint = torch.load("model/googleModelWK.pth", weights_only=False)
    vMin = checkpoint['vmin']
    vMax = checkpoint['vmax']

    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print(f"using {device}")

    model = myModel().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    close = 341.68
    high = 342.32
    low = 315.47
    open = 317.14
    volume = 117630000

    raw_num = torch.tensor([close, high, low, open, volume], dtype=torch.float32).to(device)
    
    normalized_num = (raw_num - vMin.to(device)) / (vMax.to(device) - vMin.to(device))
    
    normalized_num = normalized_num.unsqueeze(0)

    with torch.no_grad():
        prediction = model(normalized_num)

    close_min = vMin[0].item()
    close_max = vMax[0].item()
    final_price = (prediction * (close_max - close_min)) + close_min
    print(f"predicted price: {final_price.item()}")

if __name__ == "__main__":
    train()
    #using()
