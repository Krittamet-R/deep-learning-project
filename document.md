# Stock Prediction Model

| Model | Status | last version |
| :---: | :---: | :---: |
| MLP | Ongoing | 2 |
| CNN | Not yet | 0 |
| RNN | Not yet | 0 |
| GRU | Not yet | 0 |
| LSTM | Not yet | 0 |
| Transformer | Not yet | 0 |

## Document

```md
Version: MLP-V2

### Goal
- try to add indicator to the input and calculate the validation loss

### Data
- Source: GOOGL stock history
- Time range: 2020-2026
- Split: training set(2020-2024), validation set(2025), test set(2026)

### Features and Target
- Input features: return Close, return Open, return Volume, RSI, VWAP
- Target: next return close (percentage)
- Input shape: `[N, 5]`

### Preprocessing
- Normalization: logarithm
- Outlier handling: Divide RSI by 100 to get range 0-1
- Other transforms: none

### Model
- Architecture: `5-10-10-10-5-1`
- Activation: LeakyReLU on hidden layers, add some dropout to prevent overfitting
- Loss: MSELoss
- Optimizer: Adam

### Training
- Batch strategy: Mini-batch size 16 and do some overlap each batch
- Epochs: 200 but 50 is enough likely overfitting
- Learning rate: `0.0001`
- Early stopping: None

### Evaluation
- Validation metric(s): use dataset of 2025
- Test metric(s): Not implemented
- Main result: Training loss and Validation loss printed every 10 epochs

### NOTE (Unique to this version)
- make code more structure and reuseable for future project
- only got Directional Accuracy: 44.74%
```

```md
Version: MLP-V1

### Goal
- Build a first baseline MLP in PyTorch to predict next-step Close price.

### Data
- Source: `GOOGL_history.json`
- Time range: Historical records of stock
- Split: No explicit train/validation/test split in V1

### Features and Target
- Input features: Close, High, Low, Open, Volume (CHLOV)
- Target: Next row Close value
- Input shape: `[N, 5]`

### Preprocessing
- Normalization: Min-max per column
- Outlier handling: Divide volume by 100000000
- Other transforms: Target is shifted by one timestep to create next-step prediction

### Model
- Architecture: `5-10-10-10-5-3-1`
- Activation: ReLU on hidden layers
- Loss: MSELoss
- Optimizer: Adam

### Training
- Batch strategy: Mini-batch size 32
- Epochs: 181 total iterations (`epoch = 180`, loop uses `range(epoch+1)`)
- Learning rate: `0.0001`
- Early stopping: None

### Evaluation
- Validation metric(s): Not implemented
- Test metric(s): Not implemented
- Main result: Training loss printed every 10 epochs

### NOTE (Unique to this version)
- Uses `torch.accelerator` when available, otherwise CPU
- Saves normalization stats (`vmin`, `vmax`) with model state for inverse transform during inference
- Uses next-step Close prediction from current CHLOV row (simple sequence shift, no recurrent memory)
```
