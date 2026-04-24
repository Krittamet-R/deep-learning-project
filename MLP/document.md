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
