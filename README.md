# NYSE Stock Closing Price Prediction – Deep Regression with PyTorch  
**R² = 0.999940 | RMSE = $0.634**  

Haimeur Assia 

---

### 1. Project Objective
Implement a complete deep regression solution using PyTorch to predict the **closing price (`close`)** of NYSE-listed stocks from same-day intraday features:  
→ `open`, `high`, `low`, `volume`

---

### 2. Dataset
- **Source**: Kaggle – NYSE Stocks Historical Prices (~851,264 rows)
- **Selected Features**: 4 continuous numerical variables
- **Target**: `close` (price in USD)
- No missing values after cleaning

---

### 3. Exploratory Data Analysis (EDA)
- Extremely high correlation (> 0.999) between `close` and `open`, `high`, `low`
- Closing price almost always lies between the daily high and low
- Multimodal but continuous distribution of closing prices
![time series plot](images/loss_curves.png)
(Screenshots to insert: time series plot, correlation heatmap, close price histogram)

---

### 4. Preprocessing
- Standardization of features and target using `StandardScaler` (fit only on training set)
- Train / Validation / Test split: 70% / 15% / 15% (random state = 42)
- Custom PyTorch `Dataset` and `DataLoader` implementation

---

### 5. Network Architecture & Hyperparameter Search
A fully parameterizable **Multi-Layer Perceptron (MLP)** was implemented and evaluated across **32 configurations** using GridSearch:

| Hyperparameter         | Tested Values                  |
|------------------------|--------------------------------|
| Hidden layers          | [64→32] and [128→64→32]        |
| Learning rate          | 0.001 and 0.0005               |
| Dropout                | 0.0 and 0.2                    |
| Batch Normalization    | True / False                   |
| Weight Decay (L2)      | 0.0 and 1e-4                   |

**Best configuration automatically selected**:

```text
RegressionMLP(
  (net): Sequential(
    (0): Linear(4  → 64)  + ReLU
    (1): Linear(64 → 32)  + ReLU
    (2): Linear(32 → 1)   → raw output (regression)
  )
)
Optimal Hyperparameters:
- hidden layers = [64, 32]
- lr = 0.0005
- dropout = 0.0
- batchnorm = False
- weight_decay = 0.0
- optimizer = Adam
