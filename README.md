# Lab-1
Haimeur Assia 

---
## 1- NYSE Stock Closing Price Prediction – Deep Regression with PyTorch  

### 1. lab Objective
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
![time series plot](images/diag1.png)
![correlation heatmap](images/diag2.png)
![close price histogram](images/diag3.png)
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
```

### 6. Quantitative Results

| Model                                | R² (test)    | RMSE (test)   | Best Validation Loss (scaled MSE) |
|--------------------------------------|--------------|---------------|------------------------------------|
| **Best Model (GridSearch)**          | **0.999940** | **$0.6342**   | **6.09 × 10⁻⁵**                    |
| Simple Model (no regularization)     | 0.999940     | $0.6340       | 6.16 × 10⁻⁵                        |

**Interpretation**:  
Near-perfect performance achieved within the first few epochs.  
An average error of **63 cents** on stocks ranging from $1 to $1,600 is outstanding.

---

### 7. Training Curves Analysis

![loss_r2_curves](images/diag4.png)

**Key Observations**:
- Loss (MSE) drops dramatically in less than 10 epochs and stabilizes near zero  
- R² exceeds **0.9999** from epoch 3 on both training and validation sets  
- Training and validation curves are **perfectly overlapped** → **no overfitting**  
- Early Stopping typically triggers between epochs 25–30 (patience = 12)

---

### 8. Prediction Visualization on Test Set

![real_vs_predicted](images/diag5.png)
![error_distribution](images/diag6.png)

**Key Observations:
- Points perfectly aligned along the **y = x** line  
- Maximum observed error ≈ **±$5**  
- **99.9%** of errors lie between **-$2** and **+$2**  
- Error distribution perfectly centered at 0 with an extremely sharp peak  
→ The model predicts closing price with **exceptional accuracy**

---

### 9. Effect of Regularization Techniques
![ Regularization](images/diag8.png)

| Configuration                             | R² (test) | RMSE (test) | Val Loss   | Regularization Used                        |
|-------------------------------------------|-----------|-------------|------------|--------------------------------------------|
| Best Model (GridSearch)                   | 0.999940  | $0.6342     | 6.09e-05   | None (dropout=0, batchnorm=False, wd=0)    |
| Simple Model (no regularization)          | 0.999940  | $0.6340     | 6.16e-05   | None                                       |

**Conclusion**:  
Performance is **strictly identical** with or without regularization.  
**Reason**: The task is extremely easy due to near-perfect correlation (> 0.9999) between inputs (`open`, `high`, `low`) and target (`close`). The network learns the relationship almost instantly → **zero overfitting risk**.  
Regularization techniques (dropout, batch normalization, weight decay) bring **no improvement** in this case.
