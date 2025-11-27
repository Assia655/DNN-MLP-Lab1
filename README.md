# NYSE Stock Closing Price Prediction – Deep Regression with PyTorch  
**R² = 0.999940 | RMSE = $0.634**  

Haimeur Assia 

---

### Project Goal
Predict the **closing price (`close`)** of NYSE stocks using only same-day intraday features:  
`open`, `high`, `low`, `volume` → `close`

This is a **regression task** solved with a fully-connected neural network (MLP) in **PyTorch**.

> Warning: Academic exercise only – In real algorithmic trading, using `high`, `low`, `volume` of the same day to predict `close` constitutes **data leakage**. Here it's perfect for learning deep learning practice.

---

### Dataset
- **Source**: https://www.kaggle.com/datasets/dereksm/stock-data (`prices.csv`)
- **Size**: 851,264 rows
- **Features used**: `open`, `high`, `low`, `volume` → **4 inputs**
- **Target**: `close`**: continuous value in USD

---

### Final Selected Model Architecture (Best from GridSearch)

```text
RegressionMLP(
  Sequential(
    (0): Linear(in_features=4,   out_features=64,  bias=True)   → ReLU
    (1): Linear(in_features=64,  out_features=32,  bias=True)   → ReLU
    (2): Linear(in_features=32,  out_features=1,   bias=True)   → No activation (regression)
  )
)
