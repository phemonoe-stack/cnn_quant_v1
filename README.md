# CNN QUANT v0.01a
From MNIST → Quant Finance: The Core Analogy by a Space Monkey with revision by Claude

I can see your MNIST CNN project — it's actually a great foundation for quant! Let me map out the conceptual bridge and then help you build toward it.

## From MNIST → Quant Finance: The Core Analogy

Your MNIST code does this:
- Takes **image data** (28×28 pixels) → flattens → classifies digits

Quant CNN does this:
- Takes **price/volume data** (time × features) → slides a window → predicts signal (buy/sell/hold)

The "image" in quant is a **lookback window** of market data. A candlestick chart is literally an image — and researchers do feed raw chart images into CNNs — but more commonly you use **1D CNNs over time-series feature windows**.

---

## What Your Current Code Does Well (Keep These Ideas)

| Your MNIST Code | Quant Equivalent |
|---|---|
| KNN prototype selection | Feature/sample selection (reduce noise) |
| Train/test split | In-sample / out-of-sample backtest |
| Error rate bars | Sharpe ratio, win rate, drawdown metrics |
| CNN fast distance | Rolling window similarity search |

---

## A Free Student-Friendly Quant CNN — Let me build it for youHere's what the script does and how it maps to your MNIST work:

**Install everything free with one line:**
```bash
pip install yfinance pandas-ta tensorflow scikit-learn matplotlib seaborn
```

---

### The MNIST → Quant Bridge (annotated in the code)

| MNIST Concept | Quant Equivalent | In Your Code |
|---|---|---|
| `mnist.load_data()` | `yf.download("SPY")` | Free Yahoo Finance data |
| 28×28 pixel image | 30-day × 9-feature window | `create_windows()` |
| Digit label (0–9) | Signal label (Sell/Hold/Buy) | `generate_labels()` |
| Flatten pixels | `StandardScaler` normalize | Same importance |
| `Conv2D` spatial scan | `Conv1D` temporal scan | `build_cnn()` |
| Your error bar plots | Confusion matrix + curves | `plot_results()` |

Your **KNN prototype selection** idea is actually directly applicable here too — you could use it to reduce your training windows to the most "representative" market regimes, which is an active research area called *instance selection for time-series*.

### Key "Gotcha" for Finance vs. MNIST
**Never shuffle your train/test split.** MNIST images are i.i.d. (independent), but market days are not — future data leaking into training gives falsely great results (called *look-ahead bias*). The code does an 80/20 temporal split to prevent this.
--------------------------//---------------------
Inspired by: https://machinelearningmastery.com/using-cnn-for-financial-time-series-prediction/
