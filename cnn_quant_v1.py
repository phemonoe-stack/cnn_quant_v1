"""
quant_cnn_v1.py
================
A student-friendly 1D CNN for quantitative finance signal prediction.
Bridges directly from your MNIST CNN / KNN prototype work.

ANALOGY TO YOUR MNIST CODE:
  - MNIST:  28x28 pixel image     → CNN → digit class (0-9)
  - Quant:  30-day price window   → CNN → signal class (Buy / Sell / Hold)

FREE DATA SOURCES (no API key needed):
  - yfinance  : Yahoo Finance historical OHLCV
  - pandas-ta : Technical indicators (RSI, MACD, etc.)
  - tensorflow/keras : CNN model (same as your MNIST stack)

INSTALL:
  pip install yfinance pandas-ta tensorflow scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────
# 1. FETCH FREE MARKET DATA (like mnist.load_data)
# ─────────────────────────────────────────────
def load_market_data(ticker="SPY", period="10y", interval="1d"):
    """
    Downloads OHLCV data from Yahoo Finance.
    Analogous to: (images, labels), (...) = mnist.load_data()
    """
    print(f"Fetching {ticker} data from Yahoo Finance...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    print(f"  Loaded {len(df)} rows of {ticker} daily data.")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING (like flattening MNIST pixels)
# ─────────────────────────────────────────────
def add_features(df):
    """
    Adds technical indicators as feature channels.
    In MNIST, each pixel is a feature.
    Here, each indicator value per day is a feature.
    """
    df = df.copy()

    # Price-derived features
    df["Return"]     = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Momentum
    df["RSI"]        = ta.rsi(df["Close"], length=14)
    macd             = ta.macd(df["Close"])
    df["MACD"]       = macd["MACD_12_26_9"]
    df["MACD_Signal"]= macd["MACDs_12_26_9"]

    # Trend
    df["EMA_20"]     = ta.ema(df["Close"], length=20)
    df["EMA_50"]     = ta.ema(df["Close"], length=50)
    df["EMA_ratio"]  = df["EMA_20"] / df["EMA_50"]

    # Volatility
    bb               = ta.bbands(df["Close"], length=20)
    df["BB_upper"]   = bb["BBU_20_2.0"]
    df["BB_lower"]   = bb["BBL_20_2.0"]
    df["BB_width"]   = (df["BB_upper"] - df["BB_lower"]) / df["Close"]
    df["ATR"]        = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Volume
    df["Volume_MA"]  = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_MA"]

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. LABEL GENERATION (the "digit" in MNIST)
# ─────────────────────────────────────────────
def generate_labels(df, forward_days=5, buy_thresh=0.01, sell_thresh=-0.01):
    """
    Generates Buy / Hold / Sell labels from future returns.

    Like MNIST digits (0-9), we assign a class to each window:
      2 = Buy  (forward return > +1%)
      1 = Hold (between thresholds)
      0 = Sell (forward return < -1%)
    """
    future_return = df["Close"].shift(-forward_days) / df["Close"] - 1
    labels = np.where(future_return > buy_thresh, 2,
             np.where(future_return < sell_thresh, 0, 1))
    return labels


# ─────────────────────────────────────────────
# 4. SLIDING WINDOW (the "image" in MNIST)
# ─────────────────────────────────────────────
def create_windows(features, labels, window_size=30):
    """
    Converts time series into 3D windows:
      Shape: (num_samples, window_size, num_features)

    This is the MNIST image equivalent:
      MNIST:  (60000, 28, 28)   → 28x28 spatial grid
      Quant:  (N, 30, 13)       → 30-day × 13-feature grid
    """
    X, y = [], []
    for i in range(window_size, len(features) - 1):
        X.append(features[i - window_size:i])
        y.append(labels[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 5. BUILD THE 1D CNN (same Keras as your MNIST model)
# ─────────────────────────────────────────────
def build_cnn(input_shape, num_classes=3):
    """
    1D CNN for time-series classification.

    MNIST analogy:
      Conv2D scans spatial patches of pixels.
      Conv1D scans temporal patches of indicator values.
    """
    model = Sequential([
        # First conv block — captures short-term patterns (3-5 day moves)
        Conv1D(filters=64, kernel_size=3, activation="relu",
               padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        # Second conv block — captures medium-term patterns (week-scale)
        Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        # Third conv block — high-level abstraction
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.25),

        # Classification head (same as MNIST Dense layers)
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")   # Buy / Hold / Sell
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────
# 6. EVALUATE + PLOT (mirrors your error bar plots)
# ─────────────────────────────────────────────
def plot_results(history, y_true, y_pred, class_names=["Sell", "Hold", "Buy"]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Quant CNN Results", fontsize=14, fontweight="bold")

    # Training curves
    axes[0].plot(history.history["accuracy"],     label="Train Acc", color="MediumSlateBlue")
    axes[0].plot(history.history["val_accuracy"],  label="Val Acc",   color="Tomato")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss", color="MediumSlateBlue")
    axes[1].plot(history.history["val_loss"],  label="Val Loss",   color="Tomato")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # Confusion matrix (like your MNIST digit confusion matrix)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="PurpleRed_r",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[2])
    axes[2].set_title("Confusion Matrix\n(Sell / Hold / Buy)")
    axes[2].set_ylabel("True Label")
    axes[2].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig("quant_cnn_results.png", dpi=150)
    plt.show()
    print("Plot saved: quant_cnn_results.png")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # --- Load data (free, no API key) ---
    df_raw = load_market_data(ticker="SPY", period="10y")

    # --- Feature engineering ---
    df = add_features(df_raw)

    # --- Labels ---
    labels_raw = generate_labels(df, forward_days=5,
                                  buy_thresh=0.01, sell_thresh=-0.01)

    # --- Feature columns ---
    FEATURE_COLS = [
        "Return", "Log_Return", "RSI", "MACD", "MACD_Signal",
        "EMA_ratio", "BB_width", "ATR", "Volume_ratio"
    ]
    feature_matrix = df[FEATURE_COLS].values

    # --- Normalize (important! like dividing MNIST pixels by 255) ---
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    # --- Sliding windows ---
    WINDOW = 30
    X, y = create_windows(feature_matrix, labels_raw, window_size=WINDOW)

    print(f"\nDataset shape:  X={X.shape}  y={y.shape}")
    print(f"Class distribution: Sell={np.sum(y==0)}  Hold={np.sum(y==1)}  Buy={np.sum(y==2)}")

    # --- Train / test split (temporal — never shuffle time series!) ---
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # One-hot encode labels (same as MNIST to_categorical)
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat  = to_categorical(y_test,  num_classes=3)

    # --- Build model ---
    model = build_cnn(input_shape=(WINDOW, X.shape[2]))
    model.summary()

    # --- Train ---
    early_stop = EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True)
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Sell", "Hold", "Buy"]))

    # --- Plot results ---
    try:
        import seaborn as sns
        plot_results(history, y_test, y_pred)
    except ImportError:
        print("Install seaborn for plots: pip install seaborn")

    # --- Save model ---
    model.save("quant_cnn_model.keras")
    print("\nModel saved: quant_cnn_model.keras")

    # ── NOTE FOR STUDENTS ──────────────────────────────────────────────────
    # This model predicts directional signals, NOT dollar amounts.
    # Real quant systems also need:
    #   - Transaction cost modeling  (slippage, commissions)
    #   - Position sizing            (Kelly criterion, volatility targeting)
    #   - Walk-forward validation    (re-train on rolling windows)
    #   - Regime detection           (bull/bear/sideways market filters)
    # ────────────────────────────────────────────────────────────────────────
