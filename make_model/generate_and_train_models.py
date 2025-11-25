import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create folders if they don’t exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# List of stocks to train models for
symbols = ["AAPL", "GOOGL", "TSLA", "MSFT"]

print("📥 Downloading data using yfinance...")

for symbol in symbols:
    print(f"\n🔹 Processing {symbol}...")
    
    # 1️⃣ Download last 2 years of daily data
    data = yf.download(symbol, period="2y", interval="1d")
    if data.empty:
        print(f"⚠️ No data found for {symbol}, skipping...")
        continue
    
    # 2️⃣ Save raw dataset
    csv_path = f"datasets/{symbol}_data.csv"
    data.to_csv(csv_path)
    print(f"✅ Saved dataset: {csv_path}")
    
    # 3️⃣ Prepare training data (predict next day's Close from current Close)
    data["Prev_Close"] = data["Close"].shift(1)
    data.dropna(inplace=True)
    
    X = data[["Prev_Close"]].values
    y = data["Close"].values
    
    # 4️⃣ Train simple Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # 5️⃣ Evaluate model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"📊 Model performance for {symbol}:")
    print(f"   MSE  = {mse:.4f}")
    print(f"   R²   = {r2:.4f}")
    
    # 6️⃣ Save model
    model_path = f"models/{symbol}_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Saved model: {model_path}")

print("\n✅ All models trained and saved successfully!")
