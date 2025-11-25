import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

def predict_portfolio():
    # Create folder to save charts for Flask static files
    os.makedirs("static/charts", exist_ok=True)

    # Load portfolio
    portfolio = pd.read_csv("portfolio.csv")

    # Define colors for plotting
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    # Prepare a list to store predictions
    prediction_summary = []

    for idx, row in portfolio.iterrows():
        symbol = row["Stock"]
        
        # Load trained model
        model_path = f"models/{symbol}_model.pkl"
        if not os.path.exists(model_path):
            print(f"⚠️ Model for {symbol} not found, skipping...")
            continue
        model = joblib.load(model_path)
        
        # Fetch latest 60 days of data
        data = yf.download(symbol, period="60d", interval="1d")
        if data.empty:
            print(f"⚠️ No data for {symbol}, skipping...")
            continue

        # Prepare features
        data["Prev_Close"] = data["Close"].shift(1)
        data.dropna(inplace=True)
        last_close = float(data["Close"].iloc[-1])

        # Predict next close
        predicted_next = float(model.predict([[last_close]])[0])

        # Calculate predicted % change
        change_pct = ((predicted_next - last_close) / last_close) * 100

        # Recommendation logic
        if change_pct > 1.0:
            recommendation = "BUY 🟢"
        elif change_pct < -1.0:
            recommendation = "SELL 🔴"
        else:
            recommendation = "HOLD 🟡"

        # Append to summary
        prediction_summary.append({
            "Stock": symbol,
            "Last_Close": round(last_close, 2),
            "Predicted_Next_Close": round(predicted_next, 2),
            "Change_Pct": round(change_pct, 2),
            "Recommendation": recommendation
        })

        # Plot chart for the stock
        plt.figure(figsize=(10,5))
        color = colors[idx % len(colors)]
        plt.plot(data.index, data["Close"], label=f"{symbol} Actual", color=color, marker='o')
        plt.scatter(data.index[-1] + pd.Timedelta(days=1), predicted_next, color=color, marker='x', s=100, label=f"{symbol} Predicted")
        plt.title(f"{symbol} Price Trend")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save chart in static folder for Flask
        chart_path = f"static/charts/{symbol}_trend.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"✅ Saved chart for {symbol}: {chart_path}")

    # Save portfolio prediction summary CSV
    summary_df = pd.DataFrame(prediction_summary)
    summary_df.to_csv("portfolio_prediction.csv", index=False)
    print("\n📊 Portfolio prediction summary saved as 'portfolio_prediction.csv'.")

    # Return summary for Flask
    return summary_df, [f"charts/{row['Stock']}_trend.png" for _, row in portfolio.iterrows()]

