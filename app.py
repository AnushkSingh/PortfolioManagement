from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import yfinance as yf
import joblib

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("models", exist_ok=True)  # Ensure your trained models are here

# Define colors for plotting
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan', 'magenta']

def predict_portfolio(filepath):
    portfolio = pd.read_csv(filepath)
    prediction_summary = []

    for idx, row in portfolio.iterrows():
        symbol = row["Stock"]
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
        if change_pct > 0.5:
            recommendation = "BUY 🟢"
        elif change_pct < -0.5:
            recommendation = "SELL 🔴"
        else:
            recommendation = "HOLD 🟡"

        prediction_summary.append({
            "Stock": symbol,
            "Last_Close": round(last_close, 2),
            "Predicted_Next_Close": round(predicted_next, 2),
            "Change_Pct": round(change_pct, 2),
            "Recommendation": recommendation
        })

        # Plot chart
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

        chart_path = f"static/charts/{symbol}_trend.png"
        plt.savefig(chart_path)
 
        plt.savefig(chart_path)
        plt.close()
        print(f"✅ Saved chart for {symbol}: {chart_path}")

    summary_df = pd.DataFrame(prediction_summary)
    summary_csv = os.path.join("uploads", "portfolio_prediction.csv")
    summary_df.to_csv(summary_csv, index=False)
    return summary_df, [f"charts/{row['Stock']}_trend.png" for _, row in portfolio.iterrows()]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        summary_df, chart_files = predict_portfolio(filepath)
        return render_template('results.html', tables=[summary_df.to_html(classes='data', index=False)], chart_files=chart_files)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
