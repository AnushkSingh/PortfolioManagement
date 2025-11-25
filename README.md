

⸻

📈 Portfolio Management Prediction using Polynomial Regression

This project predicts short-term stock price movements to help with portfolio management decisions. The system downloads the last 2 years of stock data using yfinance, applies Polynomial Regression (degree 3) with Ridge regularization, and outputs predicted future prices.

⸻

🚀 Features
	•	Fetches latest 2-years dataset from Yahoo Finance
	•	Cleans & prepares market data
	•	Builds Polynomial Regression (deg=3) model
	•	Uses Ridge Regularization to avoid overfitting
	•	Saves trained models as .pkl files
	•	Predicts next-day stock prices
	•	Lightweight & easy to run

⸻

🧠 Why Polynomial Regression?

Stock prices rarely follow straight lines → they are non-linear.

Model	Why Not Used
Linear Regression	Too simple → fails on curves
ARIMA / LSTM	Need very large datasets & heavy compute
SVM / Random Forest	Hard to interpret & tune
Polynomial Regression (Chosen)	Captures non-linear patterns with small datasets

✔ Best fit for short-term trend-based forecasting
✔ Works even with 30 days data

⸻

📊 Dataset Used

Field	Description
Source	Yahoo Finance via yfinance
Range	Last 2 yeaars from current date
Fields Used	Open, High, Low, Close, Adj Close, Volume
Target Variable	Close (future prediction)


⸻

🛠 Tech Stack
	•	Python
	•	yfinance
	•	Pandas
	•	NumPy
	•	Scikit-Learn
	•	Matplotlib

⸻

📦 Installation

1️⃣ Clone Repository

git clone https://github.com/your-username/portfolio-management.git
cd portfolio-management

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run Prediction Script

python stock_predict.py


⸻

📈 Output

For a ticker like AAPL, the script will:
	•	Show actual vs predicted curve
	•	Train model using polynomial features (degree 3)
	•	Save model as aapl.pkl
	•	Predict next-day price

⸻

🔮 Future Improvements
	•	Add LSTM Neural Networks
	•	Multi-stock portfolio optimization
	•	Risk metrics (Sharpe, Beta)
	•	Deployment using Streamlit / Flask

⸻

👨‍💻 Author

Anushk Singh

---
Roll No. - 23115901
CSE 5th Semester, NIT Raipur
