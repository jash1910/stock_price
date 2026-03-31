# 📈 Stock Price Prediction & Analysis Dashboard

## 🚀 Overview

This project is an end-to-end Machine Learning pipeline that analyzes and predicts stock prices using real-world-like messy data.

It covers the complete workflow:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training & Evaluation
* Interactive Dashboard Deployment

---

## 📊 Dataset Description

The dataset contains over **100,000+ rows** across multiple stock tickers (AAPL, TSLA, GOOGL, etc.) with real-world data issues such as:

* Missing values
* Duplicate rows
* Inconsistent formats
* Outliers and anomalies
* Incorrect data types

---

## 🧹 Data Cleaning Pipeline

* Converted inconsistent data types (strings → numeric)
* Handled missing values using forward fill
* Removed duplicate rows
* Fixed logical inconsistencies (High < Low, etc.)
* Standardized ticker formats
* Removed outliers using IQR method

---

## 📈 Exploratory Data Analysis

* Stock price trends over time
* Correlation analysis between features
* Volume vs price relationships
* Volatility patterns across stocks

---

## ⚙️ Feature Engineering

* Daily Returns
* 7-day Moving Average (MA7)
* Volatility (rolling standard deviation)

---

## 🤖 Models Used

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

---

## 🏆 Model Performance

| Model             | MAE      | RMSE      | R²         |
| ----------------- | -------- | --------- | ---------- |
| Linear Regression | ~305     | ~476      | ~0.48      |
| XGBoost           | ~10.6    | ~53.6     | ~0.993     |
| Random Forest     | **~7.9** | **~48.1** | **~0.995** |

👉 **Best Model: Random Forest Regressor**

---

## 📊 Streamlit Dashboard Features

* 📈 Stock price visualization
* 📊 KPI metrics
* 🔍 Interactive stock analysis
* 🤖 Real-time prediction interface
* 📂 CSV upload for batch predictions
* 📉 Model performance comparison

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn, XGBoost
* Streamlit
* Plotly

---

## 📁 Project Structure

```
stock-price-prediction/
│
├── data/
├── notebooks/
├── models/
├── app/
├── src/
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🌐 Deployment

The app is deployed using Streamlit Community Cloud.

---

## 💡 Key Insights

* Linear Regression underperformed due to non-linear stock behavior
* Tree-based models captured complex patterns effectively
* Random Forest achieved the best performance with lowest error
* Feature engineering significantly improved model accuracy

---

## 🔮 Future Improvements

* Time-series forecasting (LSTM / ARIMA)
* Real-time stock API integration
* Hyperparameter tuning
* Model optimization for production

---

## 👩‍💻 Author

**Jashvitha Lakshmi Omkaram**

---

⭐ If you like this project, feel free to star the repo!
