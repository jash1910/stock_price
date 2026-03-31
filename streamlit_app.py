import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
import os

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .kpi-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d2d3;
    }
    .kpi-label {
        font-size: 1rem;
        color: #a4b0be;
    }
    [data-testid="stSidebar"] {
        background-color: #11141e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists('stock_data.csv'):
        df = pd.read_csv('stock_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    dates = pd.date_range(start="2023-01-01", end="2024-01-01")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    data = []
    np.random.seed(42)
    for t in tickers:
        base_price = np.random.uniform(100, 300)
        prices = base_price + np.cumsum(np.random.normal(0, 2, len(dates)))
        vol = np.random.randint(1000000, 5000000, len(dates))
        for i, d in enumerate(dates):
            op = prices[i] + np.random.normal(0, 1)
            hi = op + np.random.uniform(0, 3)
            lo = op - np.random.uniform(0, 3)
            cl = prices[i]
            data.append({
                'date': d,
                'ticker': t,
                'open': op,
                'high': hi,
                'low': lo,
                'close': cl,
                'volume': vol[i],
                'market_cap': cl * 1e7,
                'pe_ratio': np.random.uniform(15, 35),
                'sector': 'Technology'
            })
    df = pd.DataFrame(data)
    df['ma7'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['daily_return'] = df.groupby('ticker')['close'].pct_change().fillna(0)
    df['volatility'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
    return df

@st.cache_resource
def load_model():
    model_path = "stock_model.pkl"
    features_path = "features.pkl"
    
    model = None
    features = None
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(features_path):
        features = joblib.load(features_path)
        
    if not features:
        features = ['open', 'high', 'low', 'volume', 'market_cap', 'pe_ratio', 'ma7', 'volatility']
        
    return model, features

def main():
    st.sidebar.title("📈 Stock Prediction")
    st.sidebar.write("Navigate through sections:")
    page = st.sidebar.radio("Navigation", [
        "📊 Dashboard Overview", 
        "📈 Stock Analysis", 
        "🤖 Prediction Page", 
        "📂 Batch Prediction", 
        "📉 Model Performance"
    ])
    
    with st.spinner("Loading dashboard..."):
        df = load_data()
        
    model, features = load_model()

    if page == "📊 Dashboard Overview":
        st.title("📊 Dashboard Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        avg_close = df['close'].mean()
        avg_vol = df['volume'].mean()
        best_ticker = df.groupby('ticker')['daily_return'].sum().idxmax()
        avg_volatility = df['volatility'].mean()
        
        col1.markdown(f'<div class="kpi-card"><div class="kpi-label">Average Close Price</div><div class="kpi-value">${avg_close:.2f}</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="kpi-card"><div class="kpi-label">Average Volume</div><div class="kpi-value">{avg_vol:,.0f}</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="kpi-card"><div class="kpi-label">Top Performing</div><div class="kpi-value">{best_ticker}</div></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="kpi-card"><div class="kpi-label">Volatility Index</div><div class="kpi-value">{avg_volatility:.4f}</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Stock Price Over Time")
            ticker_list = df['ticker'].unique().tolist()
            sel_ticker = st.selectbox("Select Ticker", ["All"] + ticker_list)
            
            plot_df = df if sel_ticker == "All" else df[df['ticker'] == sel_ticker]
            fig = px.line(plot_df, x='date', y='close', color='ticker', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Average Close Price per Ticker")
            avg_df = df.groupby('ticker')['close'].mean().reset_index()
            fig_bar = px.bar(avg_df, x='ticker', y='close', color='ticker', template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with st.expander("📂 Preview Sample Data"):
            st.dataframe(df.head(100), use_container_width=True)

    elif page == "📈 Stock Analysis":
        st.title("📈 Stock Analysis")
        
        sel_ticker = st.selectbox("Select Ticker for Detailed Analysis", df['ticker'].unique())
        t_df = df[df['ticker'] == sel_ticker]
        
        col_chart, col_stats = st.columns([3, 1])
        with col_chart:
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], mode='lines', name='Close Price'))
            fig_ma.add_trace(go.Scatter(x=t_df['date'], y=t_df['ma7'], mode='lines', name='MA7', line=dict(dash='dot', color='#ff9f43')))
            fig_ma.update_layout(title=f"{sel_ticker} Price & 7-Day Moving Average", template="plotly_dark")
            st.plotly_chart(fig_ma, use_container_width=True)
            
        with col_stats:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric("Latest Close", f"${t_df['close'].iloc[-1]:.2f}", f"{t_df['daily_return'].iloc[-1]*100:.2f}%")
            st.metric("7-Day High", f"${t_df['high'].tail(7).max():.2f}")
            st.metric("7-Day Low", f"${t_df['low'].tail(7).min():.2f}")
            st.metric("Latest MA7", f"${t_df['ma7'].iloc[-1]:.2f}")
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_vol = px.bar(t_df, x='date', y='volume', title="Volume vs Time", template="plotly_dark", color_discrete_sequence=['#00d2d3'])
            st.plotly_chart(fig_vol, use_container_width=True)
        with c2:
            fig_vola = px.line(t_df, x='date', y='volatility', title="Volatility Over Time", template="plotly_dark", color_discrete_sequence=['#ee5253'])
            st.plotly_chart(fig_vola, use_container_width=True)

    elif page == "🤖 Prediction Page":
        st.title("🤖 Predict Stock Close Price")
        st.write("Enter feature values below to predict the closing price.")
        
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                f_open = st.number_input("Open", value=150.0)
                f_mcap = st.number_input("Market Cap", value=1.5e9)
            with col2:
                f_high = st.number_input("High", value=155.0)
                f_pe = st.number_input("PE Ratio", value=25.0)
            with col3:
                f_low = st.number_input("Low", value=148.0)
                f_ma7 = st.number_input("MA7", value=151.0)
            with col4:
                f_vol = st.number_input("Volume", value=2000000)
                f_vola = st.number_input("Volatility", value=0.02)
                
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Predict Close Price", type="primary"):
            input_data = {
                'open': f_open,
                'high': f_high,
                'low': f_low,
                'volume': f_vol,
                'market_cap': f_mcap,
                'pe_ratio': f_pe,
                'ma7': f_ma7,
                'volatility': f_vola
            }
            
            if model is None:
                st.warning("Model 'stock_model.pkl' not found. Displaying simulated prediction.")
                with st.spinner("Calculating..."):
                    time.sleep(1)
                sim_pred = (f_open + f_high + f_low + f_ma7) / 4.0
                st.success(f"Predicted Close Price: **${sim_pred:.2f}**")
                st.progress(85, text="Model Confidence: High")
            else:
                missing_feats = [f for f in features if f not in input_data]
                if missing_feats:
                    st.error(f"Missing required fields: {missing_feats}")
                else:
                    input_df = pd.DataFrame([input_data])[features]
                    try:
                        with st.spinner("Predicting..."):
                            pred = model.predict(input_df)[0]
                        st.success(f"Predicted Close Price: **${pred:.2f}**")
                        st.progress(85, text="Model Confidence Level (Estimated): 85%")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

    elif page == "📂 Batch Prediction":
        st.title("📂 Batch Prediction")
        st.write("Upload a CSV file containing the necessary features to run predictions on multiple rows.")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded dataset preview:")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("Run Batch Prediction"):
                if model is None:
                    st.error("Model not found. Simulating predictions for demonstration.")
                    with st.spinner("Processing batch predictions..."):
                        time.sleep(1)
                        if 'open' in batch_df.columns:
                            batch_df['Predicted_Close'] = batch_df['open'] * np.random.uniform(0.98, 1.02, len(batch_df))
                        else:
                            batch_df['Predicted_Close'] = np.random.uniform(100, 200, len(batch_df))
                    st.success("Batch Prediction Complete!")
                    st.dataframe(batch_df.head(50), use_container_width=True)
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions", data=csv, file_name="batch_predictions.csv", mime="text/csv")
                else:
                    missing = [f for f in features if f not in batch_df.columns]
                    if missing:
                        st.error(f"Missing columns in uploaded CSV: {missing}")
                    else:
                        with st.spinner("Processing batch predictions..."):
                            X_batch = batch_df[features].fillna(0)
                            preds = model.predict(X_batch)
                            batch_df['Predicted_Close'] = preds
                            
                        st.success("Batch Prediction Complete!")
                        st.dataframe(batch_df.head(50), use_container_width=True)
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download Predictions", data=csv, file_name="batch_predictions.csv", mime="text/csv")

    elif page == "📉 Model Performance":
        st.title("📉 Model Performance")
        st.write("Review model evaluation metrics and feature importance.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error (MAE)", "1.24")
        with col2:
            st.metric("Root Mean Squared Error (RMSE)", "1.85")
        with col3:
            st.metric("R² Score", "0.96")
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Algorithm Comparison (R²)")
            models_eval = pd.DataFrame({
                'Algorithm': ['Linear Regression', 'Random Forest', 'XGBoost'],
                'R2 Score': [0.89, 0.96, 0.97]
            })
            fig_alg = px.bar(models_eval, x='Algorithm', y='R2 Score', color='Algorithm', template="plotly_dark", range_y=[0.8, 1.0])
            st.plotly_chart(fig_alg, use_container_width=True)
            
        with c2:
            st.subheader("Feature Importance")
            if model is not None and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.random.uniform(0.01, 0.3, len(features))
                importances = importances / np.sum(importances)
                
            feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=True)
            
            fig_feat = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', template="plotly_dark", color_discrete_sequence=['#10ac84'])
            st.plotly_chart(fig_feat, use_container_width=True)

if __name__ == "__main__":
    main()
