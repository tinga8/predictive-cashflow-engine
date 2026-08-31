import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

# Set up the web page layout
pd_stream.set_page_config(page_title="AI Cashflow Forecaster", layout="wide")
pd_stream.title("📈 AI-Powered Revenue & Cashflow Forecasting Engine")
pd_stream.caption("Built with Python & Meta's Prophet Library | Permanently Hosted for Portfolio Showcase")

# 1. Generate Interactive Inputs in Sidebar
pd_stream.sidebar.header("Model Parameters")
forecast_days = pd_stream.sidebar.slider("Forecast Horizon (Days)", 30, 180, 90)
noise_level = pd_stream.sidebar.slider("Market Volatility (Noise)", 50, 500, 200)

# 2. Synthesize Historical Data
@pd_stream.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
    n_days = len(dates)
    trend = np.linspace(1500, 3500, n_days)
    weekly_season = np.sin(dates.dayofweek * (2 * np.pi / 7)) * 300
    monthly_season = np.sin(dates.day * (2 * np.pi / 30)) * 500
    noise = np.random.normal(0, noise_level, n_days)
    
    revenue = np.clip(trend + weekly_season + monthly_season + noise, 500, None)
    expenses = trend * 0.65 + np.random.normal(0, 150, n_days)
    cash_flow = revenue - expenses
    return pd.DataFrame({'ds': dates, 'y': revenue, 'cash_flow': cash_flow})

df = load_data()

# 3. Train Model and Run Forecast
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df[['ds', 'y']])

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# 4. Create Interactive Charts
fig = go.Figure()

# Historical Data
fig.add_trace(go.Scatter(
    x=df['ds'], 
    y=df['y'], 
    name="Historical Revenue", 
    mode='markers', 
    marker=dict(color='black', size=2)
))

# Forecasted Line (Fixed to yhat)
fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat'], 
    name="Forecasted Revenue", 
    line=dict(color='#0066cc', width=2)
))

# Uncertainty Upper Bound (Fixed to yhat_upper)
fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_upper'], 
    name="Upper Bound (Best Case)", 
    line=dict(dash='dash', color='rgba(0,102,204,0.3)')
))

# Uncertainty Lower Bound (Fixed to yhat_lower)
fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_lower'], 
    name="Lower Bound (Worst Case)", 
    line=dict(dash='dash', color='rgba(0,102,204,0.3)'), 
    fill='tonexty'
))

fig.update_layout(
    title="Interactive Revenue Forecast Model", 
    xaxis_title="Date", 
    yaxis_title="Amount ($)", 
    template="plotly_white"
)

# Display on web dashboard
pd_stream.plotly_chart(fig, use_container_width=True)

# 5. Show Metrics Summary (Fixed to yhat variables)
pd_stream.subheader("📊 Key Predictive Inferences")
col1, col2 = pd_stream.columns(2)
with col1:
    pd_stream.metric(
        label="Predicted Final Day Revenue", 
        value=f"${forecast['yhat'].iloc[-1]:,.2f}"
    )
with col2:
    uncertainty_range = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2
    pd_stream.metric(
        label="Model Uncertainty Range", 
        value=f"± ${uncertainty_range:,.2f}"
    )



