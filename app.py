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

# New Section: File Uploader in Sidebar
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Upload Your Data")
uploaded_file = pd_stream.sidebar.file_uploader(
    "Upload your historical data (Excel or CSV)", 
    type=["xlsx", "csv"]
)

# 2. Load Data (Handles Uploaded File or falls back to Simulated Data)
def load_data():
    if uploaded_file is not None:
        try:
            # Check if it's CSV or Excel
            if uploaded_file.name.endswith('.csv'):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)
            
            # Map user columns to Prophet format
            # Expecting columns named 'Date' and 'Value'
            if 'Date' in user_df.columns and 'Value' in user_df.columns:
                final_df = user_df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
                final_df['ds'] = pd.to_datetime(final_df['ds'])
                pd_stream.sidebar.success("🎉 Data uploaded successfully!")
                return final_df, False
            else:
                pd_stream.sidebar.error("❌ Error: File must contain 'Date' and 'Value' columns.")
        except Exception as e:
            pd_stream.sidebar.error(f"❌ Error reading file: {e}")
            
    # Fallback to simulated data if no file is uploaded or if upload fails
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
    n_days = len(dates)
    trend = np.linspace(1500, 3500, n_days)
    weekly_season = np.sin(dates.dayofweek * (2 * np.pi / 7)) * 300
    monthly_season = np.sin(dates.day * (2 * np.pi / 30)) * 500
    noise = np.random.normal(0, 200, n_days)
    revenue = np.clip(trend + weekly_season + monthly_season + noise, 500, None)
    
    fallback_df = pd.DataFrame({'ds': dates, 'y': revenue})
    return fallback_df, True

df, using_simulated = load_data()

if using_simulated:
    pd_stream.info("💡 Currently showing **simulated demonstration data**. Upload your own Excel file in the sidebar to see your own forecast!")

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
    name="Historical Data", 
    mode='markers', 
    marker=dict(color='black', size=3)
))

# Forecasted Line
fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat'], 
    name="AI Prediction", 
    line=dict(color='#0066cc', width=2)
))

# Uncertainty Upper Bound
fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_upper'], 
    name="Upper Bound (Best Case)", 
    line=dict(dash='dash', color='rgba(0,102,204,0.3)')
))

# Uncertainty Lower Bound
fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_lower'], 
    name="Lower Bound (Worst Case)", 
    line=dict(dash='dash', color='rgba(0,102,204,0.3)'), 
    fill='tonexty'
))

fig.update_layout(
    title="Custom Predictive Forecast Model", 
    xaxis_title="Date", 
    yaxis_title="Value ($)", 
    template="plotly_white"
)

# Display on web dashboard
pd_stream.plotly_chart(fig, use_container_width=True)

# 5. Show Metrics Summary
pd_stream.subheader("📊 Key Predictive Inferences")
col1, col2 = pd_stream.columns(2)
with col1:
    pd_stream.metric(
        label="Predicted Final Day Value", 
        value=f"${forecast['yhat'].iloc[-1]:,.2f}"
    )
with col2:
    uncertainty_range = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2
    pd_stream.metric(
        label="Model Uncertainty Range", 
        value=f"± ${uncertainty_range:,.2f}"
    )



