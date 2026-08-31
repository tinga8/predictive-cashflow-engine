import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import io

# Set up the web page layout
pd_stream.set_page_config(page_title="AI Cashflow Forecaster", layout="wide")
pd_stream.title("📈 AI-Powered Revenue & Cashflow Forecasting Engine")
pd_stream.caption("Built with Python & Meta's Prophet Library | Permanently Hosted for Portfolio Showcase")

# 1. Generate Interactive Inputs in Sidebar
pd_stream.sidebar.header("Model Parameters")
forecast_days = pd_stream.sidebar.slider("Forecast Horizon (Days)", 30, 180, 90)

# Section: File Management in Sidebar
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Custom Data Input")

# Create a sample template file in memory for users to download
@pd_stream.cache_data
def generate_sample_template():
    # Make a tiny sample dataframe matching required formatting
    sample_dates = pd.date_range(start="2026-01-01", periods=10, freq="D")
    sample_values = [1500, 1620, 1480, 1550, 1700, 1950, 1300, 1510, 1600, 1650]
    sample_df = pd.DataFrame({'Date': sample_dates, 'Value': sample_values})
    
    # Save to an excel bytes stream
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sample_df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

template_data = generate_sample_template()

# Add the template download button
pd_stream.sidebar.download_button(
    label="📥 Download Sample Excel Template",
    data=template_data,
    file_name="forecast_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# File Uploader
uploaded_file = pd_stream.sidebar.file_uploader(
    "Upload your customized file below", 
    type=["xlsx", "csv"]
)

# 2. Load Data (Handles Uploaded File or falls back to Simulated Data)
def load_data():
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)
            
            if 'Date' in user_df.columns and 'Value' in user_df.columns:
                final_df = user_df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
                final_df['ds'] = pd.to_datetime(final_df['ds'])
                pd_stream.sidebar.success("🎉 Custom data loaded successfully!")
                return final_df, False
            else:
                pd_stream.sidebar.error("❌ Column Layout Error: File must have exact 'Date' and 'Value' headers.")
        except Exception as e:
            pd_stream.sidebar.error(f"❌ Read Error: {e}")
            
    # Default mock behavior if nothing is uploaded
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
    pd_stream.info("💡 **Demo Mode:** Showing generated mock telemetry. Use the template down in the left panel to inject your metrics!")

# 3. Train Model and Run Forecast
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df[['ds', 'y']])

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# 4. Create Interactive Charts
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['ds'], 
    y=df['y'], 
    name="Historical Record", 
    mode='markers', 
    marker=dict(color='#222222', size=3)
))

fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat'], 
    name="Predictive Median Target", 
    line=dict(color='#0066cc', width=2.5)
))

fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_upper'], 
    name="Upper Ceiling Trend (Aggressive)", 
    line=dict(dash='dash', color='rgba(0,102,204,0.25)')
))

fig.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_lower'], 
    name="Lower Floor Trend (Conservative)", 
    line=dict(dash='dash', color='rgba(0,102,204,0.25)'), 
    fill='tonexty'
))

fig.update_layout(
    title="Custom Predictive Horizon Workspace", 
    xaxis_title="Timeline Interval", 
    yaxis_title="Currency Value ($)", 
    template="plotly_white"
)

pd_stream.plotly_chart(fig, use_container_width=True)

# 5. Show Metrics Summary
pd_stream.subheader("📊 Key Predictive Inferences")
col1, col2 = pd_stream.columns(2)
with col1:
    pd_stream.metric(
        label="Terminal Prediction Valuation", 
        value=f"${forecast['yhat'].iloc[-1]:,.2f}"
    )
with col2:
    uncertainty_range = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2
    pd_stream.metric(
        label="Variance Uncertainty Band", 
        value=f"± ${uncertainty_range:,.2f}"
    )
