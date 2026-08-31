import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import io

# Set up the web page layout
pd_stream.set_page_config(page_title="AI Cashflow Forecaster", layout="wide")
pd_stream.title("📈 AI-Powered Revenue & Cashflow Forecasting Engine")
pd_stream.caption("Built with Python & Meta's Prophet Library | Custom Multi-Category Extension")

# 1. Generate Interactive Inputs in Sidebar
pd_stream.sidebar.header("Model Parameters")
forecast_days = pd_stream.sidebar.slider("Forecast Horizon (Days)", 30, 180, 90)

# Section: File Management in Sidebar
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Custom Data Input")

# File Uploader
uploaded_file = pd_stream.sidebar.file_uploader(
    "Upload your customized file below", 
    type=["xlsx", "csv"]
)

# 2. Parse Custom Complex Layout or Fallback
def process_uploaded_data(file):
    try:
        # Read the file
        if file.name.endswith('.csv'):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)
            
        # Clean up column spaces and strip headers
        raw_df.columns = raw_df.columns.str.strip()
        
        # Scenario A: User uploaded the complex multi-product spreadsheet
        first_col = raw_df.columns[0]
        if "Category" in first_col or "Product" in first_col:
            # Clean product rows (remove empty spaces and category headers)
            raw_df[first_col] = raw_df[first_col].astype(str).str.strip()
            valid_products = raw_df[raw_df.iloc[:, 1].notna()][first_col].tolist()
            
            # Add a dropdown selector in the sidebar for products
            selected_item = pd_stream.sidebar.selectbox("🎯 Select Product to Forecast", valid_products)
            
            # Extract row for selected product
            product_row = raw_df[raw_df[first_col] == selected_item].iloc[0]
            
            # Find all historical 2024 columns
            months_2024 = ['Jan-2024', 'Feb-2024', 'Mar-2024', 'Apr-2024', 'May-2024', 'Jun-2024', 
                           'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024', 'Nov-2024', 'Dec-2024']
            
            # Transform horizontal data into vertical Time-Series format
            values = [float(product_row[m]) for m in months_2024]
            date_range = pd.date_range(start="2024-01-01", end="2024-12-01", freq="MS")
            
            final_df = pd.DataFrame({'ds': date_range, 'y': values})
            pd_stream.sidebar.success(f"🎉 Processing model for: {selected_item}")
            return final_df, False

        # Scenario B: Flat layout with 'Date' and 'Value' fields
        elif 'Date' in raw_df.columns and 'Value' in raw_df.columns:
            final_df = raw_df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
            final_df['ds'] = pd.to_datetime(final_df['ds'])
            pd_stream.sidebar.success("🎉 Flat layout data loaded successfully!")
            return final_df, False
            
        else:
            pd_stream.sidebar.error("❌ Structure Mismatch: Unrecognized layout column format.")
            return None, True
            
    except Exception as e:
        pd_stream.sidebar.error(f"❌ Read Error: {e}")
        return None, True

# Execution logic routing
if uploaded_file is not None:
    df, is_demo = process_uploaded_data(uploaded_file)
    if df is None: # Error fallback
        is_demo = True
else:
    is_demo = True

if is_demo:
    if uploaded_file is None:
        pd_stream.info("💡 **Demo Mode:** Showing generated mock telemetry. Drop your multi-category dataset in the sidebar panel!")
    # Generate default demo data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2026-08-31", freq="D")
    revenue = np.clip(np.linspace(1500, 3500, len(dates)) + np.random.normal(0, 200, len(dates)), 500, None)
    df = pd.DataFrame({'ds': dates, 'y': revenue})

# 3. Train Model and Run Forecast
model = Prophet(yearly_seasonality=False, weekly_seasonality=False) # Turned off for pure monthly trends
model.fit(df[['ds', 'y']])

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# 4. Create Interactive Charts
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Historical Record", mode='markers+lines', marker=dict(color='#222222', size=5)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predictive Median Target", line=dict(color='#0066cc', width=2.5)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Bound", line=dict(dash='dash', color='rgba(0,102,204,0.2)')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Bound", line=dict(dash='dash', color='rgba(0,102,204,0.2)'), fill='tonexty'))

fig.update_layout(title="Custom Predictive Horizon Workspace", xaxis_title="Timeline Interval", yaxis_title="Metric Value", template="plotly_white")
pd_stream.plotly_chart(fig, use_container_width=True)

# 5. Show Metrics Summary
pd_stream.subheader("📊 Key Predictive Inferences")
col1, col2 = pd_stream.columns(2)
with col1:
    pd_stream.metric(label="Terminal Forecast Value", value=f"${forecast['yhat'].iloc[-1]:,.2f}")
with col2:
    uncertainty_range = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2
    pd_stream.metric(label="Variance Uncertainty Band", value=f"± ${uncertainty_range:,.2f}")
