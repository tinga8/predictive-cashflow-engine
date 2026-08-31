import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

# Set up the web page layout
pd_stream.set_page_config(page_title="AI Month-wise Forecaster", layout="wide")
pd_stream.title("📈 AI-Powered Month-Wise Forecasting Engine")
pd_stream.caption("Built with Python & Meta's Prophet Library | Aggregated Monthly Financial Analytics")

# 1. Generate Interactive Inputs in Sidebar
pd_stream.sidebar.header("Model Parameters")
# Slide by number of months now instead of days
forecast_months = pd_stream.sidebar.slider("Forecast Horizon (Months)", 3, 24, 12)

# Section: File Management in Sidebar
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Custom Data Input")

uploaded_file = pd_stream.sidebar.file_uploader(
    "Upload your customized file below", 
    type=["xlsx", "csv"]
)

# 2. Parse Custom Complex Layout or Fallback
def process_uploaded_data(file):
    try:
        if file.name.endswith('.csv'):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)
            
        raw_df.columns = raw_df.columns.str.strip()
        first_col = raw_df.columns[0]
        
        # Scenario A: Complex multi-product spreadsheet (Like your custom sheet)
        if "Category" in first_col or "Product" in first_col:
            raw_df[first_col] = raw_df[first_col].astype(str).str.strip()
            valid_products = raw_df[raw_df.iloc[:, 1].notna()][first_col].tolist()
            
            selected_item = pd_stream.sidebar.selectbox("🎯 Select Product to Forecast", valid_products)
            product_row = raw_df[raw_df[first_col] == selected_item].iloc[0]
            
            # Map historical monthly columns dynamically
            months_2024 = ['Jan-2024', 'Feb-2024', 'Mar-2024', 'Apr-2024', 'May-2024', 'Jun-2024', 
                           'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024', 'Nov-2024', 'Dec-2024']
            
            values = [float(product_row[m]) for m in months_2024]
            date_range = pd.date_range(start="2024-01-01", end="2024-12-01", freq="MS")
            
            final_df = pd.DataFrame({'ds': date_range, 'y': values})
            pd_stream.sidebar.success(f"🎉 Monthly model loaded for: {selected_item}")
            return final_df, False

        # Scenario B: Flat layout with 'Date' and 'Value' fields
        elif 'Date' in raw_df.columns and 'Value' in raw_df.columns:
            final_df = raw_df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
            final_df['ds'] = pd.to_datetime(final_df['ds'])
            
            # CRITICAL: Force group data into pure Month-Start intervals to fix discrepancies
            final_df['ds'] = final_df['ds'].dt.to_period('M').dt.to_timestamp()
            final_df = final_df.groupby('ds', as_index=False).sum()
            
            pd_stream.sidebar.success("🎉 Grouped flat data into monthly intervals!")
            return final_df, False
            
        else:
            pd_stream.sidebar.error("❌ Structure Mismatch: Columns must be formatted properly.")
            return None, True
            
    except Exception as e:
        pd_stream.sidebar.error(f"❌ Read Error: {e}")
        return None, True

# Execution logic routing
if uploaded_file is not None:
    df, is_demo = process_uploaded_data(uploaded_file)
    if df is None:
        is_demo = True
else:
    is_demo = True

if is_demo:
    if uploaded_file is None:
        pd_stream.info("💡 **Demo Mode:** Displaying macro monthly cycles. Upload data to change variables.")
    # Generate default monthly data for testing
    dates = pd.date_range(start="2024-01-01", end="2026-06-01", freq="MS")
    trend = np.linspace(2000, 4500, len(dates))
    seasonal_effect = np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 600
    revenue = trend + seasonal_effect + np.random.normal(0, 150, len(dates))
    df = pd.DataFrame({'ds': dates, 'y': revenue})

# 3. Train Model and Run Forecast
# Turn off daily/weekly cycles since intervals are purely monthly
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(df[['ds', 'y']])

# CRITICAL FIX: set freq='MS' (Month Start) so predictions roll over systematically by month
future = model.make_future_dataframe(periods=forecast_months, freq='MS')
forecast = model.predict(future)

# 4. Create Interactive Charts
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Historical Actuals (Monthly)", mode='markers+lines', line=dict(color='#111111'), marker=dict(size=6)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Monthly Prediction", line=dict(color='#0066cc', width=3)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Bound", line=dict(dash='dash', color='rgba(0,102,204,0.2)')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Bound", line=dict(dash='dash', color='rgba(0,102,204,0.2)'), fill='tonexty'))

fig.update_layout(
    title="Month-Wise Predictive Forecasting Canvas", 
    xaxis_title="Monthly Timeline", 
    yaxis_title="Value ($)", 
    template="plotly_white",
    xaxis=dict(tickformat="%b %Y") # Formats dates cleanly as "Jan 2025", "Feb 2025"
)
pd_stream.plotly_chart(fig, use_container_width=True)

# 5. Show Metrics Summary
pd_stream.subheader("📊 Key Monthly Predictive Inferences")
col1, col2 = pd_stream.columns(2)
with col1:
    pd_stream.metric(label="Target Month Valuation Estimate", value=f"${forecast['yhat'].iloc[-1]:,.2f}")
with col2:
    uncertainty_range = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2
    pd_stream.metric(label="Expected Monthly Variance", value=f"± ${uncertainty_range:,.2f}")

# Optional: Display a readable table containing upcoming predicted months
pd_stream.markdown("### 📋 Upcoming Predicted Months (Raw Values)")
forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_months).copy()
forecast_table['ds'] = forecast_table['ds'].dt.strftime('%B %Y')
forecast_table.columns = ['Forecast Month', 'Expected Midpoint Value', 'Minimum Floor', 'Maximum Ceiling']
pd_stream.dataframe(forecast_table.style.format({
    'Expected Midpoint Value': '${:,.2f}', 'Minimum Floor': '${:,.2f}', 'Maximum Ceiling': '${:,.2f}'
}), use_container_width=True)
