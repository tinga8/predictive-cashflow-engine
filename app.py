import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# Set up the web page layout
pd_stream.set_page_config(page_title="AI Corporate Financial Dashboard", layout="wide")
pd_stream.title("📊 Enterprise Predictive Revenue Analytics Dashboard")
pd_stream.caption("Built with Python, Meta's Prophet & Plotly | Comprehensive Portfolio Business Intelligence View")

# 1. Generate Interactive Inputs in Sidebar
pd_stream.sidebar.header("🎛️ Model Parameters")
forecast_months = pd_stream.sidebar.slider("Forecast Horizon (Months)", 3, 24, 12)

# Section: File Management in Sidebar
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Custom Data Input")

uploaded_file = pd_stream.sidebar.file_uploader(
    "Upload your multi-category asset files below", 
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
        
        # Complex multi-product spreadsheet parse sequence
        if "Category" in first_col or "Product" in first_col:
            raw_df[first_col] = raw_df[first_col].astype(str).str.strip()
            # Drop empty row indicators
            valid_df = raw_df[raw_df.iloc[:, 1].notna()]
            valid_products = valid_df[first_col].tolist()
            
            selected_item = pd_stream.sidebar.selectbox("🎯 Select Product to Forecast", valid_products)
            product_row = raw_df[raw_df[first_col] == selected_item].iloc[0]
            
            months_2024 = ['Jan-2024', 'Feb-2024', 'Mar-2024', 'Apr-2024', 'May-2024', 'Jun-2024', 
                           'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024', 'Nov-2024', 'Dec-2024']
            
            values = [float(product_row[m]) for m in months_2024]
            date_range = pd.date_range(start="2024-01-01", end="2024-12-01", freq="MS")
            
            final_df = pd.DataFrame({'ds': date_range, 'y': values})
            
            # Map full categories breakdown for the structural Pie Charts matrix
            breakdown_dict = {}
            for prod in valid_products:
                try:
                    row_vals = valid_df[valid_df[first_col] == prod].iloc[0]
                    breakdown_dict[prod] = sum([float(row_vals[m]) for m in months_2024])
                except:
                    pass
            breakdown_df = pd.DataFrame(list(breakdown_dict.items()), columns=['Product', 'Total_Volume'])
            
            pd_stream.sidebar.success(f"🎉 Fully loaded data structures for: {selected_item}")
            return final_df, breakdown_df, False

        else:
            pd_stream.sidebar.error("❌ Column layout must use exact 'Category / Product' headers from template.")
            return None, None, True
            
    except Exception as e:
        pd_stream.sidebar.error(f"❌ Core Parse Failure: {e}")
        return None, None, True

# Execution logic routing
if uploaded_file is not None:
    df, breakdown_df, is_demo = process_uploaded_data(uploaded_file)
    if df is None:
        is_demo = True
else:
    is_demo = True

if is_demo:
    if uploaded_file is None:
        pd_stream.info("💡 **Demo Mode Activated:** Showing complete chart matrices with simulation data. Drop your sheet in the sidebar to sync your specific business units!")
    
    # Generate balanced mock timelines
    dates = pd.date_range(start="2024-01-01", end="2025-12-01", freq="MS")
    trend = np.linspace(2500, 5000, len(dates))
    seasonal_effect = np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 800
    revenue = trend + seasonal_effect + np.random.normal(0, 100, len(dates))
    df = pd.DataFrame({'ds': dates, 'y': revenue})
    
    # Generate default category volume mapping
    breakdown_df = pd.DataFrame({
        'Product': ['Cola', 'Juice', 'Chips', 'Nuts', 'Cookies', 'Milk', 'Cheese', 'Pizza', 'Ice Cream'],
        'Total_Volume': [445.2, 420.5, 395.1, 350.8, 380.2, 310.4, 290.9, 410.3, 460.7]
    })

# 3. Train Model and Run Forecast
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(df[['ds', 'y']])

future = model.make_future_dataframe(periods=forecast_months, freq='MS')
forecast = model.predict(future)

# --- VISUALIZATION TABS FRAMEWORK ---
tab1, tab2, tab3 = pd_stream.tabs(["📈 Primary Forecast Canvas", "📊 Component Breakdown Matrix", "📋 Tabular Inferences"])

with tab1:
    pd_stream.subheader("Predictive Horizon Curve and Variance Bounds")
    
    # Interactive Main Line & Area Chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Historical Financial Record", mode='markers+lines', line=dict(color='#222222'), marker=dict(size=6)))
    fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Projected Metric (Median)", line=dict(color='#0066cc', width=3)))
    fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Ceiling (Optimistic Target)", line=dict(dash='dash', color='rgba(0,102,204,0.3)')))
    fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Floor (Conservative Target)", line=dict(dash='dash', color='rgba(0,102,204,0.3)'), fill='tonexty'))
    
    fig_line.update_layout(xaxis_title="Timeline Interval", yaxis_title="Valuation Scale ($)", template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    pd_stream.plotly_chart(fig_line, use_container_width=True)

with tab2:
    col_chart1, col_chart2 = pd_stream.columns(2)
    
    with col_chart1:
        pd_stream.subheader("🍰 Product Category Volume Allocation")
        # Dynamic Pie Chart showing volume market share splits
        fig_pie = px.pie(breakdown_df, values='Total_Volume', names='Product', color_discrete_sequence=px.colors.qualitative.Prism, hole=0.4)
        fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        pd_stream.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        pd_stream.subheader("🪟 Monthly Scaling Velocities")
        # Bar Chart showing monthly historical actual steps
        fig_bar = px.bar(df, x='ds', y='y', color='y', labels={'ds': 'Timeline', 'y': 'Valuation'}, color_continuous_scale=px.colors.sequential.Blugrn)
        fig_bar.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        pd_stream.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    pd_stream.subheader("📋 Algorithmic Projections Ledger")
    
    # Format precise output data tables
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_months).copy()
    forecast_table['ds'] = forecast_table['ds'].dt.strftime('%B %Y')
    forecast_table.columns = ['Target Forecast Month', 'Expected Value (Midpoint)', 'Minimum Bound (Floor)', 'Maximum Bound (Ceiling)']
    
    pd_stream.dataframe(
        forecast_table.style.format({
            'Expected Value (Midpoint)': '${:,.2f}', 
            'Minimum Bound (Floor)': '${:,.2f}', 
            'Maximum Bound (Ceiling)': '${:,.2f}'
        }), 
        use_container_width=True, 
        hide_index=True
    )

# 5. Core Performance Matrix Summaries
pd_stream.markdown("---")
pd_stream.subheader("🎯 Top-Line Predictive Benchmarks")
col_m1, col_m2, col_m3 = pd_stream.columns(3)

with col_m1:
    pd_stream.metric(label="Terminal Forecast Runway Valuation", value=f"${forecast['yhat'].iloc[-1]:,.2f}")
with col_m2:
    uncertainty_range = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2
    pd_stream.metric(label="Model Margin Variance Range", value=f"± ${uncertainty_range:,.2f}")
with col_m3:
    historical_avg = df['y'].mean()
    pd_stream.metric(label="Historical Base Period Running Average", value=f"${historical_avg:,.2f}")
