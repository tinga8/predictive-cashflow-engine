import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# Set up the web page layout
pd_stream.set_page_config(page_title="AI BI Decision Dashboard", layout="wide")
pd_stream.title("🎯 Advanced Financial Decision Canvas")
pd_stream.caption("Built with Python & Meta's Prophet | High-Flexibility Temporal Filtering Matrix")

# 1. Generate Interactive Inputs in Sidebar
pd_stream.sidebar.header("🎛️ Model Parameters")
forecast_months = pd_stream.sidebar.slider("Forecast Horizon (Months)", 3, 24, 12)

# --- 2. FILE UPLOADER & DATA PARSING ---
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Custom Data Input")

uploaded_file = pd_stream.sidebar.file_uploader(
    "Upload your multi-category asset files below", 
    type=["xlsx", "csv"]
)

def process_uploaded_data(file):
    try:
        if file.name.endswith('.csv'):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)
            
        raw_df.columns = raw_df.columns.str.strip()
        first_col = raw_df.columns
        
        if "Category" in first_col or "Product" in first_col:
            raw_df[first_col] = raw_df[first_col].astype(str).str.strip()
            valid_df = raw_df[raw_df.iloc[:, 1].notna()]
            valid_products = valid_df[first_col].tolist()
            
            selected_item = pd_stream.sidebar.selectbox("🎯 Select Product to Analyze", valid_products)
            product_row = raw_df[raw_df[first_col] == selected_item].iloc
            
            months_2024 = ['Jan-2024', 'Feb-2024', 'Mar-2024', 'Apr-2024', 'May-2024', 'Jun-2024', 
                           'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024', 'Nov-2024', 'Dec-2024']
            
            values = [float(product_row[m]) for m in months_2024]
            date_range = pd.date_range(start="2024-01-01", end="2024-12-01", freq="MS")
            
            final_df = pd.DataFrame({'ds': date_range, 'y': values})
            
            breakdown_dict = {}
            for prod in valid_products:
                try:
                    row_vals = valid_df[valid_df[first_col] == prod].iloc
                    breakdown_dict[prod] = sum([float(row_vals[m]) for m in months_2024])
                except:
                    pass
            breakdown_df = pd.DataFrame(list(breakdown_dict.items()), columns=['Product', 'Total_Volume'])
            
            pd_stream.sidebar.success(f"🎉 Loaded structures for: {selected_item}")
            return final_df, breakdown_df, False
        else:
            pd_stream.sidebar.error("❌ Layout must use exact 'Category / Product' headers.")
            return None, None, True
    except Exception as e:
        pd_stream.sidebar.error(f"❌ Core Parse Failure: {e}")
        return None, None, True

# Route data source
if uploaded_file is not None:
    df, breakdown_df, is_demo = process_uploaded_data(uploaded_file)
    if df is None:
        is_demo = True
else:
    is_demo = True

if is_demo:
    if uploaded_file is None:
        pd_stream.info("💡 **Demo Mode:** Showing dynamic date filtering using multi-year simulated data. Upload your file to customize.")
    
    # Generate mock timeline spanning multiple years for strategic choice testing
    dates = pd.date_range(start="2024-01-01", end="2026-12-01", freq="MS")
    trend = np.linspace(2500, 6000, len(dates))
    seasonal_effect = np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 700
    revenue = trend + seasonal_effect + np.random.normal(0, 100, len(dates))
    df = pd.DataFrame({'ds': dates, 'y': revenue})
    
    breakdown_df = pd.DataFrame({
        'Product': ['Cola', 'Juice', 'Chips', 'Nuts', 'Cookies', 'Milk', 'Cheese', 'Pizza', 'Ice Cream'],
        'Total_Volume': [445.2, 420.5, 395.1, 350.8, 380.2, 310.4, 290.9, 410.3, 460.7]
    })

# --- 3. DYNAMIC MONTH & YEAR FLEXIBILITY SELECTORS ---
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("⏳ Time Frame Filters")

# Extract available parameters from data
df['Year'] = df['ds'].dt.year
df['Month_Name'] = df['ds'].dt.strftime('%B')

all_years = sorted(df['Year'].unique().tolist())
all_months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]

# Year Multi-Select (Defaults to selecting all available years)
selected_years = pd_stream.sidebar.multiselect(
    "📆 Choose Year(s)", 
    options=all_years, 
    default=all_years
)

# Month Multi-Select (Defaults to selecting all 12 months)
selected_months = pd_stream.sidebar.multiselect(
    "🌙 Choose Month(s)", 
    options=all_months, 
    default=all_months
)

# Apply User-Selected Filters to the dataset
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Month_Name'].isin(selected_months))].copy()

# Guard rail to make sure the app doesn't crash if the user unchecks everything
if filtered_df.empty:
    pd_stream.error("⚠️ No data points match your filter choices. Please choose at least one Year and one Month in the sidebar panel.")
else:
    # 4. Train Model and Run Forecast based strictly on filtered criteria
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(filtered_df[['ds', 'y']])

    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast = model.predict(future)

    # Combine datasets for cohesive visualization structures
    historical_clean = filtered_df[['ds', 'y']].copy().rename(columns={'y': 'Value'})
    historical_clean['Type'] = 'Filtered Actuals'

    forecast_clean = forecast[['ds', 'yhat']].tail(forecast_months).copy().rename(columns={'yhat': 'Value'})
    forecast_clean['Type'] = 'AI Projections'

    combined_chart_df = pd.concat([historical_clean, forecast_clean], ignore_index=True)

    # --- 5. RENDER CHOSEN CONFIGURATIONS ---
    tab1, tab2, tab3 = pd_stream.tabs(["📈 Executive Chart Workspace", "📊 Comparative Breakdown", "📋 Data Matrix Ledger"])

    with tab1:
        pd_stream.subheader("Custom Forecast Analysis Canvas")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name="Filtered Historical Record", mode='markers+lines', line=dict(color='#222222'), marker=dict(size=6)))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Projected Metric (Median)", line=dict(color='#0066cc', width=3)))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Ceiling Trend", line=dict(dash='dash', color='rgba(0,102,204,0.3)')))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Floor Trend", line=dict(dash='dash', color='rgba(0,102,204,0.3)'), fill='tonexty'))
        
        fig_line.update_layout(xaxis_title="Timeline Interval", yaxis_title="Valuation Scale ($)", template="plotly_white")
        pd_stream.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        col_c1, col_c2 = pd_stream.columns(2)
        with col_c1:
            pd_stream.subheader("🍩 Category Contribution Breakdown")
            fig_pie = px.pie(breakdown_df, values='Total_Volume', names='Product', color_discrete_sequence=px.colors.qualitative.Safe, hole=0.45)
            pd_stream.plotly_chart(fig_pie, use_container_width=True)
        with col_c2:
            pd_stream.subheader("📊 Chronological Trend Velocity")
            fig_bar = px.bar(combined_chart_df, x='ds', y='Value', color='Type', barmode='group', color_discrete_map={'Filtered Actuals': '#333333', 'AI Projections': '#0066cc'})
            fig_bar.update_layout(template="plotly_white")
            pd_stream.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        pd_stream.subheader("📋 Decision Ledger Matrix")
        
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

    # 6. Core Performance Matrix Summaries
    pd_stream.markdown("---")
    pd_stream.subheader("🎯 Active Top-Line Benchmarks (Based on Custom Selections)")
    col_m1, col_m2, col_m3 = pd_stream.columns(3)

    with col_m1:
        pd_stream.metric(label="Terminal Forecast Runway Valuation", value=f"${forecast['yhat'].iloc[-1]:,.2f}")
    with col_m2:
        uncertainty_range = (forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2
        pd_stream.metric(label="Model Margin Variance Range", value=f"± ${uncertainty_range:,.2f}")
    with col_m3:
        historical_avg = filtered_df['y'].mean()
        pd_stream.metric(label="Selected Base Period Running Average", value=f"${historical_avg:,.2f}")
