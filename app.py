import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import re

# Set up clean page configuration
pd_stream.set_page_config(page_title="Enterprise BI Forecasting Engine", layout="wide")
pd_stream.title("📊 Enterprise Power BI Style Forecasting Canvas")
pd_stream.caption("Production-Grade Machine Learning Dashboard | Prophet Time-Series Predictive Analytics")

# 1. GENERATE POWER BI STYLE SIDEBAR CONTROLS
pd_stream.sidebar.header("🎛️ Dashboard Filters")

# Forecast Timeline Slider
forecast_months = pd_stream.sidebar.slider("Forecast Horizon (Future Months)", 3, 36, 12)

# MS Office / Power BI Chart Selector Switch
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("🎨 Chart Settings")
chart_selection = pd_stream.sidebar.selectbox(
    "Select Visualization Style",
    [
        "📉 Line Chart (Forecast with Confidence Bands)",
        "📊 Column Chart (Historical vs. Predicted Volumes)",
        "⛰️ Stacked Area Chart (Cumulative Structural View)",
        "🍩 Donut Chart (Product Portfolio Value Weight)",
        "📋 Ledger Matrix (Pure Tabular Accounting View)"
    ]
)

# File Uploader
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Data Source")
uploaded_file = pd_stream.sidebar.file_uploader("Upload financial Excel or CSV files", type=["xlsx", "csv"])

# 2. DATA INGESTION MATRIX
def parse_financial_data(file):
    try:
        if file.name.endswith('.csv'):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)
            
        if raw_df.empty:
            return None, None, True
            
        # Clean header formatting spaces
        raw_df.columns = raw_df.columns.astype(str).str.strip()
        
        # Regex to isolate time data layout arrays (e.g. 'Jan-2024')
        month_year_pattern = re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_\s]?\d{2,4}\b', re.IGNORECASE)
        
        time_headers = []
        descriptive_headers = []
        
        for col in raw_df.columns:
            if month_year_pattern.search(col):
                if "growth" not in col.lower() and "forecast" not in col.lower():
                    time_headers.append(col)
            else:
                descriptive_headers.append(col)
                
        # SCENARIO A: Complex Horizontal Matrix Sheet
        if time_headers and descriptive_headers:
            label_col = descriptive_headers[0]
            
            cleaned_df = raw_df[raw_df[label_col].notna()].copy()
            cleaned_df[label_col] = cleaned_df[label_col].astype(str).str.strip()
            
            # Remove generic text group total titles
            ignored_titles = ['nan', 'none', 'category', 'product', 'beverages', 'snacks', 'dairy', 'frozen foods', 'personal care']
            valid_rows = cleaned_df[~cleaned_df[label_col].str.lower().isin(ignored_titles)].copy()
            
            for col in time_headers:
                valid_rows[col] = pd.to_numeric(valid_rows[col], errors='coerce')
                
            valid_rows = valid_rows.dropna(subset=time_headers, how='all')
            available_products = sorted(valid_rows[label_col].unique().tolist())
            
            if available_products:
                selected_product = pd_stream.sidebar.selectbox("🎯 Select Product Category", available_products)
                product_data_row = valid_rows[valid_rows[label_col] == selected_product].iloc[0]
                
                dates_list = []
                values_list = []
                for col in time_headers:
                    val = product_data_row[col]
                    dt_parsed = pd.to_datetime(col, errors='coerce')
                    if pd.notna(val) and pd.notna(dt_parsed):
                        values_list.append(float(val))
                        dates_list.append(dt_parsed)
                        
                final_series_df = pd.DataFrame({'ds': dates_list, 'y': values_list}).sort_values('ds').reset_index(drop=True)
                
                # Allocation Summary Matrix for Pie Charts
                allocation_array = []
                for p in available_products:
                    try:
                        p_row = valid_rows[valid_rows[label_col] == p].iloc[0]
                        tot = sum([float(p_row[c]) for c in time_headers if pd.notna(p_row[c])])
                        allocation_array.append({'Product': p, 'Total_Volume': tot})
                    except:
                        pass
                breakdown_df = pd.DataFrame(allocation_array)
                
                pd_stream.sidebar.success(f"✅ Active View: {selected_product}")
                return final_series_df, breakdown_df, False

        # SCENARIO B: Flat Standard List View
        else:
            potential_date = None
            potential_val = None
            for col in raw_df.columns:
                if pd.to_datetime(raw_df[col], errors='coerce').notna().sum() > len(raw_df) * 0.4:
                    potential_date = col
                elif pd.to_numeric(raw_df[col], errors='coerce').notna().sum() > len(raw_df) * 0.4:
                    potential_val = col
                    
            if potential_date and potential_val:
                flat_df = raw_df[[potential_date, potential_val]].copy()
                flat_df.columns = ['ds', 'y']
                flat_df['ds'] = pd.to_datetime(flat_df['ds'], errors='coerce')
                flat_df['y'] = pd.to_numeric(flat_df['y'], errors='coerce')
                flat_df = flat_df.dropna().sort_values('ds').reset_index(drop=True)
                
                flat_df['ds'] = flat_df['ds'].dt.to_period('M').dt.to_timestamp()
                flat_df = flat_df.groupby('ds', as_index=False).sum()
                
                breakdown_df = pd.DataFrame({'Product': ['Database Core Target'], 'Total_Volume': [flat_df['y'].sum()]})
                pd_stream.sidebar.success("✅ Active View: Standard Database Ledger")
                return flat_df, breakdown_df, False
                
        return None, None, True
    except:
        return None, None, True

# Route active datasets
if uploaded_file is not None:
    df, breakdown_df, is_demo = parse_financial_data(uploaded_file)
    if df is None or len(df) < 2:
        is_demo = True
else:
    is_demo = True

if is_demo:
    if uploaded_file is None:
        pd_stream.info("💡 Showing dynamic visualization playground with baseline metrics. Drop your file in the sidebar to populate your custom workspace.")
    else:
        pd_stream.sidebar.warning("⚠️ Formatting schema unreadable. Reverted to balanced baseline profile placeholders.")
        
    # Generate multi-year baseline mock profile (2024 to 2026)
    dates = pd.date_range(start="2024-01-01", end="2026-06-01", freq="MS")
    trend = np.linspace(3500, 6500, len(dates))
    seasonal_effect = np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 600
    revenue = trend + seasonal_effect + np.random.normal(0, 40, len(dates))
    df = pd.DataFrame({'ds': dates, 'y': revenue})
    
    breakdown_df = pd.DataFrame({
        'Product': ['Beverages', 'Snacks', 'Dairy Products', 'Frozen Deliveries', 'Personal Care Items'],
        'Total_Volume': [15200.40, 12400.80, 9800.20, 11500.60, 8400.10]
    })

# 3. ADVANCED TIME FRAME FILTERS BLOCK
df['Year'] = df['ds'].dt.year
df['Month_Name'] = df['ds'].dt.strftime('%B')

available_years = sorted(df['Year'].unique().tolist())
calendar_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("⏳ Time Frame Filters")
selected_years = pd_stream.sidebar.multiselect("📆 Choose Year(s)", options=available_years, default=available_years)
selected_months = pd_stream.sidebar.multiselect("🌙 Choose Month(s)", options=calendar_months, default=calendar_months)

# Apply explicit slice logic
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Month_Name'].isin(selected_months))].copy()

if filtered_df.empty or len(filtered_df) < 2:
    pd_stream.error("⚠️ Filter Error: Please expand your Year/Month filter selections to include at least 2 historical data periods.")
else:
    # 4. PREDICTIVE FORECASTING ENGINE
    has_sufficient_breadth = len(selected_years) > 1
    model = Prophet(yearly_seasonality=has_sufficient_breadth, weekly_seasonality=False, daily_seasonality=False)
    model.fit(filtered_df[['ds', 'y']])
    
    # Calculate months into future horizons cleanly
    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast = model.predict(future)
    
    # Structural separation parsing maps
    historical_clean = filtered_df[['ds', 'y']].copy().rename(columns={'y': 'Value'})
    historical_clean['Data Type'] = 'Historical Actual'
    
    forecast_clean = forecast[['ds', 'yhat']].tail(forecast_months).copy().rename(columns={'yhat': 'Value'})
    forecast_clean['Data Type'] = 'AI Future Projection'
    
    combined_rendering_df = pd.concat([historical_clean, forecast_clean], ignore_index=True)

    # 5. DYNAMIC INTERACTIVE VISUALIZATION MATRIX
    pd_stream.subheader(f"📊 Active Canvas Profile: {chart_selection.split('(')[0].strip()}")
    
    if "Line Chart" in chart_selection:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name="Historical Actuals", mode='markers+lines', line=dict(color='#2b2b2b', width=2), marker=dict(size=6)))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Midpoint Target Forecast", line=dict(color='#0066cc', width=3)))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Optimistic Ceiling (Upper Bound)", line=dict(dash='dash', color='rgba(0,102,204,0.3)')))
