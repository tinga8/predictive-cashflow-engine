import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# Set up the web page layout
pd_stream.set_page_config(page_title="Universal AI Forecaster", layout="wide")
pd_stream.title("🎯 Universal Business Intelligence Forecasting Engine")
pd_stream.caption("Built with Python & Meta's Prophet | 100% Crash-Proof Smart Data Mapper")

# 1. Generate Interactive Inputs in Sidebar
pd_stream.sidebar.header("🎛️ Model Parameters")
forecast_months = pd_stream.sidebar.slider("Forecast Horizon (Months)", 3, 24, 12)

# --- 2. FILE UPLOADER PANEL ---
pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("📂 Universal Data Input")

uploaded_file = pd_stream.sidebar.file_uploader(
    "Upload ANY Excel or CSV spreadsheet safely", 
    type=["xlsx", "csv"]
)

# 3. THE OMNIVOROUS PARSING ENGINE (ZERO ASSUMPTIONS)
def robust_data_parser(file):
    try:
        # Read file format dynamically
        if file.name.endswith('.csv'):
            raw_df = pd.read_csv(file)
        else:
            raw_df = pd.read_excel(file)
            
        if raw_df.empty:
            return None, None, True
            
        # Clean column metadata structures
        raw_df.columns = raw_df.columns.astype(str).str.strip()
        
        # Step A: Identify types of columns present
        numeric_cols = []
        text_cols = []
        date_cols = []
        
        for col in raw_df.columns:
            # Check if column is primary date-like structures
            parsed_dates = pd.to_datetime(raw_df[col], errors='coerce')
            if parsed_dates.notna().sum() > len(raw_df) * 0.4:
                date_cols.append(col)
                continue
                
            # Check if column is numeric metrics
            parsed_nums = pd.to_numeric(raw_df[col], errors='coerce')
            if parsed_nums.notna().sum() > len(raw_df) * 0.4:
                numeric_cols.append(col)
            else:
                text_cols.append(col)

        # ----------------------------------------------------
        # SCENARIO 1: FLAT RECTANGULAR LAYOUT (Vertical Columns: Date + Value)
        # ----------------------------------------------------
        if date_cols and numeric_cols:
            d_col = date_cols[0]
            v_col = numeric_cols[0]
            
            flat_df = raw_df[[d_col, v_col]].copy()
            flat_df.columns = ['ds', 'y']
            flat_df['ds'] = pd.to_datetime(flat_df['ds'], errors='coerce')
            flat_df['y'] = pd.to_numeric(flat_df['y'], errors='coerce')
            flat_df = flat_df.dropna().sort_values('ds').reset_index(drop=True)
            
            # Aggregate down to Month-Start boundaries
            flat_df['ds'] = flat_df['ds'].dt.to_period('M').dt.to_timestamp()
            flat_df = flat_df.groupby('ds', as_index=False).sum()
            
            breakdown_df = pd.DataFrame({'Product': ['Database Ledger Target'], 'Total_Volume': [flat_df['y'].sum()]})
            pd_stream.sidebar.success("🚀 Auto-Detected: Vertical Timeline Format")
            return flat_df, breakdown_df, False

        # ----------------------------------------------------
        # SCENARIO 2: MATRIX HORIZONTAL LAYOUT (Rows = Items, Columns = Time series)
        # ----------------------------------------------------
        valid_time_headers = []
        for col in raw_df.columns:
            if pd.to_datetime(col, errors='coerce') is not pd.NaT or '-' in col:
                if pd.to_numeric(raw_df[col], errors='coerce').notna().sum() > 0:
                    valid_time_headers.append(col)
                    
        if valid_time_headers and text_cols:
            label_col = text_cols[0]
            
            cleaned_df = raw_df[raw_df[label_col].notna()].copy()
            cleaned_df[label_col] = cleaned_df[label_col].astype(str).str.strip()
            
            valid_rows = cleaned_df[pd.to_numeric(cleaned_df[valid_time_headers[0]], errors='coerce').notna()]
            row_items = valid_rows[label_col].unique().tolist()
            
            if row_items:
                selected_item = pd_stream.sidebar.selectbox("🎯 Select Target Row to Forecast", row_items)
                target_row = valid_rows[valid_rows[label_col] == selected_item].iloc[0]
                
                timeline_array = []
                values_array = []
                for col in valid_time_headers:
                    val = pd.to_numeric(target_row[col], errors='coerce')
                    parsed_dt = pd.to_datetime(col, errors='coerce')
                    if pd.notna(val) and pd.notna(parsed_dt):
                        values_array.append(float(val))
                        timeline_array.append(parsed_dt)
                        
                final_df = pd.DataFrame({'ds': timeline_array, 'y': values_array}).sort_values('ds').reset_index(drop=True)
                
                allocation_list = []
                for item in row_items:
                    try:
                        item_row = valid_rows[valid_rows[label_col] == item].iloc[0]
                        total_v = sum([float(pd.to_numeric(item_row[c], errors='coerce')) for c in valid_time_headers if pd.notna(pd.to_numeric(item_row[c], errors='coerce'))])
                        allocation_list.append({'Product': item, 'Total_Volume': total_v})
                    except:
                        pass
                    
                breakdown_df = pd.DataFrame(allocation_list)
                pd_stream.sidebar.success(f"🚀 Auto-Detected: Matrix Row Format ({selected_item})")
                return final_df, breakdown_df, False

        return None, None, True
    except:
        return None, None, True

# Route operational status
if uploaded_file is not None:
    df, breakdown_df, is_demo = robust_data_parser(uploaded_file)
    if df is None or len(df) < 2:
        is_demo = True
        pd_stream.sidebar.warning("⚠️ Formatting parsing limit met. Displaying demonstration dashboard baseline.")
else:
    is_demo = True

if is_demo:
    if uploaded_file is None:
        pd_stream.info("💡 **Universal Smart Dashboard Live:** Ready for any CSV/Excel layout. Upload a file on the left panel to test parsing automation!")
        
    dates = pd.date_range(start="2024-01-01", end="2025-12-01", freq="MS")
    trend = np.linspace(3000, 6000, len(dates))
    seasonal_effect = np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 500
    revenue = trend + seasonal_effect + np.random.normal(0, 50, len(dates))
    df = pd.DataFrame({'ds': dates, 'y': revenue})
    
    breakdown_df = pd.DataFrame({
        'Product': ['Item A', 'Item B', 'Item C', 'Item D', 'Item E'],
        'Total_Volume': [500, 400, 300, 200, 100]
    })

# --- 4. DYNAMIC DATE SELECTION FILTERS ---
df['Year'] = df['ds'].dt.year
df['Month_Name'] = df['ds'].dt.strftime('%B')

all_years = sorted(df['Year'].unique().tolist())
all_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("⏳ Time Frame Filters")
selected_years = pd_stream.sidebar.multiselect("📆 Choose Year(s)", options=all_years, default=all_years)
selected_months = pd_stream.sidebar.multiselect("🌙 Choose Month(s)", options=all_months, default=all_months)

filtered_df = df[(df['Year'].isin(selected_years)) & (df['Month_Name'].isin(selected_months))].copy()

if filtered_df.empty or len(filtered_df) < 2:
    pd_stream.error("⚠️ Keep at least two data point blocks selected in the timeframe parameters to calculate model arrays.")
else:
    # 5. ML Prophet Core Training execution
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(filtered_df[['ds', 'y']])
    
    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast = model.predict(future)
    
    # Combined plot configurations
    historical_clean = filtered_df[['ds', 'y']].copy().rename(columns={'y': 'Value'})
    historical_clean['Type'] = 'Historical Track'
    forecast_clean = forecast[['ds', 'yhat']].tail(forecast_months).copy().rename(columns={'yhat': 'Value'})
    forecast_clean['Type'] = 'AI Projections'
    combined_chart_df = pd.concat([historical_clean, forecast_clean], ignore_index=True)

    # --- 6. RENDER INTERACTION PANELS ---
    tab1, tab2, tab3 = pd_stream.tabs(["📈 Executive Chart Workspace", "📊 Comparative Breakdown", "📋 Data Matrix Ledger"])
    
    with tab1:
        pd_stream.subheader("Custom Forecast Analysis Canvas")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name="Actual Historical Metrics", mode='markers+lines', line=dict(color='#222222'), marker=dict(size=6)))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Projected Median", line=dict(color='#0066cc', width=3)))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Target Ceiling", line=dict(dash='dash', color='rgba(0,102,204,0.3)')))
        fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Operational Floor", line=dict(dash='dash', color='rgba(0,102,204,0.3)'), fill='tonexty'))
        fig_line.update_layout(xaxis_title="Timeline Interval", yaxis_title="Valuation Scale ($)", template="plotly_white")
        pd_stream.plotly_chart(fig_line, use_container_width=True)
        
    with tab2:
        col_c1, col_c2 = pd_stream.columns(2)
        with col_c1:
            pd_stream.subheader("Structural Share Distribution Matrix")
            fig_pie = px.pie(breakdown_df, values='Total_Volume', names='Product', color_discrete_sequence=px.colors.qualitative.Safe, hole=0.45)
            pd_stream.plotly_chart(fig_pie, use_container_width=True)
        with col_c2:
