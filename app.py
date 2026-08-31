import streamlit as pd_stream
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import re

pd_stream.set_page_config(page_title="Enterprise BI Canvas", layout="wide")
pd_stream.title("📊 Enterprise Power BI Style Forecasting Canvas")
pd_stream.caption("Production Machine Learning Dashboard | Meta Prophet Predictive Analytics")

pd_stream.sidebar.header("🎛️ Dashboard Filters")
forecast_months = pd_stream.sidebar.slider("Forecast Horizon (Future Months)", 3, 24, 12)

pd_stream.sidebar.markdown("---")
chart_selection = pd_stream.sidebar.selectbox(
    "Select Visualization Style",
    ["📉 Line Chart", "📊 Column Chart", "⛰️ Stacked Area Chart", "🍩 Donut Chart", "📋 Ledger Matrix"]
)

pd_stream.sidebar.markdown("---")
uploaded_file = pd_stream.sidebar.file_uploader("Upload financial Excel/CSV files", type=["xlsx", "csv"])

def parse_financial_data(file):
    try:
        raw_df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if raw_df.empty: return None, None, True
        raw_df.columns = raw_df.columns.astype(str).str.strip()
        month_year_pattern = re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_\s]?\d{2,4}\b', re.IGNORECASE)
        time_headers, descriptive_headers = [], []
        for col in raw_df.columns:
            if month_year_pattern.search(col):
                if "growth %" not in col.lower(): time_headers.append(col)
            else: descriptive_headers.append(col)
        if time_headers and descriptive_headers:
            label_col = descriptive_headers[0]
            cleaned_df = raw_df[raw_df[label_col].notna()].copy()
            cleaned_df[label_col] = cleaned_df[label_col].astype(str).str.strip()
            ignored = ['nan', 'none', 'category', 'product', 'beverages', 'snacks', 'dairy', 'frozen foods', 'personal care']
            valid_rows = cleaned_df[~cleaned_df[label_col].str.lower().isin(ignored)].copy()
            for col in time_headers: valid_rows[col] = pd.to_numeric(valid_rows[col], errors='coerce')
            valid_rows = valid_rows.dropna(subset=time_headers, how='all')
            available_products = sorted(valid_rows[label_col].unique().tolist())
            if available_products:
                selected_product = pd_stream.sidebar.selectbox("🎯 Select Product Category", available_products)
                product_data_row = valid_rows[valid_rows[label_col] == selected_product].iloc[0]
                dates_list, values_list = [], []
                for col in time_headers:
                    val = product_data_row[col]
                    date_match = month_year_pattern.search(col)
                    if date_match:
                        dt_parsed = pd.to_datetime(date_match.group(), errors='coerce')
                        if pd.notna(val) and pd.notna(dt_parsed):
                            values_list.append(float(val))
                            dates_list.append(dt_parsed)
                final_series_df = pd.DataFrame({'ds': dates_list, 'y': values_list})
                final_series_df = final_series_df.groupby('ds', as_index=False).last().sort_values('ds').reset_index(drop=True)
                allocation_array = []
                for p in available_products:
                    p_row = valid_rows[valid_rows[label_col] == p].iloc[0]
                    tot = sum([float(p_row[c]) for c in time_headers if pd.notna(p_row[c])])
                    allocation_array.append({'Product': p, 'Total_Volume': tot})
                breakdown_df = pd.DataFrame(allocation_array)
                pd_stream.sidebar.success(f"✅ Ingested Data for: {selected_product}")
                return final_series_df, breakdown_df, False
        return None, None, True
    except:
        return None, None, True

if uploaded_file is not None:
    df, breakdown_df, is_demo = parse_financial_data(uploaded_file)
    if df is None or len(df) < 2: is_demo = True
else:
    is_demo = True

if is_demo:
    if uploaded_file is None: pd_stream.info("💡 Upload data file in the sidebar to populate your custom workspace.")
    dates = pd.date_range(start="2024-01-01", end="2025-12-01", freq="MS")
    revenue = np.linspace(3500, 6500, len(dates)) + np.sin(np.arange(len(dates)) * (2 * np.pi / 12)) * 600
    df = pd.DataFrame({'ds': dates, 'y': revenue})
    breakdown_df = pd.DataFrame({
        'Product': ['Beverages', 'Snacks', 'Dairy Products', 'Frozen Deliveries'],
        'Total_Volume': [15200.4, 12400.8, 9800.2, 11500.6]
    })

df['Year'] = df['ds'].dt.year
df['Month_Name'] = df['ds'].dt.strftime('%B')
available_years = sorted(df['Year'].unique().tolist())
calendar_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

pd_stream.sidebar.markdown("---")
pd_stream.sidebar.header("⏳ Time Frame Filters")
selected_years = pd_stream.sidebar.multiselect("📆 Choose Year(s)", options=available_years, default=available_years)
selected_months = pd_stream.sidebar.multiselect("🌙 Choose Month(s)", options=calendar_months, default=calendar_months)

filtered_df = df[(df['Year'].isin(selected_years)) & (df['Month_Name'].isin(selected_months))].copy()

if filtered_df.empty or len(filtered_df) < 2:
    pd_stream.error("⚠️ Filter Error: Select at least 2 historical data periods.")
else:
    model = Prophet(yearly_seasonality=(len(selected_years) > 1), weekly_seasonality=False, daily_seasonality=False)
    model.fit(filtered_df[['ds', 'y']])
    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast = model.predict(future)
    
    historical_clean = filtered_df[['ds', 'y']].copy().rename(columns={'y': 'Value'})
    historical_clean['Data Type'] = 'Historical Track'
    forecast_clean = forecast[['ds', 'yhat']].tail(forecast_months).copy().rename(columns={'yhat': 'Value'})
    forecast_clean['Data Type'] = 'AI Future Projection'
    combined_rendering_df = pd.concat([historical_clean, forecast_clean], ignore_index=True)

    pd_stream.subheader(f"🎨 Active Style: {chart_selection}")
    
    if "Line Chart" in chart_selection:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['ds'], y=filtered_df['y'], name="Actuals", mode='markers+lines', line=dict(color='#2b2b2b')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color='#0066cc', width=3)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Ceiling", line=dict(dash='dash', color='rgba(0,102,204,0.2)')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Floor", line=dict(dash='dash', color='rgba(0,102,204,0.2)'), fill='tonexty'))
        fig.update_layout(template="plotly_white")
        pd_stream.plotly_chart(fig, use_container_width=True)
        
    elif "Column Chart" in chart_selection:
        fig = px.bar(combined_rendering_df, x='ds', y='Value', color='Data Type', barmode='group', color_discrete_map={'Historical Track': '#333333', 'AI Future Projection': '#0066cc'})
        fig.update_layout(template="plotly_white")
        pd_stream.plotly_chart(fig, use_container_width=True)
        
    elif "Stacked Area Chart" in chart_selection:
        fig = px.area(combined_rendering_df, x='ds', y='Value', color='Data Type', color_discrete_map={'Historical Track': 'rgba(51,51,51,0.6)', 'AI Future Projection': 'rgba(0,102,204,0.6)'})
        fig.update_layout(template="plotly_white")
        pd_stream.plotly_chart(fig, use_container_width=True)
        
    elif "Donut Chart" in chart_selection:
        fig = px.pie(breakdown_df, values='Total_Volume', names='Product', hole=0.45)
        pd_stream.plotly_chart(fig, use_container_width=True)
        
    elif "Ledger Matrix" in chart_selection:
        display_ledger = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_months).copy()
        display_ledger['ds'] = display_ledger['ds'].dt.strftime('%B %Y')
        display_ledger.columns = ['Fiscal Month', 'Expected Target', 'Floor Limit', 'Ceiling Limit']
        pd_stream.dataframe(display_ledger.style.format({'Expected Target': '${:,.2f}', 'Floor Limit': '${:,.2f}', 'Ceiling Limit': '${:,.2f}'}), use_container_width=True, hide_index=True)

    pd_stream.markdown("---")
    pd_stream.subheader("🎯 Executive Performance Benchmarks")
    col_m1, col_m2, col_m3 = pd_stream.columns(3)
    col_m1.metric(label="Terminal Runway Valuation Projection", value=f"${forecast['yhat'].iloc[-1]:,.2f}")
    col_m2.metric(label="Statistical Margin Variance", value=f"± ${(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])/2:,.2f}")
    col_m3.metric(label="Historical Base Running Average", value=f"${filtered_df['y'].mean():,.2f}")
