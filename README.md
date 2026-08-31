# 📈 AI-Powered Revenue & Cashflow Forecasting Engine

A permanently hosted, interactive Time-Series Forecasting web application built using **Python**, **Meta's Prophet Library**, and **Streamlit**. This business intelligence tool allows corporate finance teams, founders, and stakeholders to upload historical financial records and instantly generate a 90-day predictive look-ahead horizon with risk-adjusted uncertainty bounds.

🌐 **Live Application URL:** [PASTE_YOUR_STREAMLIT_APP_LINK_HERE]

---

## 💼 Business Problem & Solution Space

Traditional financial forecasting relies heavily on static spreadsheet formulas that fail to capture nonlinear market dynamics, shifting seasonal variables, and complex compounding growth trends. 

This engine solves that visibility gap by applying **Bayesian curve fitting** to historical transactional data. It automatically decomposes business performance telemetry into long-term trends, weekly cyclical variations, and yearly seasonal patterns—giving executive leadership an algorithmic "Best Case vs. Worst Case" roadmap for strategic runway planning.

---

## 🛠️ Tech Stack & Architecture

*   **Core Modeling Engine:** Meta (Facebook) Prophet Library for robust time-series forecasting.
*   **Interface & UX Framework:** Streamlit Community Cloud for fast, responsive web rendering.
*   **Data Manipulation:** Pandas & NumPy for matrix operations, vector scaling, and date-time parsing.
*   **Interactive Visualizations:** Plotly Graph Objects for responsive, hover-enabled data points.
*   **Excel Engine Dependency:** OpenPyXL for runtime memory processing of `.xlsx` binary data streams.

---

## 🚀 Key Engineering Features

1. **Dynamic Custom Data ingestion:** Users can download a standardized sample Excel layout directly from the dashboard UI, input native organizational data, and re-upload the data file to instantly train the model.
2. **Algorithmic Trend Isolation:** Isolates baseline scaling trajectories from temporary behavioral noise (such as lower sales volume on weekends or seasonal month-end expenditure surges).
3. **Dynamic Horizon Scaling:** Integrated an interactive parameter slider that re-computes mathematical projections across variable timelines (30 to 180 days out) on the fly.
4. **Risk Boundary Calculation:** Computes explicit upper ceiling and lower floor variance paths (`yhat_upper` and `yhat_lower`) to safeguard corporate operations against volatile market disruptions.

---

## 📥 File Structure & Column Mapping Requirements

To execute successful forecasting matrix evaluations on custom assets, uploaded files (`.csv` or `.xlsx`) must strictly leverage the following header configuration:

| Target Column | Required Format | Description |
| :--- | :--- | :--- |
| **`Date`** | YYYY-MM-DD (or standard text date) | The chronological timestamp baseline interval. |
| **`Value`** | Integer / Float (e.g., 2500.50) | The corresponding metric target (Revenue or Cash Flow). |

---

## ⚙️ Local Installation & Setup

Want to run this machine learning app locally on your machine? Execute these simple steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd predictive-cashflow-engine
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit framework locally:**
   ```bash
   streamlit run app.py
   ```
