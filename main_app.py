import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Retail Forecasting Dashboard", layout="wide")

st.title("📈 Retail Demand Forecasting Dashboard")

# 1. File upload
uploaded_file = st.file_uploader("📂 Upload sales CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        
        required_cols = {"date", "sku", "sales", "store_id"}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ The CSV must contain these columns: {required_cols}")
        else:
            # Sidebar filters
            st.sidebar.header("🔍 Filter Options")
            sku_options = df["sku"].unique().tolist()
            store_options = df["store_id"].unique().tolist()

            selected_sku = st.sidebar.selectbox("Select SKU", sku_options)
            selected_store = st.sidebar.selectbox("Select Store ID", store_options)
            forecast_days = st.sidebar.slider("📆 Days to Forecast", min_value=7, max_value=90, value=30)

            # Filter data
            filtered_df = df[(df["sku"] == selected_sku) & (df["store_id"] == selected_store)]
            daily_sales = filtered_df.groupby("date").agg({"sales": "sum"}).reset_index()

            if daily_sales.empty:
                st.warning("⚠️ No sales data found for this SKU and store.")
            else:
                # Prepare for Prophet
                prophet_df = daily_sales.rename(columns={"date": "ds", "sales": "y"})

                # Show stats
                st.subheader("📊 Summary Statistics")
                st.metric("Total Sales", int(prophet_df["y"].sum()))
                st.metric("Average Daily Sales", round(prophet_df["y"].mean(), 2))

                # Forecasting
                model = Prophet()
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

                # Forecast plots
                st.subheader("📈 Forecast Plot")
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                st.subheader("📂 Trend & Seasonality")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                # Download forecast
                st.subheader("📥 Download Forecast")
                forecast_download = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                forecast_download.rename(columns={"ds": "date"}, inplace=True)
                csv_buffer = BytesIO()
                forecast_download.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"forecast_{selected_sku}_store{selected_store}.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
