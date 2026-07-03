import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# page title and design setting
st.set_page_config(page_title="Sales Forecaster", layout="wide")
st.title("📈 E-Commerce Sales Forecasting Dashboard")
st.write("This app compares *ARIMA (SARIMAX)* and *Facebook Prophet* models on real-world sales data.")

# STEP 1: DATA LOAD & CACHING
@st.cache_data  
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.date_range(start="2025-01-01", periods=len(df), freq="MS")
    return df

df = load_data()

# option in Sidebar
st.sidebar.header("Forecast Settings")
months_to_forecast = st.sidebar.slider("Select months to forecast:", min_value=3, max_value=12, value=12)

# toggle switch for data preview
if st.sidebar.checkbox("Show Raw Data Preview"):
    st.write(df.head())

# STEP 2: MODEL 1 - ARIMA (SARIMAX)
df_arima = df.set_index('Date')
sales_data = df_arima['Sales']
model_arima = SARIMAX(sales_data, order=(1,1,1), seasonal_order=(1,1,1,12))
results_arima = model_arima.fit(disp=False)

forecast_arima = results_arima.get_forecast(steps=months_to_forecast)
arima_output = forecast_arima.predicted_mean

# STEP 3: MODEL 2 - FACEBOOK PROPHET
df_prophet = pd.DataFrame()
df_prophet['ds'] = df['Date']
df_prophet['y'] = df['Sales']

model_prophet = Prophet()
model_prophet.fit(df_prophet)

future_dates = model_prophet.make_future_dataframe(periods=months_to_forecast, freq='M')
forecast_prophet = model_prophet.predict(future_dates)
prophet_future = forecast_prophet.iloc[-months_to_forecast:]

# STEP 4: PLOTTING GRAPH IN STREAMLIT
st.subheader("Model Comparison: ARIMA vs Facebook Prophet")

fig, ax = plt.subplots(figsize=(12, 5))

# Actual Data
ax.plot(sales_data.index, sales_data, label='Actual Past Sales', color='black', marker='o', linewidth=2)

# ARIMA
ax.plot(arima_output.index, arima_output, label='ARIMA Forecast', color='red', linestyle='--', marker='s')

# Prophet
ax.plot(prophet_future['ds'], prophet_future['yhat'], label='Facebook Prophet Forecast', color='blue', linestyle=':', marker='^')

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# matplotlib graph in streamlit
st.pyplot(fig)

st.success("App loaded successfully! Try changing the slider in the sidebar to dynamically change the forecast horizon.")
