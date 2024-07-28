import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

import yfinance as yf

st.title("Stock Price Predictor App")

# Text input for the stock ID
stock = st.text_input("Enter the stock ID", 'NVDA')

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download stock data
nvidia_data = yf.download(stock, start=start, end=end)

# Load the pre-trained model
model = load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data")
st.write(nvidia_data)

splitting_len = int(len(nvidia_data) * 0.7)
x_test = pd.DataFrame(nvidia_data['Close'][splitting_len:])

# Define function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Plotting moving averages
st.subheader('Original Close Price and MA for 250 days')
nvidia_data['MA_for_250_days'] = nvidia_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15, 6), nvidia_data['MA_for_250_days'], nvidia_data))

st.subheader('Original Close Price and MA for 200 days')
nvidia_data['MA_for_200_days'] = nvidia_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15, 6), nvidia_data['MA_for_200_days'], nvidia_data))

st.subheader('Original Close Price and MA for 100 days')
nvidia_data['MA_for_100_days'] = nvidia_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15, 6), nvidia_data['MA_for_100_days'], nvidia_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), nvidia_data['MA_for_100_days'], nvidia_data, 1, nvidia_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

inv_pred = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pred.reshape(-1)
}, index=nvidia_data.index[splitting_len+100:])

st.subheader("Original values vs Predicted values")
st.write(plotting_data)

st.subheader('Original Close price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([nvidia_data['Close'][:splitting_len+100], plotting_data], axis=0))
plt.legend(['Data not used', "Original Test data", "Predicted Test data"])
st.pyplot(fig)
