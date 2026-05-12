import pandas as pd
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

DATA_URL = "https://raw.githubusercontent.com/MutiaraCR/Dataset/refs/heads/main/ihsg_daily.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    df2 = df.copy()
    df2['Volatility_pct'] = ((df2['High'] - df2['Low']) / df2['Open']) * 100
    df2['Return']         = ((df2['Close'] - df2['Open']) / df2['Open']) * 100
    df2['MA_5']           = df2['Close'].rolling(window=5).mean()
    df2['MA_10']          = df2['Close'].rolling(window=10).mean()
    df2['EMA_5']          = df2['Close'].ewm(span=5, adjust=False).mean()
    df2['EMA_10']         = df2['Close'].ewm(span=10, adjust=False).mean()

    delta    = df2['Close'].diff()
    gain     = delta.where(delta > 0, 0).rolling(14).mean()
    loss     = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df2['RSI_14'] = 100 - (100 / (1 + gain / loss))

    ema_12 = df2['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df2['Close'].ewm(span=26, adjust=False).mean()
    df2['MACD']        = ema_12 - ema_26
    df2['MACD_signal'] = df2['MACD'].ewm(span=9, adjust=False).mean()
    df2['MACD_hist']   = df2['MACD'] - df2['MACD_signal']

    low_14  = df2['Low'].rolling(window=14).min()
    high_14 = df2['High'].rolling(window=14).max()
    df2['Stochastic_%K'] = ((df2['Close'] - low_14) / (high_14 - low_14)) * 100
    df2['Stochastic_%D'] = df2['Stochastic_%K'].rolling(window=3).mean()

    return df, df2


@st.cache_resource
def load_lstm_results():
    with open('models/lstm_results.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_prophet_model():
    with open('models/prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/prophet_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return model, data