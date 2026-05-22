import os
import pickle
import requests
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

DRIVE_FILES = {
    'lstm_meta.pkl'    : '1tuOWKn6YTTxXuFAXRvTMOcKT8GXg2F0o',
    'lstm_model.keras' : '1SurytprBGVGCAdglf-DTcbua-uHdBIdA',
    'lstm_results.pkl' : '1mV5U1hvuiXpSbFH70Yrr4XFN8yfM7XtW',
    'prophet_data.pkl' : '1a5q5k2nz8K_w-ANiWhgtAxivMLvf_zZs',
    'prophet_model.pkl': '1a8TrZ9TvOFfuTb_YRXSprAGNtSHgovyw',
    'scaler_close.pkl' : '1bGzguJt9N8f4pcgU8JoB_1HDWo-B5vae',
    'scaler.pkl'       : '1SbgfXmXX8E2rI9RYc3bA4i-Z8siX_lwt',
}

MODEL_DIR = '/tmp/ihsg_models'
DATA_URL  = "https://raw.githubusercontent.com/MutiaraCR/Dataset/refs/heads/main/ihsg_daily.csv"


def _download_file(file_id: str, dest_path: str):
    url     = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(url, params={'confirm': value}, stream=True)
            break
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def _ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, file_id in DRIVE_FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            _download_file(file_id, dest)


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

    delta         = df2['Close'].diff()
    gain          = delta.where(delta > 0, 0).rolling(14).mean()
    loss          = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df2['RSI_14'] = 100 - (100 / (1 + gain / loss))

    ema_12             = df2['Close'].ewm(span=12, adjust=False).mean()
    ema_26             = df2['Close'].ewm(span=26, adjust=False).mean()
    df2['MACD']        = ema_12 - ema_26
    df2['MACD_signal'] = df2['MACD'].ewm(span=9, adjust=False).mean()
    df2['MACD_hist']   = df2['MACD'] - df2['MACD_signal']

    low_14               = df2['Low'].rolling(window=14).min()
    high_14              = df2['High'].rolling(window=14).max()
    df2['Stochastic_%K'] = ((df2['Close'] - low_14) / (high_14 - low_14)) * 100
    df2['Stochastic_%D'] = df2['Stochastic_%K'].rolling(window=3).mean()

    return df, df2


@st.cache_resource
def load_lstm_results():
    _ensure_models()
    with open(os.path.join(MODEL_DIR, 'lstm_results.pkl'), 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_lstm_model_and_scalers():
    """Return (model, scaler_4fitur, scaler_close)"""
    _ensure_models()
    model = load_model(os.path.join(MODEL_DIR, 'lstm_model.keras'))
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler_close.pkl'), 'rb') as f:
        scaler_close = pickle.load(f)
    return model, scaler, scaler_close


@st.cache_resource
def load_prophet_model():
    _ensure_models()
    with open(os.path.join(MODEL_DIR, 'prophet_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'prophet_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return model, data