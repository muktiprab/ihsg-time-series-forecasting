import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data, load_lstm_results, load_prophet_model, load_lstm_model_and_scalers

st.set_page_config(page_title="Prediksi - IHSG", page_icon="🤖", layout="wide")
st.title("🤖 Prediksi IHSG")
st.markdown("Perbandingan hasil forecasting menggunakan **LSTM** dan **Prophet** pada data periode 2020–2025.")

# ── Load semua resource ──────────────────────────────────────
df, df2                     = load_data()
lstm_results                = load_lstm_results()
model_prophet, prophet_data = load_prophet_model()

# ── Unpack LSTM ──────────────────────────────────────────────
lstm_pred       = lstm_results['pred']
lstm_actual     = lstm_results['actual']
lstm_test_index = lstm_results['test_index']
train_index     = lstm_results['train_index']
train_actual    = lstm_results['train_actual']

# ── Unpack Prophet ───────────────────────────────────────────
train_prophet    = prophet_data['train']
test_prophet     = prophet_data['test']
forecast_prophet = prophet_data['forecast']

prophet_pred   = forecast_prophet['yhat'].values
prophet_actual = test_prophet['y'].values

# ── Fungsi Evaluasi ───────────────────────────────────────────
def evaluate(actual, pred):
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape

mae_l, rmse_l, mape_l = evaluate(lstm_actual, lstm_pred)
mae_p, rmse_p, mape_p = evaluate(prophet_actual, prophet_pred)

# ══════════════════════════════════════════════════════════════
# HASIL EVALUASI MODEL
# ══════════════════════════════════════════════════════════════
st.subheader("Hasil Evaluasi Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 LSTM")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mae_l:.2f}")
    m2.metric("RMSE", f"{rmse_l:.2f}")
    m3.metric("MAPE", f"{mape_l:.2f}%")

with col2:
    st.markdown("#### 🟢 Prophet")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mae_p:.2f}",  delta=f"{mae_l  - mae_p:.2f}",  delta_color="normal")
    m2.metric("RMSE", f"{rmse_p:.2f}", delta=f"{rmse_l - rmse_p:.2f}", delta_color="normal")
    m3.metric("MAPE", f"{mape_p:.2f}%",delta=f"{mape_l - mape_p:.2f}%",delta_color="normal")

st.caption("Delta pada Prophet menunjukkan selisih terhadap LSTM. Nilai positif = Prophet lebih baik.")

st.divider()

# ══════════════════════════════════════════════════════════════
# VISUALISASI LSTM
# ══════════════════════════════════════════════════════════════
st.subheader("Prediksi LSTM")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_index,     train_actual,
        label='Train',           color='steelblue',  linewidth=0.8)
ax.plot(lstm_test_index, lstm_actual,
        label='Aktual (Test)',   color='black',      linewidth=1)
ax.plot(lstm_test_index, lstm_pred,
        label='Prediksi LSTM',  color='darkorange', linewidth=1.2, linestyle='--')
ax.set_title('Prediksi IHSG - LSTM')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Close')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ══════════════════════════════════════════════════════════════
# VISUALISASI PROPHET
# ══════════════════════════════════════════════════════════════
st.subheader("Prediksi Prophet")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_prophet['ds'], train_prophet['y'],
        label='Train',            color='steelblue',     linewidth=0.8)
ax.plot(test_prophet['ds'],  prophet_actual,
        label='Aktual (Test)',    color='black',         linewidth=1)
ax.plot(test_prophet['ds'],  prophet_pred,
        label='Prediksi Prophet', color='mediumseagreen',linewidth=1.2, linestyle='--')
ax.fill_between(
    test_prophet['ds'],
    forecast_prophet['yhat_lower'].values,
    forecast_prophet['yhat_upper'].values,
    alpha=0.15, color='mediumseagreen', label='Confidence Interval 95%'
)
ax.set_title('Prediksi IHSG - Prophet')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Close')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ══════════════════════════════════════════════════════════════
# PERBANDINGAN OVERLAY
# ══════════════════════════════════════════════════════════════
st.subheader("Perbandingan LSTM vs Prophet (Periode Test)")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(lstm_test_index,    lstm_actual,
        label='Aktual',  color='black',         linewidth=1)
ax.plot(lstm_test_index,    lstm_pred,
        label='LSTM',    color='darkorange',    linewidth=1.1, linestyle='--')
ax.plot(test_prophet['ds'], prophet_pred,
        label='Prophet', color='mediumseagreen',linewidth=1.1, linestyle='--')
ax.set_title('LSTM vs Prophet — Periode Test')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Close')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ══════════════════════════════════════════════════════════════
# TABEL RINGKASAN
# ══════════════════════════════════════════════════════════════
st.subheader("Tabel Ringkasan")

summary = pd.DataFrame({
    'Model': ['LSTM', 'Prophet'],
    'MAE'  : [round(mae_l, 2),  round(mae_p, 2)],
    'RMSE' : [round(rmse_l, 2), round(rmse_p, 2)],
    'MAPE' : [f"{mape_l:.2f}%", f"{mape_p:.2f}%"],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# FORECAST HARI KE DEPAN
# ══════════════════════════════════════════════════════════════
st.subheader("🔮 Forecast Hari ke Depan")
st.markdown("Prediksi harga IHSG untuk beberapa hari ke depan dari data terakhir yang tersedia.")

n_days = st.slider("Jumlah hari yang ingin diprediksi", min_value=1, max_value=7, value=7)

st.warning(
    "⚠️ **Catatan:** LSTM menggunakan prediksi *recursive* — output hari sebelumnya "
    "menjadi input hari berikutnya. Akurasi menurun seiring bertambahnya horizon prediksi. "
    "Hasil forecast ini bersifat ilustratif dan tidak dapat dijadikan rekomendasi investasi."
)

# ── Data historis ─────────────────────────────────────────────
close_recent = df['Close']['2020-01-01':].copy()
last_date    = close_recent.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

# ── Load model dan scaler via Google Drive ────────────────────
model_lstm_future, scaler_future, scaler_close_future = load_lstm_model_and_scalers()

LOOK_BACK = lstm_results.get('look_back', 120)
SEED_SIZE = LOOK_BACK + 30
FEATURES  = ['Close', 'MA_5', 'RSI_14', 'MACD']


# ── Helper: hitung indikator dari array Close ─────────────────
def compute_indicators_arr(close_arr: np.ndarray) -> np.ndarray:
    s      = pd.Series(close_arr)
    ma5    = s.rolling(5).mean().bfill()
    delta  = s.diff()
    gain   = delta.where(delta > 0, 0).rolling(14).mean().bfill()
    loss   = (-delta.where(delta < 0, 0)).rolling(14).mean().bfill()
    rsi    = (100 - (100 / (1 + gain / loss))).bfill()
    ema12  = s.ewm(span=12, adjust=False).mean()
    ema26  = s.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    return np.column_stack([s.values, ma5.values, rsi.values, macd.values])


# ── LSTM Recursive Forecast ───────────────────────────────────
running_close     = list(close_recent.values[-SEED_SIZE:])
lstm_future_preds = []

for _ in range(n_days):
    features_arr   = compute_indicators_arr(np.array(running_close))
    last_n_scaled  = scaler_future.transform(
        features_arr[-LOOK_BACK:].astype(np.float32)
    )
    x_input        = last_n_scaled.reshape(1, LOOK_BACK, len(FEATURES))
    pred_scaled    = float(model_lstm_future.predict(x_input, verbose=0)[0][0])
    pred_close     = float(scaler_close_future.inverse_transform([[pred_scaled]])[0][0])
    lstm_future_preds.append(pred_close)
    running_close.append(pred_close)

lstm_future_preds = np.array(lstm_future_preds)


# ── Helper: hitung indikator untuk Prophet ────────────────────
def compute_indicators_prophet(series: pd.Series) -> pd.DataFrame:
    df_ind           = pd.DataFrame({'Close': series})
    df_ind['MA_5']   = df_ind['Close'].rolling(window=5).mean()
    delta            = df_ind['Close'].diff()
    gain             = delta.where(delta > 0, 0).rolling(14).mean()
    loss             = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df_ind['RSI_14'] = 100 - (100 / (1 + gain / loss))
    ema_12           = df_ind['Close'].ewm(span=12, adjust=False).mean()
    ema_26           = df_ind['Close'].ewm(span=26, adjust=False).mean()
    df_ind['MACD']   = ema_12 - ema_26
    return df_ind.drop(columns='Close')


# ── Prophet Future Forecast ───────────────────────────────────
ind_hist  = compute_indicators_prophet(close_recent).dropna()
last_ind  = ind_hist.iloc[-1]

future_prophet_df = pd.DataFrame({
    'ds'     : future_dates,
    'MA_5'   : last_ind['MA_5'],
    'RSI_14' : last_ind['RSI_14'],
    'MACD'   : last_ind['MACD'],
})

forecast_future      = model_prophet.predict(future_prophet_df)
prophet_future_preds = forecast_future['yhat'].values


# ── Visualisasi ───────────────────────────────────────────────
history_tail = close_recent.iloc[-60:]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(history_tail.index, history_tail.values,
        label='Historis (60 hari terakhir)', color='steelblue', linewidth=1)
ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.plot(future_dates, lstm_future_preds,
        label=f'LSTM ({n_days} hari ke depan)',    color='darkorange',
        linewidth=1.3, linestyle='--', marker='o', markersize=4)
ax.plot(future_dates, prophet_future_preds,
        label=f'Prophet ({n_days} hari ke depan)', color='mediumseagreen',
        linewidth=1.3, linestyle='--', marker='s', markersize=4)
ax.fill_between(
    future_dates,
    forecast_future['yhat_lower'].values,
    forecast_future['yhat_upper'].values,
    alpha=0.12, color='mediumseagreen', label='CI 95% Prophet'
)
ax.set_title(f'Forecast IHSG — {n_days} Hari ke Depan')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Close')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)


# ── Tabel Prediksi ────────────────────────────────────────────
st.markdown("#### Tabel Prediksi")

future_table = pd.DataFrame({
    'Tanggal'           : future_dates.strftime('%Y-%m-%d'),
    'LSTM'              : lstm_future_preds.round(2),
    'Prophet'           : prophet_future_preds.round(2),
    'CI Lower (Prophet)': forecast_future['yhat_lower'].values.round(2),
    'CI Upper (Prophet)': forecast_future['yhat_upper'].values.round(2),
})
st.dataframe(future_table, use_container_width=True, hide_index=True)

st.caption(
    "LSTM menggunakan prediksi recursive — akurasi menurun seiring bertambahnya hari. "
    "Prophet memproyeksikan tren dan seasonality; indikator teknikal future menggunakan "
    "nilai terakhir dari data historis sebagai proxy."
)