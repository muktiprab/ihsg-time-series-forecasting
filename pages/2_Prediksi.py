import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data, load_lstm_results, load_prophet_model, load_lstm_model_and_scaler

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

# ── Evaluasi ─────────────────────────────────────────────────
def evaluate(actual, pred):
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape

mae_l, rmse_l, mape_l = evaluate(lstm_actual, lstm_pred)
mae_p, rmse_p, mape_p = evaluate(prophet_actual, prophet_pred)

# ── Metrik ───────────────────────────────────────────────────
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
    m1.metric("MAE",  f"{mae_p:.2f}", delta=f"{mae_l - mae_p:.2f}",   delta_color="normal")
    m2.metric("RMSE", f"{rmse_p:.2f}", delta=f"{rmse_l - rmse_p:.2f}", delta_color="normal")
    m3.metric("MAPE", f"{mape_p:.2f}%", delta=f"{mape_l - mape_p:.2f}%", delta_color="normal")

st.caption("Delta pada Prophet menunjukkan selisih terhadap LSTM. Nilai positif = Prophet lebih baik.")

st.divider()

# ── Visualisasi LSTM ─────────────────────────────────────────
st.subheader("Prediksi LSTM")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_index, train_actual,
        label='Train', color='steelblue', linewidth=0.8)
ax.plot(lstm_test_index, lstm_actual,
        label='Aktual (Test)', color='black', linewidth=1)
ax.plot(lstm_test_index, lstm_pred,
        label='Prediksi LSTM', color='darkorange', linewidth=1.2, linestyle='--')
ax.set_title('Prediksi IHSG - LSTM')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Close')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ── Visualisasi Prophet ──────────────────────────────────────
st.subheader("Prediksi Prophet")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_prophet['ds'], train_prophet['y'],
        label='Train', color='steelblue', linewidth=0.8)
ax.plot(test_prophet['ds'], prophet_actual,
        label='Aktual (Test)', color='black', linewidth=1)
ax.plot(test_prophet['ds'], prophet_pred,
        label='Prediksi Prophet', color='mediumseagreen', linewidth=1.2, linestyle='--')
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

# ── Perbandingan Overlay ─────────────────────────────────────
st.subheader("Perbandingan LSTM vs Prophet (Periode Test)")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(lstm_test_index, lstm_actual,
        label='Aktual', color='black', linewidth=1)
ax.plot(lstm_test_index, lstm_pred,
        label='LSTM', color='darkorange', linewidth=1.1, linestyle='--')
ax.plot(test_prophet['ds'], prophet_pred,
        label='Prophet', color='mediumseagreen', linewidth=1.1, linestyle='--')
ax.set_title('LSTM vs Prophet — Periode Test')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Close')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ── Tabel Ringkasan ──────────────────────────────────────────
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
# FORECAST FUTURE
# ══════════════════════════════════════════════════════════════
st.subheader("🔮 Forecast Hari ke Depan")
st.markdown("Prediksi harga IHSG untuk beberapa hari ke depan dari data terakhir yang tersedia.")

n_days = st.slider("Jumlah hari yang ingin diprediksi", min_value=1, max_value=30, value=7)

# ── Data historis lengkap ─────────────────────────────────────
close_recent = df['Close']['2020-01-01':].copy()
last_date    = close_recent.index[-1]

# Buat tanggal hari kerja ke depan (skip Sabtu & Minggu)
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

# ── LSTM Recursive Forecast ───────────────────────────────────
LOOK_BACK = lstm_results.get('look_back', 60)
model_lstm_future, scaler_future = load_lstm_model_and_scaler()

# Seed: 60 hari terakhir dari data historis
seed_prices  = close_recent.values[-LOOK_BACK:].reshape(-1, 1)
seed_scaled  = scaler_future.transform(seed_prices)
current_seq  = seed_scaled.flatten().tolist()

lstm_future_preds = []
for _ in range(n_days):
    x_input = np.array(current_seq[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
    pred_scaled = model_lstm_future.predict(x_input, verbose=0)[0][0]
    lstm_future_preds.append(pred_scaled)
    current_seq.append(pred_scaled)

lstm_future_preds = scaler_future.inverse_transform(
    np.array(lstm_future_preds).reshape(-1, 1)
).flatten()

# ── Prophet Future Forecast ───────────────────────────────────
future_prophet_df = pd.DataFrame({'ds': future_dates})
forecast_future   = model_prophet.predict(future_prophet_df)
prophet_future_preds = forecast_future['yhat'].values

# ── Visualisasi Future ────────────────────────────────────────
# Tampilkan 60 hari historis terakhir sebagai konteks
history_tail  = close_recent.iloc[-60:]

fig, ax = plt.subplots(figsize=(14, 4))

# Historis
ax.plot(history_tail.index, history_tail.values,
        label='Historis (60 hari terakhir)', color='steelblue', linewidth=1)

# Garis pemisah
ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1, alpha=0.7)

# LSTM future
ax.plot(future_dates, lstm_future_preds,
        label=f'LSTM ({n_days} hari ke depan)', color='darkorange',
        linewidth=1.3, linestyle='--', marker='o', markersize=4)

# Prophet future
ax.plot(future_dates, prophet_future_preds,
        label=f'Prophet ({n_days} hari ke depan)', color='mediumseagreen',
        linewidth=1.3, linestyle='--', marker='s', markersize=4)

# Confidence interval Prophet
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

# ── Tabel Hasil Forecast ──────────────────────────────────────
st.markdown("#### Tabel Prediksi")

future_table = pd.DataFrame({
    'Tanggal'       : future_dates.strftime('%Y-%m-%d'),
    'LSTM'          : lstm_future_preds.round(2),
    'Prophet'       : prophet_future_preds.round(2),
    'CI Lower (Prophet)': forecast_future['yhat_lower'].values.round(2),
    'CI Upper (Prophet)': forecast_future['yhat_upper'].values.round(2),
})
st.dataframe(future_table, use_container_width=True, hide_index=True)

st.caption(
    "⚠️ LSTM menggunakan prediksi recursive — akurasi menurun seiring bertambahnya hari. "
    "Prophet hanya memproyeksikan tren dan seasonality tanpa memperhitungkan kondisi pasar terkini."
)