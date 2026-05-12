import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_data

st.set_page_config(page_title="EDA - IHSG", page_icon="🔍", layout="wide")
st.title("🔍 Exploratory Data Analysis")

df, df2 = load_data()

# ── Info Dataset ─────────────────────────────────────────────
st.subheader("Info Dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data", f"{len(df):,} hari")
col2.metric("Periode", f"{df.index[0].year} – {df.index[-1].year}")
col3.metric("Close Tertinggi", f"{df['Close'].max():,.0f}")
col4.metric("Close Terendah", f"{df['Close'].min():,.0f}")

st.dataframe(df.describe().round(2), use_container_width=True)

st.divider()

# ── Harga Close Historis ─────────────────────────────────────
st.subheader("Harga Close IHSG")

year_range = st.slider(
    "Filter Tahun",
    min_value=int(df.index.year.min()),
    max_value=int(df.index.year.max()),
    value=(2020, int(df.index.year.max()))
)
df_filtered = df[str(year_range[0]):str(year_range[1])]
df2_filtered = df2[str(year_range[0]):str(year_range[1])]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df_filtered.index, df_filtered['Close'], color='steelblue', linewidth=0.9)
ax.set_title(f'Harga Close IHSG ({year_range[0]}–{year_range[1]})')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga Close')
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ── Indikator Teknikal ───────────────────────────────────────
st.subheader("Indikator Teknikal")
tab1, tab2, tab3, tab4 = st.tabs(["MA & EMA", "RSI", "MACD", "Stochastic"])

with tab1:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df2_filtered.index, df2_filtered['Close'],  label='Close',  linewidth=0.8, color='black')
    ax.plot(df2_filtered.index, df2_filtered['MA_5'],   label='MA 5',   linewidth=1.0, color='steelblue')
    ax.plot(df2_filtered.index, df2_filtered['MA_10'],  label='MA 10',  linewidth=1.0, color='orange')
    ax.plot(df2_filtered.index, df2_filtered['EMA_5'],  label='EMA 5',  linewidth=1.0, color='green', linestyle='--')
    ax.plot(df2_filtered.index, df2_filtered['EMA_10'], label='EMA 10', linewidth=1.0, color='red',   linestyle='--')
    ax.set_title('Moving Average & EMA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df2_filtered.index, df2_filtered['RSI_14'], color='purple', linewidth=0.9)
    ax.axhline(70, linestyle='--', color='red',   alpha=0.6, label='Overbought (70)')
    ax.axhline(30, linestyle='--', color='green', alpha=0.6, label='Oversold (30)')
    ax.set_title('RSI 14')
    ax.set_ylabel('RSI')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    ax1.plot(df2_filtered.index, df2_filtered['Close'], color='black', linewidth=0.8)
    ax1.set_title('Harga Close')
    ax1.grid(True, alpha=0.3)

    ax2.plot(df2_filtered.index, df2_filtered['MACD'],        label='MACD',   color='steelblue')
    ax2.plot(df2_filtered.index, df2_filtered['MACD_signal'], label='Signal', color='orange')
    ax2.bar(df2_filtered.index,  df2_filtered['MACD_hist'],
            color=['green' if v >= 0 else 'red' for v in df2_filtered['MACD_hist']],
            alpha=0.4, label='Histogram')
    ax2.set_title('MACD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df2_filtered.index, df2_filtered['Stochastic_%K'], label='%K', color='steelblue')
    ax.plot(df2_filtered.index, df2_filtered['Stochastic_%D'], label='%D', color='orange')
    ax.axhline(80, linestyle='--', color='red',   alpha=0.6, label='Overbought (80)')
    ax.axhline(20, linestyle='--', color='green', alpha=0.6, label='Oversold (20)')
    ax.set_title('Stochastic Oscillator')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Volatility & Return ──────────────────────────────────────
st.subheader("Volatility & Return")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df2_filtered.index, df2_filtered['Volatility_pct'], color='tomato', linewidth=0.8)
    ax.set_title('Daily Volatility (%)')
    ax.set_ylabel('Volatility (%)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df2_filtered.index, df2_filtered['Return'], color='mediumseagreen', linewidth=0.8)
    ax.set_title('Daily Return (%)')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(df2['Return'].dropna(), bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
ax.set_title('Distribusi Return Harian')
ax.set_xlabel('Return (%)')
ax.set_ylabel('Frekuensi')
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ── Correlation Heatmap ──────────────────────────────────────
st.subheader("Correlation Heatmap")

features = ['Open','High','Low','Close','Volatility_pct','Return',
            'MA_5','MA_10','EMA_5','EMA_10','RSI_14',
            'MACD','MACD_signal','MACD_hist','Stochastic_%K','Stochastic_%D']

corr = df2[features].corr()
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, linewidths=0.3)
ax.set_title('Feature Correlation Heatmap')
fig.tight_layout()
st.pyplot(fig)
