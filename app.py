import streamlit as st

st.set_page_config(
    page_title="IHSG Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 IHSG Forecasting Dashboard")
st.markdown("""
Selamat datang di dashboard analisis dan prediksi **Indeks Harga Saham Gabungan (IHSG)**.

Dashboard ini terdiri dari dua halaman utama:

- **EDA** — Eksplorasi data historis IHSG, indikator teknikal, dan distribusi return
- **Prediksi** — Perbandingan hasil forecasting menggunakan LSTM dan Prophet

Gunakan menu di sidebar kiri untuk berpindah halaman.
""")

st.info("Data bersumber dari dataset publik IHSG harian (1995–2025). Model dilatih menggunakan data periode 2020–2025.")

@st.cache_resource
def load_lstm_results():
    with open('models/lstm_results.pkl', 'rb') as f:
        return pickle.load(f)