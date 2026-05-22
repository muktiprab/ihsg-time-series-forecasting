# 📈 IHSG Time Series Forecasting

> Comparative analysis and forecasting of JCI (IHSG) stock index using **Prophet** and **LSTM** deep learning models, complete with an interactive Streamlit dashboard.

🚀 **[Live Demo → ihsg-forecasting-mukti.streamlit.app](https://ihsg-forecasting-mukti.streamlit.app/)**

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Meta-0081FB?style=for-the-badge&logo=meta&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk membandingkan performa dua model forecasting pada data historis **Indeks Harga Saham Gabungan (IHSG)** periode **2020–2025**:

- **Prophet** — Model time series dari Meta yang menggunakan dekomposisi tren, seasonality, dan regressor eksternal (MA_5, RSI_14, MACD)
- **LSTM (Long Short-Term Memory)** — Model deep learning berbasis sequence dengan pendekatan multivariate (Close, MA_5, RSI_14, MACD)

Hasil perbandingan ditampilkan melalui dashboard interaktif berbasis **Streamlit** yang mencakup evaluasi model, visualisasi prediksi, dan forecast hari ke depan.

---

## 📂 Struktur Proyek

```
ihsg-dashboard/
│
├── app.py                  # Entry point Streamlit
├── requirements.txt        # Daftar dependensi
├── .gitignore
│
├── pages/
│   ├── 1_EDA.py            # Exploratory Data Analysis
│   └── 2_Prediksi.py       # Perbandingan & forecast LSTM vs Prophet
│
├── utils/
│   └── data_loader.py      # Helper load data, model, hasil
│
└── models/                 # ⚠️ Tidak di-upload (lihat setup di bawah)
    ├── lstm_model.keras
    ├── lstm_results.pkl
    ├── scaler.pkl
    ├── scaler_close.pkl
    ├── prophet_model.pkl
    └── prophet_data.pkl
```

---

## 📊 Fitur Dashboard

| Halaman | Konten |
|---|---|
| **EDA** | Visualisasi data historis IHSG, distribusi, tren, dan indikator teknikal |
| **Prediksi** | Evaluasi MAE/RMSE/MAPE, grafik prediksi LSTM & Prophet, perbandingan overlay |
| **Forecast** | Forecast N hari ke depan (max 7 hari) dengan tabel hasil prediksi |

---

## 🧪 Hasil Evaluasi Model

| Metrik | LSTM | Prophet |
|---|---|---|
| MAE | 141.99 | 77.58 |
| RMSE | 180.60 | 105.49 |
| MAPE | 1.96% | 1.10% |

Prophet mengungguli LSTM di semua metrik. Prophet mampu menangkap pola tren jangka panjang dan seasonality IHSG dengan lebih baik, sementara LSTM sebagai model sequence multivariate membutuhkan data yang lebih volatile untuk menunjukkan keunggulannya.

---

## ⚙️ Setup & Instalasi

### 1. Clone repository
```bash
git clone https://github.com/muktiprab/ihsg-time-series-forecasting.git
cd ihsg-time-series-forecasting
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
```

### 3. Download folder `models/`
Folder `models/` tidak di-include di repository karena ukuran file yang besar.
Download dari Google Drive berikut dan letakkan di root project:

> 📁 **[Download models/ dari Google Drive](https://drive.google.com/drive/folders/1UF8MPRkrKArnTAmkHvlouNsDI0LO54qN?usp=sharing)**

### 4. Jalankan dashboard
```bash
streamlit run app.py
```

---

## 📓 Notebook

Eksplorasi data dan training model dilakukan di Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muktiprab/ihsg-time-series-forecasting/blob/main/Explorasi_dataset_IHSG.ipynb)

---

## 👤 Author

**Mukti Prabowo**
[![GitHub](https://img.shields.io/badge/GitHub-muktiprab-181717?style=flat&logo=github)](https://github.com/muktiprab)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-muktiprabowo-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/muktiprabowo)
