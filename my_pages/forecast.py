import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def app():
    st.title("📈 Halaman Forecast")

    # Load model PSO-SVR
    model_path = "models/model_70.pkl"
    with open(model_path, 'rb') as file:
        svr_pso = pickle.load(file)

    # Cek apakah data sudah tersedia di session state
    if "preprocessed_data" not in st.session_state:
        st.warning("⚠️ Silakan upload data di halaman Upload Data dan lakukan preprocessing di halaman Preprocessing sebelum menjalankan forecasting.")
        return

    preprocessed_data = st.session_state["preprocessed_data"]
    all_data = st.session_state["uploaded_data"].copy()
    scaler_Y = preprocessed_data["scaler_Y"]

    if "X_scaled" not in preprocessed_data:
        st.error("❌ Data X_scaled tidak ditemukan! Pastikan preprocessing sudah dijalankan.")
        return

    X_scaled = preprocessed_data["X_scaled"]
    last_inputs = X_scaled.iloc[-1].values.reshape(1, -1)

    # **Pastikan data memiliki kolom tanggal yang benar**
    if 'Date' in all_data.columns:
        all_data['Date'] = pd.to_datetime(all_data['Date'])
        last_date = all_data['Date'].iloc[-1]
    else:
        all_data.index = pd.to_datetime(all_data.index)  # Konversi index ke datetime
        last_date = all_data.index[-1]

    # **Menentukan jumlah periode yang ingin diramalkan**
    n_forecast = 10
    forecast_results = []

    for _ in range(n_forecast):
        # Prediksi periode berikutnya
        next_forecast_scaled = svr_pso.predict(last_inputs)

        # Denormalisasi hasil prediksi
        next_forecast = scaler_Y.inverse_transform(next_forecast_scaled.reshape(-1, 1))

        # Simpan hasil prediksi
        forecast_results.append(next_forecast[0][0])

        # Perbarui input dengan memasukkan prediksi terbaru (geser ke belakang)
        last_inputs = np.roll(last_inputs, -1)
        last_inputs[0, -1] = next_forecast_scaled.item()

    # **Perbaiki tanggal agar benar**
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_forecast + 1)]

    # **Tampilkan hasil forecast**
    forecast_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Forecast': forecast_results
    })
    
    st.subheader("📊 Hasil Forecast 3 Hari ke Depan")
    st.dataframe(forecast_df)

    # **Visualisasi Forecast**
    st.subheader("📉 Grafik Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_data['Date'][-20:], all_data["Close"].values[-20:], label="Data Aktual", marker='o', linestyle='-')
    ax.plot(future_dates, forecast_results, label="Forecast", marker='x', linestyle='--', color="red")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Penutupan")
    ax.set_title("Peramalan Harga Penutupan Saham")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
