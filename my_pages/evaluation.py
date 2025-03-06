import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

def app():
    st.title("📈 Evaluasi Model")
    st.write("Grafik perbandingan antara data aktual dan hasil prediksi serta nilai MAPE data testing.")

    # Load model yang sudah disimpan
    model_path = "model_70.pkl"
    scaler_path = "scaler.pkl"
    test_data_path = "testing_70.pkl"

    try:
        with open(model_path, 'rb') as file:
            svr = pickle.load(file)

        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)

        with open(test_data_path, 'rb') as file:
            test, test_label = pickle.load(file)

        # Melakukan prediksi pada data testing
        pred_test = svr.predict(test)
        pred_test1 = pred_test.reshape(-1, 1)

        # Denormalisasi data
        denormalized_actual = pd.DataFrame(scaler.inverse_transform(test_label.reshape(-1, 1)), columns=["Actual"])
        denormalized_predicted = pd.DataFrame(scaler.inverse_transform(pred_test1), columns=["Predicted"])

        # Hitung MAPE
        mape_score = mean_absolute_percentage_error(denormalized_actual, denormalized_predicted)

        # Menampilkan MAPE
        st.write(f"🔢 **MAPE (Mean Absolute Percentage Error) pada Data Testing:** `{mape_score:.4f}`")

        # Menampilkan grafik perbandingan
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(denormalized_actual.values, label="Actual Data", color="blue", marker="o", linestyle="-")
        ax.plot(denormalized_predicted.values, label="Predicted Data", color="red", marker="x", linestyle="--")
        ax.set_title("Grafik Perbandingan Data Aktual vs Prediksi")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        # Tampilkan plot di Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memuat model atau data: {e}")

