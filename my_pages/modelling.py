import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def app():
    st.title("📊 Halaman Modelling")
    
    # Input batas parameter SVR
    st.sidebar.header("🔧 Pengaturan Batas Parameter SVR")
    C_min = st.sidebar.number_input("Batas Bawah C", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
    C_max = st.sidebar.number_input("Batas Atas C", min_value=0.1, max_value=1000.0, value=200.0, step=0.1)
    epsilon_min = st.sidebar.number_input("Batas Bawah Epsilon", min_value=0.00001, max_value=10.0, value=0.001, step=0.001)
    epsilon_max = st.sidebar.number_input("Batas Atas Epsilon", min_value=0.0001, max_value=10.0, value=0.1, step=0.001)
    gamma_min = st.sidebar.number_input("Batas Bawah Gamma", min_value=0.00001, max_value=10.0, value=0.00001, step=0.0001)
    gamma_max = st.sidebar.number_input("Batas Atas Gamma", min_value=0.00001, max_value=10.0, value=0.001, step=0.0001)
    
    # Tombol untuk menerapkan parameter
    if st.sidebar.button("Terapkan Parameter"):
        st.session_state['svr_params'] = {
            'C': C_max,
            'epsilon': epsilon_min,  # Menggunakan epsilon_min agar sesuai dengan Colab
            'gamma': gamma_max
        }
    
    # Load model PSO-SVR
    model_path = "models/model_70.pkl"
    with open(model_path, 'rb') as file:
        svr_pso = pickle.load(file)
    
    # Cek apakah data sudah tersedia di session state
    if "preprocessed_data" not in st.session_state:
        st.warning("⚠️ Silakan upload data di halaman **Upload Data** dan lakukan preprocessing di halaman **Preprocessing** sebelum menjalankan modelling.")
        return
    
    preprocessed_data = st.session_state["preprocessed_data"]
    all_data = st.session_state["uploaded_data"].copy()  
    scaler_Y = preprocessed_data["scaler_Y"]
    n_steps = preprocessed_data["n_steps"]

    # Gunakan harga penutupan asli (bukan Xt!)
    actual_prices = all_data["Close"].values.reshape(-1, 1)
    closing_prices_scaled = scaler_Y.transform(actual_prices)

    # Update parameter SVR jika sudah ada di session state
    if 'svr_params' in st.session_state:
        params = st.session_state['svr_params']
        
        # Buat ulang model dengan parameter baru
        svr_pso = SVR(kernel="rbf", C=params['C'], epsilon=params['epsilon'], gamma=params['gamma'])
        
        # Latih ulang model dengan data training yang sudah diproses sebelumnya
        svr_pso.fit(preprocessed_data["normalized_X_train"], preprocessed_data["normalized_y_train"])

    # Prediksi historis dengan sliding window berdasarkan data aktual
    historical_predictions_scaled = [
        svr_pso.predict(closing_prices_scaled[i:i + n_steps].reshape(1, -1))[0]
        for i in range(len(closing_prices_scaled) - n_steps)
    ]

    # Denormalisasi hasil prediksi
    historical_predictions = scaler_Y.inverse_transform(np.array(historical_predictions_scaled).reshape(-1, 1)).flatten()
    actual_values = actual_prices[n_steps:].flatten()

    # Hitung MAPE
    # mape_historis = mean_absolute_percentage_error(actual_values, historical_predictions)
    mape_train = mean_absolute_percentage_error(
        scaler_Y.inverse_transform(preprocessed_data["normalized_y_train"].reshape(-1, 1)).flatten(),
        scaler_Y.inverse_transform(svr_pso.predict(preprocessed_data["normalized_X_train"]).reshape(-1, 1)).flatten()
    )
    mape_test = mean_absolute_percentage_error(
        scaler_Y.inverse_transform(preprocessed_data["normalized_y_test"].reshape(-1, 1)).flatten(),
        scaler_Y.inverse_transform(svr_pso.predict(preprocessed_data["normalized_X_test"]).reshape(-1, 1)).flatten()
    )

    # Ambil parameter terbaik dari model
    best_parameters = svr_pso.get_params()

    st.markdown("---")
    
    st.subheader("📉 Hasil MAPE")
    col1, col2 = st.columns(2)
    # col1.metric(label="MAPE Historis", value=f"{mape_historis:.6f}")
    col1.metric(label="MAPE Training", value=f"{mape_train:.6f}")
    col2.metric(label="MAPE Testing", value=f"{mape_test:.6f}")

    st.markdown("---")

    st.subheader("⚙️ Best Parameters dari Model SVR-PSO")
    st.json(best_parameters)

    st.markdown("---")

    # Buat DataFrame untuk perbandingan
    comparison_df = pd.DataFrame({
        'Tanggal': all_data.index[n_steps:],
        'Aktual': actual_values,
        'Prediksi': historical_predictions,
        'Perbedaan': actual_values - historical_predictions
    }).sort_values(by='Tanggal').reset_index(drop=True)

    # Simpan hasil denormalisasi ke dalam Pickle
    # with open("models/denormalized_results.pkl", "wb") as file:
    #     pickle.dump((preprocessed_data["normalized_y_test"], svr_pso.predict(preprocessed_data["normalized_X_test"])), file)

    # st.success("✅ Hasil denormalisasi telah disimpan dalam `denormalized_results.pkl`")

    st.markdown("---")

    # Tampilkan tabel perbandingan
    st.subheader("📋 Tabel Perbandingan Data Aktual vs Prediksi")
    st.dataframe(comparison_df)

    st.markdown("---")

    # Visualisasi perbandingan data aktual vs prediksi
    st.subheader("📊 Grafik Perbandingan Data Aktual vs Prediksi")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(comparison_df['Tanggal'], comparison_df['Aktual'], label="Data Aktual", marker='o', linestyle='-')
    ax.plot(comparison_df['Tanggal'], comparison_df['Prediksi'], label="Prediksi", marker='x', linestyle='--')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Penutupan")
    ax.set_title("Perbandingan Data Aktual vs Prediksi")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
