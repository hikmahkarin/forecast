import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Fungsi untuk mengubah data menjadi format supervised learning
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        X.append(sequence[i:end_ix])
        y.append(sequence[end_ix])
    return np.array(X), np.array(y)

def app():
    st.title("⚙️ Preprocessing Data")
    st.markdown(
        """
        ### 🔹 Tahapan Preprocessing:
        - 🔄 **Transformasi ke format supervised learning**
        - ✂️ **Split Data ke Training dan Testing tanpa acak**
        - 📏 **Normalisasi fitur menggunakan StandardScaler**
        """
    )
    st.markdown("---")

    if "uploaded_data" not in st.session_state:
        st.warning("⚠️ Harap unggah data terlebih dahulu pada menu **Upload Data**.")
        return

    data = st.session_state["uploaded_data"].copy()

    if "Close" not in data.columns:
        st.error("⚠️ Kolom 'Close' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom tersebut.")
        return

    # st.subheader("📊 Imputasi Missing Values")
    # numeric_cols = data.select_dtypes(include=[np.number]).columns
    # imputer = SimpleImputer(strategy='mean')
    # data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    # st.dataframe(data.head())

    closing_prices = data['Close'].values

    st.subheader("📈 Transformasi ke Supervised Learning Format")
    n_steps = 4
    X, y = split_sequence(closing_prices, n_steps)
    df_transformasi = pd.DataFrame(X, columns=[f"t-{i}" for i in range(n_steps, 0, -1)])
    df_transformasi['Xt'] = y
    st.dataframe(df_transformasi.head())

    st.subheader("📏 Hasil Normalisasi Data")
    scaler_X, scaler_Y = StandardScaler(), StandardScaler()
    X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_Y.fit_transform(y.reshape(-1, 1)).flatten()
    df_normalisasi = pd.DataFrame(X_scaled, columns=[f"t-{i}" for i in range(n_steps, 0, -1)])
    df_normalisasi['Xt'] = y_scaled
    st.dataframe(df_normalisasi.head())

    split_index = int(0.7 * len(X_scaled))
    X_train_scaled, X_test_scaled = X_scaled[:split_index], X_scaled[split_index:]
    y_train_scaled, y_test_scaled = y_scaled[:split_index], y_scaled[split_index:]

    with open("models/scaler_X.pkl", "wb") as file:
        pickle.dump(scaler_X, file)
    with open("models/scaler_Y.pkl", "wb") as file:
        pickle.dump(scaler_Y, file)

    st.session_state["preprocessed_data"] = {
    "X_scaled": pd.DataFrame(X_scaled, columns=[f"t-{i}" for i in range(n_steps, 0, -1)]),  # Simpan X_scaled!
    "normalized_X_train": X_train_scaled,
    "normalized_X_test": X_test_scaled,
    "normalized_y_train": y_train_scaled,
    "normalized_y_test": y_test_scaled,
    "scaler_X": scaler_X,
    "scaler_Y": scaler_Y,
    "n_steps": n_steps
}


    st.success("✅ Preprocessing selesai! Data siap digunakan untuk pemodelan.")