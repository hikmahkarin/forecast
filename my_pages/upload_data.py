import streamlit as st
import pandas as pd
import io

def app():
    st.title("📤 Upload Data CSV")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca data
        df = pd.read_csv(uploaded_file)

        # Simpan data ke dalam session state agar bisa digunakan di halaman lain
        st.session_state["uploaded_data"] = df

        st.success(f"✅ Data berhasil diunggah! (Jumlah Baris: {df.shape[0]}, Kolom: {df.shape[1]})")

        st.markdown("---")

        # Menampilkan preview data
        st.subheader("📌 Preview Data")
        st.dataframe(df.head())

        st.markdown("---")


        # Statistik dasar
        st.subheader("📈 Statistik Deskriptif")
        st.dataframe(df.describe())

