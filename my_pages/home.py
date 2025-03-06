import streamlit as st

def app():
    st.title("🏠 Selamat Datang di Aplikasi Peramalan")
    
    st.markdown(
        """
        ### 📌 Tentang Aplikasi
        Aplikasi ini digunakan untuk peramalan data menggunakan **Support Vector Regression (SVR) + PSO**.
        
        #### 🔹 Fitur Utama:
        - **📤 Upload Data**  
          Mengunggah file CSV yang akan digunakan untuk analisis.
        - **⚙️ Preprocessing**  
          Menampilkan hasil preprocessing (handling missing values & normalisasi).
        - **📊 Modelling**  
          Melakukan training model menggunakan SVR dengan optimasi PSO.
        - **📉 Forecast**  
          Menampilkan hasil prediksi peramalan.
        - **📈 Evaluation**  
          Grafik perbandingan antara data aktual dan hasil prediksi.
        """,
        unsafe_allow_html=True,
    )
    
    st.sidebar.success("🔍 Pilih halaman dari menu navigasi di sidebar.")
