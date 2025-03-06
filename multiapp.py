import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from multiapp import MultiApp
from my_pages import home, upload_data, preprocessing, modelling, forecast

# Menampilkan logo di sidebar
st.sidebar.image("logo2.png", use_column_width=True)

# Membuat objek MultiApp
app = MultiApp()

# Menambahkan halaman aplikasi
pages = {
    "🏠 Home": home.app,
    "📤 Upload Data": upload_data.app,
    "⚙️ Preprocessing": preprocessing.app,
    "📊 Modelling": modelling.app,
    "📉 Forecast": forecast.app,
}

# Sidebar option menu
selected = option_menu(
    "MENU", list(pages.keys()),
    icons=["house", "cloud-upload", "gear", "bar-chart", "activity", "bar-chart"],
    menu_icon="list", default_index=0,
    orientation="vertical"
)

# Menjalankan aplikasi yang dipilih
pages[selected]()
