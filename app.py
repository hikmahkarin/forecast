import streamlit as st
from streamlit_option_menu import option_menu
from my_pages import home, upload_data, preprocessing, modelling, forecast

# Menampilkan logo di sidebar
st.sidebar.image("logo2.png", use_column_width=True)

# Sidebar option menu di bagian kanan layar
with st.sidebar:
    selected = option_menu(
        "MENU", ["🏠 Home", "📤 Upload Data", "⚙️ Preprocessing", "📊 Modelling", "📉 Forecast"],
        icons=["house", "cloud-upload", "gear", "bar-chart", "activity", "bar-chart"],
        menu_icon="list", default_index=0,
        orientation="vertical"
    )

# Menjalankan aplikasi yang dipilih
pages = {
    "🏠 Home": home.app,
    "📤 Upload Data": upload_data.app,
    "⚙️ Preprocessing": preprocessing.app,
    "📊 Modelling": modelling.app,
    "📉 Forecast": forecast.app,
}

if selected in pages:
    pages[selected]()
