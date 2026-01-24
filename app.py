import os
import streamlit as st

from config import IMG_DIR


if __name__ == "__main__":
    st.set_page_config(page_title="Climate & Health Tracker", layout="centered", page_icon="🌤️")

    st.title("Climate & Health Tracker")
    st.header("Tổng quan")
    st.write(
        "Ứng dụng mẫu cho phân tích dữ liệu thời tiết và dự đoán liên quan tới sức khỏe. Sử dụng menu bên trái hoặc thư mục `pages/` để truy cập các trang chức năng."
    )