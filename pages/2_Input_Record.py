import streamlit as st
from datetime import datetime
import pandas as pd

from config import TRAIN_PATH
from datamodules.DataModule import DataModule

st.title("Thêm / Cập nhật dữ liệu")

weather = DataModule(TRAIN_PATH)

st.write("Hiện có:", len(weather.df), "bản ghi")

with st.form("add_row"):
    st.subheader("Thêm một hàng mới vào GlobalWeatherRepository.csv")
    lat = st.number_input("Latitude", value=0.0, format="%.4f")
    lon = st.number_input("Longitude", value=0.0, format="%.4f")
    temp = st.number_input("Temperature (C)", value=20.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    pressure = st.number_input("Pressure (mb)", value=1013.0)
    cloud = st.number_input("Cloud (%)", value=0)
    submit = st.form_submit_button("Thêm vào CSV")
    if submit:
        row = {
            'latitude': lat,
            'longitude': lon,
            'temperature_celsius': temp,
            'humidity': humidity,
            'pressure_mb': pressure,
            'cloud': cloud,
            'last_updated': datetime.now().isoformat()
        }
        ok = weather.append_row(row)
        if ok:
            st.success("Đã thêm và lưu vào CSV.")
        else:
            st.error("Không thể lưu. Kiểm tra quyền ghi file.")

st.write("Preview cuối cùng:")
st.write(weather.tail(5))
