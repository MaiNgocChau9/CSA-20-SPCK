import streamlit as st
from config import TRAIN_PATH, HEALTH_PATH, SEATTLE_PATH
from datamodules.DataModule import DataModule

st.set_page_config(layout="wide")

st.title("Phân tích dữ liệu")

st.header("Tải và xem nhanh các tập dữ liệu có sẵn")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Global Weather Repository")
    weather = DataModule(TRAIN_PATH)
    st.write(weather.head())
    fig = weather.visualize_dist()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Global Climate Health Impact (2015-2025)")
    health = DataModule(HEALTH_PATH)
    st.write(health.head())

st.header("Phân tích tương quan")
fig = weather.visualize_corr()
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Không đủ trường số để hiển thị ma trận tương quan.")

st.header("Seattle sample")
seattle = DataModule(SEATTLE_PATH)
st.write(seattle.head())
