import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config import TRAIN_PATH
from datamodules.DataModule import DataModule

st.title("Huấn luyện & Thẩm định / Dự đoán")

dm = DataModule(TRAIN_PATH)
df = dm.df

st.subheader("Chuẩn bị dữ liệu mẫu")
if df.empty:
    st.error("Không tìm thấy dữ liệu ở TRAIN_PATH.")
else:
    # simple feature set similar to notebook
    df['last_updated'] = df.get('last_updated')
    try:
        df['hour'] = np.where(df['last_updated'].notnull(), pd.to_datetime(df['last_updated']).dt.hour, 0)
    except Exception:
        df['hour'] = 0

    features = [c for c in ['latitude', 'humidity', 'pressure_mb', 'wind_kph', 'cloud', 'hour'] if c in df.columns]
    target = 'temperature_celsius' if 'temperature_celsius' in df.columns else None

    if not features or target is None:
        st.info("Không đủ trường để huấn luyện. Cần các cột: latitude, humidity, pressure_mb, wind_kph, cloud, temperature_celsius")
    else:
        X = df[features].dropna()
        y = df.loc[X.index, target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if 'model' not in st.session_state:
            st.session_state['model'] = None
            st.session_state['trained'] = False

        if st.button("Huấn luyện (RandomForest)"):
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.session_state['trained'] = True
            st.success("Huấn luyện thành công.")

        if st.button("Thẩm định (tập test)"):
            if not st.session_state.get('trained', False):
                st.warning("Hãy huấn luyện mô hình trước.")
            else:
                model = st.session_state['model']
                preds = model.predict(X_test)
                rmse = mean_squared_error(y_test, preds, squared=False)
                r2 = r2_score(y_test, preds)
                st.metric("RMSE", f"{rmse:.3f}")
                st.metric("R2", f"{r2:.3f}")
                st.pyplot()

        st.markdown("---")
        st.subheader("Dự đoán nhanh (nhập tay các phép đo)")
        inputs = {}
        for f in features:
            inputs[f] = st.number_input(f, value=float(X[f].median()))
        if st.button("Dự đoán"):
            if not st.session_state.get('trained', False):
                st.warning("Hãy huấn luyện mô hình trước.")
            else:
                model = st.session_state['model']
                arr = np.array([inputs[f] for f in features]).reshape(1, -1)
                pred = model.predict(arr)[0]
                st.success(f"Dự đoán nhiệt độ: {pred:.2f} °C")
