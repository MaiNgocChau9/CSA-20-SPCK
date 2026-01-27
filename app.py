"""
Ứng dụng Phân tích Khí hậu và Sức khỏe
Dự đoán tác động của khí hậu đến sức khỏe con người
Update: Thêm Tab giải thích và Xu hướng nhiệt độ theo năm
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# Cấu hình scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="Phân tích Khí hậu & Sức khỏe",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== HÀM TIỆN ÍCH =====
@st.cache_data
def load_health_data():
    """Tải dữ liệu khí hậu và sức khỏe"""
    try:
        df = pd.read_csv('data/global_climate_health_impact_tracker_2015_2025.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file dữ liệu sức khỏe (global_climate_health_impact_tracker_2015_2025.csv)!")
        return None

@st.cache_data
def load_weather_data():
    """Tải dữ liệu thời tiết toàn cầu"""
    try:
        df = pd.read_csv('data/GlobalWeatherRepository.csv')
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        return df
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file dữ liệu thời tiết (GlobalWeatherRepository.csv)!")
        return None

@st.cache_data
def analyze_correlations(df):
    """Phân tích tương quan chi tiết giữa các biến"""
    correlations = {}
    
    # Tương quan PM2.5 với bệnh hô hấp
    correlations['pm25_respiratory'] = df[['pm25_ugm3', 'respiratory_disease_rate']].corr().iloc[0, 1]
    
    # Tương quan nhiệt độ với bệnh lây truyền qua sinh vật trung gian
    correlations['temp_vector'] = df[['temperature_celsius', 'vector_disease_risk_score']].corr().iloc[0, 1]
    
    # Tương quan nắng nóng với ca nhập viện
    correlations['heat_admission'] = df[['heat_wave_days', 'heat_related_admissions']].corr().iloc[0, 1]
    
    # Tương quan chất lượng không khí với sức khỏe tim mạch
    correlations['aqi_cardio'] = df[['air_quality_index', 'cardio_mortality_rate']].corr().iloc[0, 1]
    
    return correlations

@st.cache_data
def generate_research_findings(df):
    """Tạo các phát hiện nghiên cứu từ dữ liệu"""
    findings = []
    
    # 1. Phân tích PM2.5 và bệnh hô hấp
    pm25_high = df[df['pm25_ugm3'] > 50]
    pm25_low = df[df['pm25_ugm3'] <= 50]
    resp_diff = pm25_high['respiratory_disease_rate'].mean() - pm25_low['respiratory_disease_rate'].mean()
    
    findings.append({
        'Danh mục': 'Chất lượng Không khí',
        'Phát hiện': f'Tỷ lệ bệnh hô hấp cao hơn {resp_diff:.1f}% khi PM2.5 > 50 μg/m³',
        'Tác động': 'Cao' if resp_diff > 10 else 'Trung bình',
        'Số mẫu': len(pm25_high),
        'Độ tin cậy': 'Cao'
    })
    
    # 2. Phân tích nhiệt độ và bệnh lây truyền qua sinh vật trung gian
    temp_high = df[df['temperature_celsius'] > 25]
    vector_high = temp_high['vector_disease_risk_score'].mean()
    vector_low = df[df['temperature_celsius'] <= 25]['vector_disease_risk_score'].mean()
    vector_diff = vector_high - vector_low
    
    findings.append({
        'Danh mục': 'Nhiệt độ & Sinh vật trung gian',
        'Phát hiện': f'Rủi ro bệnh tăng {vector_diff:.1f} điểm khi nhiệt độ > 25°C',
        'Tác động': 'Cao' if vector_diff > 1 else 'Trung bình',
        'Số mẫu': len(temp_high),
        'Độ tin cậy': 'Cao'
    })
    
    # 3. Phân tích nắng nóng và ca nhập viện
    heat_wave = df[df['heat_wave_days'] > 0]
    admission_ratio = heat_wave['heat_related_admissions'].mean() / df['heat_related_admissions'].mean()
    
    findings.append({
        'Danh mục': 'Nắng nóng',
        'Phát hiện': f'Ca nhập viện tăng {(admission_ratio - 1) * 100:.1f}% trong đợt nắng nóng',
        'Tác động': 'Rất cao' if admission_ratio > 2 else 'Cao',
        'Số mẫu': len(heat_wave),
        'Độ tin cậy': 'Cao'
    })
    
    # 4. Phân tích thời tiết cực đoan
    extreme = df[df['extreme_weather_events'] > 0]
    health_impact = extreme[['respiratory_disease_rate', 'cardio_mortality_rate', 
                            'vector_disease_risk_score']].mean().mean()
    normal_health = df[df['extreme_weather_events'] == 0][
        ['respiratory_disease_rate', 'cardio_mortality_rate', 'vector_disease_risk_score']
    ].mean().mean()
    
    findings.append({
        'Danh mục': 'Thời tiết Cực đoan',
        'Phát hiện': f'Tác động sức khỏe tổng thể tăng {((health_impact/normal_health - 1) * 100):.1f}%',
        'Tác động': 'Rất cao',
        'Số mẫu': len(extreme),
        'Độ tin cậy': 'Cao'
    })
    
    return pd.DataFrame(findings)

@st.cache_data
def calculate_model_metrics(df):
    """Tính toán các chỉ số cho từng mô hình"""
    metrics = []
    
    # Model 1: Respiratory Disease
    X = df[['pm25_ugm3', 'air_quality_index']].dropna()
    y = df.loc[X.index, 'respiratory_disease_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics.append({
        'Mô hình': 'Bệnh Hô hấp',
        'Thuật toán': 'Linear Regression',
        'Đặc trưng': 'PM2.5, AQI',
        'R² Score': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Số mẫu': len(X)
    })
    
    # Model 2: Vector Disease
    X = df[['temperature_celsius', 'precipitation_mm', 'heat_related_admissions']].dropna()
    y = df.loc[X.index, 'vector_disease_risk_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics.append({
        'Mô hình': 'Bệnh lây truyền qua sinh vật trung gian',
        'Thuật toán': 'Random Forest',
        'Đặc trưng': 'Nhiệt độ, Mưa, Ca nhập viện',
        'R² Score': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Số mẫu': len(X)
    })
    
    # Model 3: Heat-related Admissions
    X = df[['temperature_celsius', 'precipitation_mm', 'heat_wave_days', 'extreme_weather_events']].dropna()
    y = df.loc[X.index, 'heat_related_admissions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics.append({
        'Mô hình': 'Ca Nhập viện do Nắng',
        'Thuật toán': 'Linear Regression',
        'Đặc trưng': 'Nhiệt độ, Mưa, Nắng nóng, Cực đoan',
        'R² Score': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Số mẫu': len(X)
    })
    
    return pd.DataFrame(metrics)

def train_respiratory_model(df):
    """Huấn luyện mô hình dự đoán bệnh hô hấp"""
    features = ['pm25_ugm3', 'air_quality_index']
    X = df[features].dropna()
    y = df.loc[X.index, 'respiratory_disease_rate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X_test, y_test, y_pred

def train_vector_disease_model(df):
    """Huấn luyện mô hình dự đoán bệnh lây truyền qua sinh vật trung gian"""
    features = ['temperature_celsius', 'precipitation_mm', 'heat_related_admissions']
    X = df[features].dropna()
    y = df.loc[X.index, 'vector_disease_risk_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X_test, y_test, y_pred

def train_heat_admission_model(df):
    """Huấn luyện mô hình dự đoán ca nhập viện do nắng nóng"""
    features = ['temperature_celsius', 'precipitation_mm', 'heat_wave_days', 'extreme_weather_events']
    X = df[features].dropna()
    y = df.loc[X.index, 'heat_related_admissions']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X_test, y_test, y_pred

def train_temperature_model(df):
    """Huấn luyện mô hình dự đoán nhiệt độ từ dữ liệu thời tiết"""
    df['hour'] = df['last_updated'].dt.hour
    features = ['latitude', 'humidity', 'pressure_mb', 'wind_kph', 'cloud', 'hour']
    
    data = df[features + ['temperature_celsius']].dropna()
    X = data[features]
    y = data['temperature_celsius']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, features

def get_realtime_weather(lat, lon):
    """Lấy dữ liệu thời tiết thực tế từ Open-Meteo API"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover&timezone=auto"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data['current']
        
        return {
            'latitude': lat,
            'humidity': current['relative_humidity_2m'],
            'pressure_mb': current['surface_pressure'],
            'wind_kph': current['wind_speed_10m'] * 3.6,
            'cloud': current['cloud_cover'],
            'hour': datetime.now().hour
        }
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy dữ liệu thời tiết: {e}")
        return None

# ===== GIAO DIỆN CHÍNH =====
def main():
    # Header
    st.title("🌍 Phân tích Tác động Khí hậu lên Sức khỏe")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Menu Điều hướng")
        
        menu = st.radio(
            "Chọn chức năng:",
            ["🏠 Tổng quan", "📈 Phân tích & Báo cáo", "🔬 Dự đoán Bệnh", 
             "🌡️ Dự đoán Nhiệt độ", "ℹ️ Hướng dẫn"],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.info("💡 Sử dụng menu để khám phá các tính năng")
    
    # ===== TRANG TỔNG QUAN =====
    if menu == "🏠 Tổng quan":
        st.header("📋 Giới thiệu Dự án")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Mục tiêu")
            st.write("""
            Dự án phân tích và dự đoán tác động của biến đổi khí hậu đến sức khỏe:
            
            - **Bệnh hô hấp** từ PM2.5 và chỉ số chất lượng không khí
            - **Bệnh lây truyền qua sinh vật trung gian** từ nhiệt độ và lượng mưa
            - **Ca nhập viện** do nắng nóng
            - **Nhiệt độ** từ dữ liệu khí tượng thực tế
            """)
        
        with col2:
            st.subheader("📊 Dữ liệu")
            st.write("""
            Hai nguồn dữ liệu chính:
            
            - **Global Climate Health Impact Tracker (2015-2025)**: 14,100 bản ghi
            - **Global Weather Repository**: Dữ liệu từ 195 quốc gia
            
            Tổng cộng hơn **30 biến số** được phân tích
            """)
        
        # Thống kê tổng quan
        health_df = load_health_data()
        weather_df = load_weather_data()
        
        if health_df is not None and weather_df is not None:
            st.divider()
            st.subheader("📊 Thống kê Tổng quan")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🗂️ Bản ghi Sức khỏe", f"{len(health_df):,}")
            with col2:
                st.metric("🌍 Quốc gia", health_df['country_name'].nunique())
            with col3:
                st.metric("📍 Địa điểm Thời tiết", len(weather_df))
            with col4:
                st.metric("📅 Năm Phân tích", f"{health_df['year'].min()}-{health_df['year'].max()}")
    
    # ===== TRANG PHÂN TÍCH & BÁO CÁO =====
    elif menu == "📈 Phân tích & Báo cáo":
        st.header("📈 Phân tích Dữ liệu & Báo cáo Nghiên cứu")
        
        health_df = load_health_data()
        
        if health_df is not None:
            # Tạo tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Tổng quan", 
                "📋 Báo cáo Nghiên cứu", 
                "🔥 Tương quan",
                "📈 Hiệu suất Mô hình",
                "📉 Xu hướng Chi tiết"
            ])
            
            # ===== TAB 1: TỔNG QUAN =====
            with tab1:
                st.subheader("📋 Thông tin Dữ liệu")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Tổng bản ghi", f"{len(health_df):,}")
                with col2:
                    st.metric("🔢 Số cột", len(health_df.columns))
                with col3:
                    st.metric("🌍 Số quốc gia", health_df['country_name'].nunique())
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🔍 Dữ liệu mẫu:**")
                    st.dataframe(health_df.head(10), use_container_width=True, height=400)
                
                with col2:
                    st.write("**📊 Thống kê Mô tả:**")
                    st.dataframe(health_df.describe().T, use_container_width=True, height=400)
                
                st.divider()
                st.write("**📝 Thông tin Cột:**")
                
                col_info = []
                for col in health_df.columns:
                    col_info.append({
                        'Tên cột': col,
                        'Kiểu': str(health_df[col].dtype),
                        'Null': health_df[col].isnull().sum(),
                        '% Null': f"{(health_df[col].isnull().sum() / len(health_df) * 100):.2f}%",
                        'Unique': health_df[col].nunique()
                    })
                
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, height=400)
            
            # ===== TAB 2: BÁO CÁO NGHIÊN CỨU =====
            with tab2:
                st.subheader("📋 Kết quả Nghiên cứu")
                
                with st.spinner("⏳ Đang phân tích dữ liệu..."):
                    findings_df = generate_research_findings(health_df)
                    correlations = analyze_correlations(health_df)
                
                # Tóm tắt
                st.info(f"""
                **Phân tích {len(health_df):,} bản ghi** từ **{health_df['country_name'].nunique()} quốc gia** 
                trong giai đoạn **{health_df['year'].min()}-{health_df['year'].max()}**
                """)
                
                st.divider()
                
                # Các phát hiện chính
                st.subheader("🔍 Các Phát hiện Chính")
                
                # Phát hiện 1: PM2.5 và Bệnh hô hấp
                pm25_high = health_df[health_df['pm25_ugm3'] > 50]
                pm25_low = health_df[health_df['pm25_ugm3'] <= 50]
                resp_diff = pm25_high['respiratory_disease_rate'].mean() - pm25_low['respiratory_disease_rate'].mean()
                
                st.write("**1️⃣ Chất lượng Không khí và Bệnh Hô hấp**")
                st.info(f"""
                **Phát hiện:** Tỷ lệ bệnh hô hấp cao hơn **{resp_diff:.1f}%** khi PM2.5 > 50 μg/m³
                
                **Cách thức tác động:**
                - **PM2.5** (bụi mịn): Hạt bụi nhỏ hơn 2.5 micromet xâm nhập sâu vào phổi, gây viêm đường hô hấp
                - **Chỉ số chất lượng không khí (AQI)**: Phản ánh tổng hợp các chất ô nhiễm, ảnh hưởng trực tiếp đến hệ hô hấp
                - Khi PM2.5 vượt ngưỡng 50 μg/m³, nguy cơ mắc các bệnh như hen suyễn, viêm phế quản tăng đáng kể
                
                **Mức độ tác động:** {'Cao' if resp_diff > 10 else 'Trung bình'} | **Số mẫu phân tích:** {len(pm25_high):,}
                """)
                
                # Phát hiện 2: Nhiệt độ và Bệnh sinh vật trung gian
                temp_high = health_df[health_df['temperature_celsius'] > 25]
                vector_high = temp_high['vector_disease_risk_score'].mean()
                vector_low = health_df[health_df['temperature_celsius'] <= 25]['vector_disease_risk_score'].mean()
                vector_diff = vector_high - vector_low
                
                st.write("**2️⃣ Nhiệt độ và Bệnh lây truyền qua Sinh vật trung gian**")
                st.info(f"""
                **Phát hiện:** Điểm rủi ro bệnh tăng **{vector_diff:.1f} điểm** khi nhiệt độ > 25°C
                
                **Cách thức tác động:**
                - **Nhiệt độ**: Môi trường ấm (>25°C) tạo điều kiện thuận lợi cho muỗi, ruồi và các sinh vật trung gian sinh sản nhanh
                - **Lượng mưa**: Tạo vũng nước đọng - nơi sinh sản lý tưởng cho muỗi truyền bệnh sốt rét, sốt xuất huyết
                - Chu kỳ sinh trưởng của muỗi rút ngắn từ 10 ngày xuống 7 ngày khi nhiệt độ tăng
                
                **Mức độ tác động:** {'Cao' if vector_diff > 1 else 'Trung bình'} | **Số mẫu phân tích:** {len(temp_high):,}
                """)
                
                # Phát hiện 3: Nắng nóng và Ca nhập viện
                heat_wave = health_df[health_df['heat_wave_days'] > 0]
                admission_ratio = heat_wave['heat_related_admissions'].mean() / health_df['heat_related_admissions'].mean()
                
                st.write("**3️⃣ Nắng nóng và Ca Nhập viện**")
                st.info(f"""
                **Phát hiện:** Ca nhập viện tăng **{(admission_ratio - 1) * 100:.1f}%** trong đợt nắng nóng
                
                **Cách thức tác động:**
                - **Số ngày nắng nóng**: Cơ thể phải điều hòa nhiệt liên tục, gây mệt mỏi và suy giảm chức năng
                - **Nhiệt độ cao**: Gây mất nước, sốc nhiệt, đột quỵ nhiệt ở người già và trẻ em
                - **Lượng mưa thấp**: Làm tăng nồng độ ô nhiễm không khí, tăng gánh nặng cho hệ hô hấp
                - **Sự kiện thời tiết cực đoan**: Đợt nóng kéo dài khiến cơ thể không kịp thích nghi
                
                **Mức độ tác động:** {'Rất cao' if admission_ratio > 2 else 'Cao'} | **Số mẫu phân tích:** {len(heat_wave):,}
                """)
                
                # Phát hiện 4: Thời tiết cực đoan
                extreme = health_df[health_df['extreme_weather_events'] > 0]
                health_impact = extreme[['respiratory_disease_rate', 'cardio_mortality_rate', 
                                        'vector_disease_risk_score']].mean().mean()
                normal_health = health_df[health_df['extreme_weather_events'] == 0][
                    ['respiratory_disease_rate', 'cardio_mortality_rate', 'vector_disease_risk_score']
                ].mean().mean()
                
                st.write("**4️⃣ Thời tiết Cực đoan**")
                st.info(f"""
                **Phát hiện:** Tác động sức khỏe tổng thể tăng **{((health_impact/normal_health - 1) * 100):.1f}%**
                
                **Cách thức tác động:**
                - **Bão, lũ lụt**: Phá hủy cơ sở hạ tầng y tế, ô nhiễm nguồn nước, lan truyền dịch bệnh
                - **Hạn hán**: Thiếu nước sạch, suy dinh dưỡng, bệnh truyền nhiễm qua đường tiêu hóa
                - **Sóng nhiệt**: Gây stress nhiệt, tăng tử vong do bệnh tim mạch
                - Các sự kiện cực đoan thường đi kèm nhau (hạn hán + nắng nóng, bão + lũ), gây tác động kép
                
                **Mức độ tác động:** Rất cao | **Số mẫu phân tích:** {len(extreme):,}
                """)
                
                # Tổng kết bằng expander
                with st.expander("📊 Xem Bảng Tóm tắt Phát hiện"):
                    summary_df = findings_df
                    st.dataframe(summary_df, use_container_width=True)
                
                st.divider()
                
                # Hệ số tương quan
                st.subheader("📊 Hệ số Tương quan Chi tiết")
                
                corr_cols = st.columns(2)
                
                with corr_cols[0]:
                    st.metric("🌫️ PM2.5 ↔ Bệnh Hô hấp", f"{correlations['pm25_respiratory']:.3f}")
                    st.caption("PM2.5 tăng → tỷ lệ bệnh hô hấp tăng")
                    
                    st.metric("🦟 Nhiệt độ ↔ Bệnh qua sinh vật trung gian", f"{correlations['temp_vector']:.3f}")
                    st.caption("Nhiệt độ cao → rủi ro bệnh tăng")
                
                with corr_cols[1]:
                    st.metric("🔥 Nắng nóng ↔ Ca Nhập viện", f"{correlations['heat_admission']:.3f}")
                    st.caption("Nắng nóng → ca nhập viện tăng")
                
                st.divider()
                
                # Phân tích theo vùng
                st.subheader("🌍 Phân tích theo Khu vực")
                
                region_stats = health_df.groupby('region').agg({
                    'respiratory_disease_rate': 'mean',
                    'vector_disease_risk_score': 'mean',
                    'heat_related_admissions': 'mean',
                    'temperature_celsius': 'mean',
                    'pm25_ugm3': 'mean'
                }).round(2)
                
                st.dataframe(region_stats, use_container_width=True)
                
                # Biểu đồ
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                region_stats['respiratory_disease_rate'].plot(kind='barh', ax=axes[0], color='steelblue')
                axes[0].set_title('Tỷ lệ Bệnh Hô hấp theo Vùng')
                axes[0].set_xlabel('Tỷ lệ (%)')
                axes[0].grid(True, alpha=0.3)
                
                region_stats['vector_disease_risk_score'].plot(kind='barh', ax=axes[1], color='coral')
                axes[1].set_title('Điểm Rủi ro Bệnh truyền nhiễm theo Vùng')
                axes[1].set_xlabel('Điểm rủi ro')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.divider()
                
                # Kết luận
                st.subheader("💡 Kết luận")
                
                st.success("""
                **Kết luận chính:**
                
                1. Chất lượng không khí có tác động trực tiếp đến bệnh hô hấp
                2. Biến đổi khí hậu làm tăng rủi ro bệnh lây truyền qua sinh vật trung gian
                3. Hiện tượng nắng nóng ngày càng nghiêm trọng
                4. Thời tiết cực đoan ảnh hưởng đa chiều đến sức khỏe
                """)
                
                st.warning("""
                **Khuyến nghị:**
                
                - Tăng cường giám sát chất lượng không khí
                - Chuẩn bị nguồn lực y tế cho khu vực nguy cơ cao
                - Nâng cao nhận thức cộng đồng
                - Tiếp tục nghiên cứu và phát triển mô hình dự đoán
                """)
            
            # ===== TAB 3: TƯƠNG QUAN =====
            with tab3:
                st.subheader("🔥 Ma trận Tương quan")
                
                numeric_cols = health_df.select_dtypes(include=[np.number]).columns.tolist()
                
                default_vars = ['temperature_celsius', 'pm25_ugm3', 'respiratory_disease_rate', 
                               'vector_disease_risk_score', 'heat_related_admissions',
                               'air_quality_index', 'precipitation_mm', 'cardio_mortality_rate']
                default_vars = [v for v in default_vars if v in numeric_cols]
                
                selected_cols = st.multiselect(
                    "Chọn biến:",
                    numeric_cols,
                    default=default_vars[:min(10, len(default_vars))]
                )
                
                if selected_cols:
                    correlation = health_df[selected_cols].corr()
                    
                    with st.expander("📊 Xem Ma trận Số"):
                        st.dataframe(correlation.style.format("{:.3f}"), use_container_width=True)
                    
                    # Heatmap
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                               fmt='.2f', ax=ax, square=True, linewidths=0.5)
                    plt.title('Ma trận Tương quan', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Các cặp tương quan cao
                    st.divider()
                    st.subheader("🔍 Tương quan Cao (|r| > 0.5)")
                    
                    high_corr = []
                    for i in range(len(correlation.columns)):
                        for j in range(i+1, len(correlation.columns)):
                            corr_val = correlation.iloc[i, j]
                            if abs(corr_val) > 0.5:
                                high_corr.append({
                                    'Biến 1': correlation.columns[i],
                                    'Biến 2': correlation.columns[j],
                                    'Hệ số': corr_val,
                                    'Loại': 'Dương' if corr_val > 0 else 'Âm'
                                })
                    
                    if high_corr:
                        st.dataframe(
                            pd.DataFrame(high_corr).sort_values('Hệ số', key=abs, ascending=False),
                            use_container_width=True
                        )
                    else:
                        st.info("Không có cặp biến nào có |r| > 0.5")
                else:
                    st.warning("⚠️ Vui lòng chọn ít nhất một biến")
            
            # ===== TAB 4: HIỆU SUẤT MÔ HÌNH =====
            with tab4:
                st.subheader("📈 Đánh giá Mô hình")
                
                with st.spinner("⏳ Đang tính toán..."):
                    metrics_df = calculate_model_metrics(health_df)
                
                st.dataframe(
                    metrics_df.style.format({
                        'R² Score': '{:.4f}',
                        'RMSE': '{:.4f}',
                        'Số mẫu': '{:,.0f}'
                    }),
                    use_container_width=True
                )
                
                st.info("""
                **Giải thích:**
                - **R² Score**: Hệ số xác định (0-1). Càng gần 1 càng tốt. R² > 0.7 = tốt
                - **RMSE**: Sai số trung bình. Giá trị càng thấp càng tốt
                """)
                
                # Biểu đồ
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    metrics_df.plot(x='Mô hình', y='R² Score', kind='barh', ax=ax, 
                                   color='steelblue', legend=False)
                    ax.set_xlabel('R² Score')
                    ax.set_ylabel('')
                    ax.set_title('So sánh R² Score')
                    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    metrics_df.plot(x='Mô hình', y='RMSE', kind='barh', ax=ax, 
                                   color='coral', legend=False)
                    ax.set_xlabel('RMSE')
                    ax.set_ylabel('')
                    ax.set_title('So sánh RMSE')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            # ===== TAB 5: XU HƯỚNG =====
            with tab5:
                st.subheader("📉 Xu hướng Chi tiết Theo Ngày")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    metric = st.selectbox(
                        "Chọn chỉ số:",
                        ['respiratory_disease_rate', 'vector_disease_risk_score', 
                         'heat_related_admissions', 'temperature_celsius', 
                         'pm25_ugm3', 'cardio_mortality_rate']
                    )
                
                with col2:
                    country = st.selectbox(
                        "Chọn quốc gia:",
                        sorted(health_df['country_name'].unique())
                    )
                
                country_data = health_df[health_df['country_name'] == country].sort_values('date')
                
                if len(country_data) > 0:
                    # Biểu đồ xu hướng
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(country_data['date'], country_data[metric], 
                           marker='o', linewidth=2, markersize=4, color='steelblue')
                    
                    # Moving average
                    if len(country_data) > 4:
                        ma = country_data[metric].rolling(window=4, center=True).mean()
                        ax.plot(country_data['date'], ma, linewidth=3, color='red', 
                               alpha=0.6, label='Xu hướng (MA-4)', linestyle='--')
                        ax.legend()
                    
                    ax.set_xlabel('Thời gian')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f'{metric.replace("_", " ").title()} - {country}')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Thống kê
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Trung bình", f"{country_data[metric].mean():.2f}")
                    with col2:
                        st.metric("📈 Cao nhất", f"{country_data[metric].max():.2f}")
                    with col3:
                        st.metric("📉 Thấp nhất", f"{country_data[metric].min():.2f}")
                    with col4:
                        st.metric("📏 Độ lệch chuẩn", f"{country_data[metric].std():.2f}")
                else:
                    st.warning(f"⚠️ Không có dữ liệu cho {country}")
    
    # ===== TRANG DỰ ĐOÁN BỆNH =====
    elif menu == "🔬 Dự đoán Bệnh":
        st.header("🔬 Dự đoán Tác động Sức khỏe")
        
        health_df = load_health_data()
        
        if health_df is not None:
            model_type = st.selectbox(
                "Chọn loại dự đoán:",
                ["Bệnh hô hấp", "Bệnh lây truyền qua sinh vật trung gian", "Ca nhập viện do nắng nóng"]
            )
            
            if st.button("🚀 Huấn luyện Mô hình", type="primary"):
                with st.spinner("⏳ Đang huấn luyện..."):
                    
                    if model_type == "Bệnh hô hấp":
                        model, rmse, r2, X_test, y_test, y_pred = train_respiratory_model(health_df)
                        st.session_state['resp_model'] = model
                        st.session_state['resp_rmse'] = rmse
                        st.session_state['resp_r2'] = r2
                        
                        st.success(f"✅ Hoàn tất! RMSE: {rmse:.2f}, R²: {r2:.2f}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(y_test, y_pred, alpha=0.5, color='darkgreen')
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                                   'r--', lw=2)
                            ax.set_xlabel('Thực tế')
                            ax.set_ylabel('Dự đoán')
                            ax.set_title('So sánh Thực tế vs Dự đoán')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.metric("R² Score", f"{r2:.4f}")
                            st.metric("RMSE", f"{rmse:.4f}")
                            st.metric("Số mẫu test", len(y_test))
                    
                    elif model_type == "Bệnh lây truyền qua sinh vật trung gian":
                        model, rmse, r2, X_test, y_test, y_pred = train_vector_disease_model(health_df)
                        st.session_state['vector_model'] = model
                        st.session_state['vector_rmse'] = rmse
                        st.session_state['vector_r2'] = r2
                        
                        st.success(f"✅ Hoàn tất! RMSE: {rmse:.2f}, R²: {r2:.2f}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(y_test, y_pred, alpha=0.5, color='darkgreen')
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                                   'r--', lw=2)
                            ax.set_xlabel('Thực tế')
                            ax.set_ylabel('Dự đoán')
                            ax.set_title('So sánh Thực tế vs Dự đoán')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.metric("R² Score", f"{r2:.4f}")
                            st.metric("RMSE", f"{rmse:.4f}")
                            st.metric("Số mẫu test", len(y_test))
                    
                    else:
                        model, rmse, r2, X_test, y_test, y_pred = train_heat_admission_model(health_df)
                        st.session_state['heat_model'] = model
                        st.session_state['heat_rmse'] = rmse
                        st.session_state['heat_r2'] = r2
                        
                        st.success(f"✅ Hoàn tất! RMSE: {rmse:.2f}, R²: {r2:.2f}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(y_test, y_pred, alpha=0.5, color='darkgreen')
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                                   'r--', lw=2)
                            ax.set_xlabel('Thực tế')
                            ax.set_ylabel('Dự đoán')
                            ax.set_title('So sánh Thực tế vs Dự đoán')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.metric("R² Score", f"{r2:.4f}")
                            st.metric("RMSE", f"{rmse:.4f}")
                            st.metric("Số mẫu test", len(y_test))
            
            st.divider()
            st.subheader("🔮 Dự đoán Mới")
            
            if model_type == "Bệnh hô hấp" and 'resp_model' in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    pm25 = st.number_input("PM2.5 (μg/m³):", min_value=0.0, max_value=500.0, value=50.0)
                with col2:
                    aqi = st.number_input("Chỉ số chất lượng không khí:", min_value=0.0, max_value=500.0, value=100.0)
                
                if st.button("🔍 Dự đoán", type="primary"):
                    pred = st.session_state['resp_model'].predict([[pm25, aqi]])[0]
                    st.success(f"### Tỷ lệ bệnh hô hấp: **{pred:.2f}%**")
            
            elif model_type == "Bệnh lây truyền qua sinh vật trung gian" and 'vector_model' in st.session_state:
                col1, col2, col3 = st.columns(3)
                with col1:
                    temp = st.number_input("Nhiệt độ (°C):", min_value=-20.0, max_value=50.0, value=25.0)
                with col2:
                    precip = st.number_input("Lượng mưa (mm):", min_value=0.0, max_value=500.0, value=50.0)
                with col3:
                    heat_adm = st.number_input("Ca nhập viện:", min_value=0.0, max_value=100.0, value=10.0)
                
                if st.button("🔍 Dự đoán", type="primary"):
                    pred = st.session_state['vector_model'].predict([[temp, precip, heat_adm]])[0]
                    st.success(f"### Điểm rủi ro: **{pred:.2f}**")
            
            elif model_type == "Ca nhập viện do nắng nóng" and 'heat_model' in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    temp = st.number_input("Nhiệt độ (°C):", min_value=-20.0, max_value=50.0, value=30.0)
                    precip = st.number_input("Lượng mưa (mm):", min_value=0.0, max_value=500.0, value=20.0)
                with col2:
                    heat_days = st.number_input("Số ngày nắng nóng:", min_value=0, max_value=30, value=5)
                    extreme = st.number_input("Số sự kiện cực đoan:", min_value=0, max_value=10, value=1)
                
                if st.button("🔍 Dự đoán", type="primary"):
                    pred = st.session_state['heat_model'].predict([[temp, precip, heat_days, extreme]])[0]
                    st.success(f"### Số ca nhập viện: **{pred:.1f}**")
    
    # ===== TRANG DỰ ĐOÁN NHIỆT ĐỘ =====
    elif menu == "🌡️ Dự đoán Nhiệt độ":
        st.header("🌡️ Dự đoán & Phân tích Nhiệt độ")
        
        weather_df = load_weather_data()
        health_df = load_health_data() # Dùng cho biểu đồ xu hướng

        if weather_df is not None:
            
            # TẠO TABS CHO PHẦN DỰ ĐOÁN NHIỆT ĐỘ
            temp_tab1, temp_tab2, temp_tab3 = st.tabs([
                "🔮 Mô hình & Dự đoán", 
                "📖 Giải thích Đặc trưng", 
                "📉 Xu hướng Nhiệt độ (Năm)"
            ])

            # --- TAB 1: MODEL ---
            with temp_tab1:
                st.subheader("🤖 Huấn luyện Mô hình & Dự báo Thực tế")
                
                if st.button("🚀 Huấn luyện Mô hình Nhiệt độ", type="primary"):
                    with st.spinner("⏳ Đang huấn luyện..."):
                        model, rmse, r2, features = train_temperature_model(weather_df)
                        st.session_state['temp_model'] = model
                        st.session_state['temp_rmse'] = rmse
                        st.session_state['temp_r2'] = r2
                        st.session_state['temp_features'] = features
                        st.session_state['temp_feature_importances'] = model.feature_importances_
                        
                        st.success(f"✅ Hoàn tất! RMSE: {rmse:.2f}°C, R²: {r2:.2f}")

                if 'temp_model' in st.session_state:
                    st.divider()
                    st.subheader("🔮 Dự đoán Thời gian thực (Open-Meteo API)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        lat = st.number_input("Vĩ độ (Latitude):", min_value=-90.0, max_value=90.0, value=21.02, step=0.01)
                        lon = st.number_input("Kinh độ (Longitude):", min_value=-180.0, max_value=180.0, value=105.83, step=0.01)
                    
                    with col2:
                        st.info("""
                        **Tọa độ tham khảo:**
                        - Hà Nội: 21.02, 105.83
                        - TP.HCM: 10.82, 106.63
                        - Đà Nẵng: 16.07, 108.22
                        """)
                    
                    if st.button("🌐 Lấy Dữ liệu & Dự đoán", type="primary"):
                        weather_data = get_realtime_weather(lat, lon)
                        
                        if weather_data:
                            new_data = pd.DataFrame([weather_data])
                            prediction = st.session_state['temp_model'].predict(new_data)[0]
                            rmse = st.session_state['temp_rmse']
                            
                            st.divider()
                            st.subheader("📊 Kết quả Dự báo")
                            
                            r_col1, r_col2, r_col3 = st.columns(3)
                            
                            with r_col1:
                                st.metric("🌡️ Nhiệt độ Dự báo", f"{prediction:.2f}°C")
                            with r_col2:
                                st.metric("❄️ Cận dưới (Min)", f"{(prediction - rmse):.2f}°C")
                            with r_col3:
                                st.metric("🔥 Cận trên (Max)", f"{(prediction + rmse):.2f}°C")
                            
                            st.caption(f"Dự báo dựa trên độ ẩm {weather_data['humidity']}%, gió {weather_data['wind_kph']:.1f} km/h, mây {weather_data['cloud']}%")

            # --- TAB 2: EXPLANATION ---
            with temp_tab2:
                st.subheader("📖 Kiến thức Khí tượng & Đặc trưng")
                
                # Hiển thị biểu đồ tầm quan trọng nếu đã train model
                if 'temp_feature_importances' in st.session_state and 'temp_features' in st.session_state:
                    st.write("**📊 Tầm quan trọng của các yếu tố (từ Mô hình đã học):**")
                    
                    feat_df = pd.DataFrame({
                        'Đặc trưng': st.session_state['temp_features'],
                        'Importance': st.session_state['temp_feature_importances']
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=feat_df, x='Importance', y='Đặc trưng', ax=ax, palette='viridis')
                    ax.set_title('Mức độ ảnh hưởng đến Nhiệt độ')
                    st.pyplot(fig)
                    plt.close()
                    st.divider()
                else:
                    st.info("💡 Hãy huấn luyện mô hình ở Tab 'Dự đoán' để xem biểu đồ mức độ quan trọng thực tế.")
                
                st.markdown("### Giải thích chi tiết các yếu tố ảnh hưởng:")
                
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    st.success("""
                    **1. Vĩ độ (Latitude)**
                    - **Ý nghĩa:** Khoảng cách từ vị trí đến xích đạo.
                    - **Tác động:** Vùng xích đạo (vĩ độ thấp) nhận nhiều năng lượng mặt trời hơn nên nóng hơn. Vùng cực (vĩ độ cao) lạnh hơn.
                    
                    **2. Độ ẩm (Humidity)**
                    - **Ý nghĩa:** Lượng hơi nước trong không khí.
                    - **Tác động:** Không khí ẩm giữ nhiệt tốt hơn (hiệu ứng nhà kính cục bộ). Độ ẩm cao làm giảm sự bay hơi, khiến cảm giác nóng bức hơn thực tế.
                    
                    **3. Giờ trong ngày (Hour)**
                    - **Ý nghĩa:** Thời điểm lấy dữ liệu (0-23h).
                    - **Tác động:** Nhiệt độ thường thấp nhất lúc bình minh và cao nhất vào khoảng 14h-15h chiều do độ trễ nhiệt của mặt đất.
                    """)
                
                with col_exp2:
                    st.info("""
                    **4. Áp suất khí quyển (Pressure)**
                    - **Ý nghĩa:** Trọng lượng của cột không khí.
                    - **Tác động:** Áp suất cao thường đi kèm trời nắng, ít mây. Áp suất thấp thường báo hiệu mưa, bão hoặc mây mù (nhiệt độ mát hơn).
                    
                    **5. Tốc độ gió (Wind Speed)**
                    - **Ý nghĩa:** Sự di chuyển của không khí.
                    - **Tác động:** Gió giúp tản nhiệt bề mặt, tăng tốc độ bay hơi làm mát. Gió mạnh cũng có thể mang khối khí nóng/lạnh từ nơi khác đến.
                    
                    **6. Độ che phủ mây (Cloud Cover)**
                    - **Ý nghĩa:** Phần trăm bầu trời bị mây che.
                    - **Tác động:** Ban ngày mây cản nắng (làm mát). Ban đêm mây giữ nhiệt từ mặt đất không cho thoát ra (làm ấm).
                    """)

            # --- TAB 3: TRENDS ---
            with temp_tab3:
                st.subheader("📉 Xu hướng Nhiệt độ Trung bình theo Năm")
                
                if health_df is not None:
                    # Lấy danh sách quốc gia
                    countries = sorted(health_df['country_name'].unique().tolist())
                    location_options = ["Toàn cầu"] + countries
                    
                    selected_location = st.selectbox("🌍 Chọn phạm vi phân tích:", location_options)
                    
                    # Lọc dữ liệu
                    if selected_location == "Toàn cầu":
                        # Group theo năm, lấy trung bình
                        trend_data = health_df.groupby('year')['temperature_celsius'].agg(['mean', 'min', 'max', 'std']).reset_index()
                        title_chart = "Nhiệt độ Trung bình Toàn cầu (2015-2025)"
                    else:
                        filtered_df = health_df[health_df['country_name'] == selected_location]
                        trend_data = filtered_df.groupby('year')['temperature_celsius'].agg(['mean', 'min', 'max', 'std']).reset_index()
                        title_chart = f"Nhiệt độ Trung bình tại {selected_location} (2015-2025)"
                    
                    # Vẽ biểu đồ
                    if not trend_data.empty:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Vẽ đường trung bình
                        ax.plot(trend_data['year'], trend_data['mean'], marker='o', linewidth=3, color='#d62728', label='Nhiệt độ TB')
                        
                        # Vẽ khoảng dao động (Min - Max)
                        ax.fill_between(trend_data['year'], trend_data['min'], trend_data['max'], color='#d62728', alpha=0.1, label='Khoảng (Min-Max)')
                        
                        # Thêm chú thích giá trị lên điểm
                        for x, y in zip(trend_data['year'], trend_data['mean']):
                            ax.annotate(f"{y:.1f}°C", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
                        
                        ax.set_title(title_chart, fontsize=14)
                        ax.set_xlabel('Năm')
                        ax.set_ylabel('Nhiệt độ (°C)')
                        ax.grid(True, linestyle='--', alpha=0.5)
                        ax.legend()
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Hiển thị bảng dữ liệu
                        with st.expander("📋 Xem dữ liệu chi tiết"):
                            st.dataframe(trend_data.style.format("{:.2f}"), use_container_width=True)
                        
                        # Nhận xét ngắn gọn
                        avg_change = trend_data['mean'].iloc[-1] - trend_data['mean'].iloc[0]
                        trend_emoji = "🔥" if avg_change > 0 else "❄️"
                        st.info(f"**Nhận xét:** Trong giai đoạn khảo sát, nhiệt độ trung bình tại {selected_location} đã thay đổi khoảng **{avg_change:+.2f}°C** {trend_emoji}.")
                    else:
                        st.warning("Không đủ dữ liệu để vẽ biểu đồ.")
                else:
                    st.error("Chưa tải được dữ liệu Health Tracker để phân tích xu hướng.")

    
    # ===== TRANG HƯỚNG DẪN =====
    else:
        st.header("ℹ️ Hướng dẫn Sử dụng")
        
        st.subheader("📖 Giới thiệu")
        st.write("Ứng dụng phân tích tác động khí hậu đến sức khỏe con người")
        
        st.divider()
        
        st.subheader("🎯 Các Chức năng")
        
        st.write("""
        **1. Tổng quan** - Thống kê chung về dữ liệu
        
        **2. Phân tích & Báo cáo** - Tab chính với 5 phần:
        - Tổng quan Dữ liệu
        - Báo cáo Nghiên cứu (phát hiện, tương quan, phân tích vùng)
        - Ma trận Tương quan
        - Hiệu suất Mô hình
        - Xu hướng Thời gian
        
        **3. Dự đoán Bệnh** - 3 mô hình:
        - Bệnh hô hấp (PM2.5, AQI)
        - Bệnh lây truyền qua sinh vật trung gian (Nhiệt độ, Mưa)
        - Ca nhập viện (Nắng nóng, Cực đoan)
        
        **4. Dự đoán Nhiệt độ** - Có 3 tab:
        - **Dự đoán**: Huấn luyện mô hình & Dự báo từ API
        - **Giải thích**: Kiến thức về các biến khí tượng
        - **Xu hướng**: Biểu đồ nhiệt độ theo năm
        """)
        
        st.divider()
        
        st.subheader("📊 Dữ liệu")
        st.write("""
        - **Global Climate Health Impact Tracker**: 14,100 bản ghi
        - **Global Weather Repository**: 195 quốc gia
        - Tổng cộng 30+ biến số
        """)
        
        st.divider()
        
        st.subheader("⚠️ Lưu ý")
        st.write("""
        - File CSV phải nằm trong `data/`
        - Cần internet cho API thời tiết
        - Kết quả chỉ mang tính tham khảo
        """)
        
        st.success("✨ Chúc bạn khám phá thành công!")

# ===== CHẠY ỨNG DỤNG =====
if __name__ == "__main__":
    main()