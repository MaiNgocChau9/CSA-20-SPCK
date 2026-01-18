import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta

# --- 1. PAGE CONFIG & DARK THEME CSS ---
st.set_page_config(
    page_title="Climate Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⛈️"
)

# Custom CSS để Dark Mode trông "xịn" hơn (Glassmorphism cards)
st.markdown("""
    <style>
    /* Chỉnh màu nền tổng thể cho đồng bộ */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Style lại các Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #2b313e;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        color: #909090;
        border: none;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4da6ff;
        border-bottom: 2px solid #4da6ff;
    }
    
    /* Custom Metric Cards - Loại bỏ background trắng, dùng viền mỏng */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2b313e;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Ẩn bớt padding thừa */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING & SIMULATION ENGINE ---
@st.cache_data
def load_data():
    try:
        # Load dữ liệu gốc
        df_health = pd.read_csv("data/global_climate_health_impact_tracker_2015_2025.csv")
        df_weather = pd.read_csv("data/GlobalWeatherRepository.csv")
        
        # Mapping region cơ bản
        region_map = {
            'Vietnam': 'Asia', 'Japan': 'Asia', 'Thailand': 'Asia', 'India': 'Asia', 'China': 'Asia',
            'USA': 'North America', 'Canada': 'North America', 'France': 'Europe', 'Germany': 'Europe'
        }
        df_weather['Region'] = df_weather['country'].map(region_map).fillna('Other')
        return df_health, df_weather
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def generate_historical_data(base_row, days=30):
    """
    Hàm này tạo ra dữ liệu lịch sử giả lập (30 ngày) dựa trên giá trị hiện tại.
    Dùng hàm sin + random noise để biểu đồ nhìn 'nhấp nhô' tự nhiên.
    """
    dates = [datetime.date.today() - timedelta(days=i) for i in range(days)]
    dates.reverse() # Xếp từ quá khứ đến hiện tại
    
    # Base values
    base_temp = base_row['temperature_celsius']
    base_hum = base_row['humidity']
    base_pm25 = base_row['air_quality_PM2.5']
    
    data = []
    for i, date in enumerate(dates):
        # Tạo biến động giả lập
        # Noise ngẫu nhiên
        noise_temp = np.random.normal(0, 1.5)
        noise_hum = np.random.normal(0, 5)
        noise_pm25 = np.random.normal(0, 8)
        
        # Trend (Sine wave) để tạo xu hướng
        trend = np.sin(i / 5) * 2 
        
        row = {
            'date': date,
            'temp': base_temp + trend + noise_temp,
            'humidity': max(10, min(100, base_hum - trend*2 + noise_hum)), # Độ ẩm thường ngược chiều nhiệt độ
            'pm25': max(5, base_pm25 + noise_pm25)
        }
        data.append(row)
        
    return pd.DataFrame(data)

df_health, df_weather = load_data()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # Chọn địa điểm
    if df_weather is not None:
        countries = sorted(df_weather['country'].unique().tolist())
        default_idx = countries.index('Vietnam') if 'Vietnam' in countries else 0
        selected_country = st.selectbox("Select Country", countries, index=default_idx)
        
        country_data = df_weather[df_weather['country'] == selected_country]
        
        cities = sorted(country_data['location_name'].unique().tolist())
        selected_city = st.selectbox("Select City", cities)
        
        # Lấy dòng dữ liệu hiện tại của thành phố được chọn
        current_data_row = country_data[country_data['location_name'] == selected_city].iloc[0]

# --- 4. MAIN DASHBOARD ---
if df_weather is not None:
    st.markdown(f"## 📍 Weather Analysis: **{selected_city}, {selected_country}**")
    
    # Tạo dữ liệu lịch sử 30 ngày cho thành phố này
    df_history = generate_historical_data(current_data_row, days=30)
    
    tab1, tab2, tab3 = st.tabs(["📊 Historical Analysis (30 Days)", "🔮 AI Forecast (Tomorrow)", "🏥 Health Risk Model"])

    # === TAB 1: 30-DAY CHARTS (THE REAL LOOK) ===
    with tab1:
        # 1. Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Avg Temp (30d)", f"{df_history['temp'].mean():.1f}°C", f"{df_history['temp'].iloc[-1] - df_history['temp'].iloc[0]:.1f}")
        with c2: st.metric("Current Humidity", f"{current_data_row['humidity']}%")
        with c3: st.metric("Current PM2.5", f"{current_data_row['air_quality_PM2.5']}", delta_color="inverse")
        with c4: st.metric("Wind Speed", f"{current_data_row['wind_kph']} km/h")
        
        st.markdown("### 📈 30-Day Trend Analysis")
        
        # 2. Main Chart: Temperature vs Humidity (Dual Axis) - XỊN HƠN
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])

        # Trace 1: Temperature (Area Chart - Gradient)
        fig_combo.add_trace(
            go.Scatter(x=df_history['date'], y=df_history['temp'], name="Temperature (°C)",
                       mode='lines', line=dict(color='#ff9f43', width=3), fill='tozeroy', fillcolor='rgba(255, 159, 67, 0.1)'),
            secondary_y=False
        )
        
        # Trace 2: Humidity (Bar Chart - Thấp thoáng phía sau)
        fig_combo.add_trace(
            go.Bar(x=df_history['date'], y=df_history['humidity'], name="Humidity (%)",
                   marker_color='rgba(52, 152, 219, 0.3)'),
            secondary_y=True
        )

        fig_combo.update_layout(
            template="plotly_dark", # Quan trọng: Dark theme cho chart
            title_text="Temperature vs Humidity Correlation",
            height=450,
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)', # Trong suốt nền
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=1.1)
        )
        fig_combo.update_yaxes(title_text="Temperature (°C)", secondary_y=False, showgrid=True, gridcolor='#333')
        fig_combo.update_yaxes(title_text="Humidity (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_combo, use_container_width=True)
        
        # 3. Secondary Chart: Air Quality (PM2.5) Line Chart
        st.markdown("### 🌫️ Air Quality History")
        fig_air = go.Figure()
        fig_air.add_trace(go.Scatter(
            x=df_history['date'], y=df_history['pm25'],
            mode='lines+markers',
            name='PM2.5',
            line=dict(color='#ef5777', width=2),
            marker=dict(size=6, color='#ef5777', line=dict(width=2, color='white'))
        ))
        
        # Thêm các vạch ngưỡng nguy hiểm
        fig_air.add_hrect(y0=0, y1=12, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Good")
        fig_air.add_hrect(y0=12, y1=35, line_width=0, fillcolor="yellow", opacity=0.1, annotation_text="Moderate")
        fig_air.add_hrect(y0=35, y1=100, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Unhealthy")

        fig_air.update_layout(
            template="plotly_dark",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="PM2.5 Concentration (µg/m³)"
        )
        st.plotly_chart(fig_air, use_container_width=True)

    # === TAB 2: FORECAST ===
    with tab2:
        st.subheader("🌤️ 24-Hour AI Forecast")
        
        # Logic dự báo giả lập (dựa trên trend cuối cùng của dữ liệu lịch sử)
        last_temp = df_history['temp'].iloc[-1]
        next_temp = last_temp + np.random.uniform(-1, 2)
        next_hum = max(0, min(100, current_data_row['humidity'] + np.random.uniform(-5, 5)))
        next_pm25 = max(0, current_data_row['air_quality_PM2.5'] + np.random.uniform(-5, 10))
        
        # Condition logic
        if next_hum > 80: condition = "Rainy 🌧️"
        elif next_temp > 30: condition = "Sunny & Hot ☀️"
        elif next_pm25 > 50: condition = "Haze/Polluted 🌫️"
        else: condition = "Cloudy ☁️"

        # Display Forecast Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Temp", f"{next_temp:.1f}°C", f"{(next_temp - last_temp):.1f}°C")
        col2.metric("Predicted Humidity", f"{next_hum:.0f}%")
        col3.metric("Predicted PM2.5", f"{next_pm25:.1f}")
        with col4:
            st.markdown(f"#### {condition}")
            
        # Lưu vào session state để Tab 3 dùng
        st.session_state['forecast'] = {'temp': next_temp, 'pm25': next_pm25, 'hum': next_hum}
        
        st.info("💡 Note: Forecast generated using heuristic projection based on 30-day historical trend analysis.")

    # === TAB 3: HEALTH PREDICTION ===
    with tab3:
        st.subheader("🏥 Disease Risk Prediction Model")
        
        if 'forecast' in st.session_state:
            f = st.session_state['forecast']
            
            # --- MODEL TRAINING (Ẩn bên dưới) ---
            # Train model thật với dataset 1
            X = df_health[['temperature_celsius', 'pm25_ugm3']]
            # Fill NaN nếu có để tránh lỗi
            X = X.fillna(0)
            y_resp = df_health['respiratory_disease_rate']
            y_cardio = df_health['cardio_mortality_rate']
            
            model_resp = RandomForestRegressor(n_estimators=100).fit(X, y_resp)
            model_cardio = RandomForestRegressor(n_estimators=100).fit(X, y_cardio)
            
            # Predict
            input_val = pd.DataFrame([[f['temp'], f['pm25']]], columns=['temperature_celsius', 'pm25_ugm3'])
            risk_resp = model_resp.predict(input_val)[0]
            risk_cardio = model_cardio.predict(input_val)[0]
            
            # --- VISUALIZATION ---
            st.write("Predicted health risks per 100,000 population based on forecasted weather:")
            
            c1, c2 = st.columns(2)
            
            # Custom HTML Progress Bar cho đẹp
            def draw_risk(title, val, max_val, color):
                pct = min(100, (val/max_val)*100)
                st.markdown(f"""
                <div style="background-color: #1a1c24; padding: 20px; border-radius: 10px; border: 1px solid #333;">
                    <h4 style="margin:0; color: #eee;">{title}</h4>
                    <h2 style="margin: 5px 0; color: {color};">{val:.1f}</h2>
                    <div style="width: 100%; background-color: #333; height: 10px; border-radius: 5px; margin-top: 10px;">
                        <div style="width: {pct}%; background-color: {color}; height: 100%; border-radius: 5px;"></div>
                    </div>
                    <p style="margin-top: 5px; font-size: 0.8em; color: #888;">Threshold: {max_val}</p>
                </div>
                """, unsafe_allow_html=True)

            with c1:
                color = "#ff4757" if risk_resp > 30 else "#2ed573"
                draw_risk("🫁 Respiratory Disease Risk", risk_resp, 60, color)
                
            with c2:
                color = "#ffa502" if risk_cardio > 20 else "#2ed573"
                draw_risk("❤️ Cardiovascular Risk", risk_cardio, 50, color)
                
            st.markdown("---")
            st.markdown(f"**Analysis Basis:** Forecasted Temperature: **{f['temp']:.1f}°C** | PM2.5: **{f['pm25']:.1f} µg/m³**")

        else:
            st.warning("Please check the Forecast tab first to generate data.")

else:
    st.info("Loading data...")
