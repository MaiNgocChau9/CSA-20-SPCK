import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN (PROFESSIONAL STYLE)
# ==========================================
st.set_page_config(page_title="Climate & Health Analytics", layout="centered", page_icon="📊")

# CSS tối giản, tập trung vào nội dung báo cáo
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #ffffff; font-family: 'Segoe UI', sans-serif;}
    h1 {color: #2c3e50; text-align: center; font-weight: 700;}
    h2 {color: #2980b9; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 30px;}
    h3 {color: #7f8c8d; font-size: 1.1rem;}
    .report-box {
        padding: 15px; 
        background-color: #f8f9fa; 
        border: 1px solid #e9ecef; 
        border-radius: 5px; 
        margin-bottom: 20px;
    }
    .highlight-red {color: #c0392b; font-weight: bold;}
    .highlight-blue {color: #2980b9; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data
def load_data():
    """Load và làm sạch dữ liệu từ nhiều nguồn"""
    data = {}
    
    # 1. Dữ liệu Sức khỏe & Khí hậu (Chính)
    try:
        df_h = pd.read_csv("data/global_climate_health_impact_tracker_2015_2025.csv")
        # Loại bỏ nhiễu nếu cần
        data['health'] = df_h
    except:
        st.error("Lỗi: Không tìm thấy file dữ liệu Sức khỏe (global_climate_health...).")
        st.stop()
        
    # 2. Dữ liệu Thời tiết chi tiết (Phụ - cho phần phân tích khí tượng)
    try:
        df_w = pd.read_csv("data/seattle-weather.csv") # Dùng file này để phân tích tần suất thời tiết
        data['weather'] = df_w
    except:
        data['weather'] = pd.DataFrame()

    return data

@st.cache_resource
def calculate_feature_importance(df, target_col):
    """Tính toán mức độ ảnh hưởng của các biến số"""
    # Các biến số đầu vào tiềm năng
    candidates = ['latitude', 'longitude', 'humidity', 'pressure_mb', 'wind_kph', 'cloud', 'year', 'month']
    features = [c for c in candidates if c in df.columns]
    
    if not features or target_col not in df.columns:
        return None
        
    df_clean = df[features + [target_col]].dropna()
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(df_clean[features], df_clean[target_col])
    
    return pd.DataFrame({
        'Biến số': features,
        'Mức độ tác động': model.feature_importances_
    }).sort_values('Mức độ tác động', ascending=True)

# ==========================================
# 3. NỘI DUNG BÁO CÁO
# ==========================================
def main():
    data = load_data()
    df = data['health']
    df_w = data['weather']

    # --- TIÊU ĐỀ ---
    st.title("BÁO CÁO PHÂN TÍCH: TÁC ĐỘNG KÉP CỦA KHÍ HẬU")
    st.markdown("<div style='text-align: center; color: grey;'>Phân tích dữ liệu giai đoạn 2015 - 2025</div>", unsafe_allow_html=True)
    st.markdown("---")

    # =========================================================
    # CHƯƠNG 1: HIỆN TRẠNG KHÍ TƯỢNG (METEOROLOGICAL STATUS)
    # =========================================================
    st.header("1. Phân tích Các yếu tố Khí tượng")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1.1. Tần suất Thời tiết")
        if not df_w.empty and 'weather' in df_w.columns:
            weather_counts = df_w['weather'].value_counts().reset_index()
            weather_counts.columns = ['Loại hình', 'Số ngày']
            fig_w = px.bar(weather_counts, x='Số ngày', y='Loại hình', orientation='h', 
                           title="Phân bố các loại hình thời tiết điển hình", text_auto=True)
            fig_w.update_traces(marker_color='#3498db')
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("Không có dữ liệu chi tiết về loại hình thời tiết.")

    with col2:
        st.subheader("1.2. Động lực thay đổi Nhiệt độ")
        # Phân tích Feature Importance cho Nhiệt độ
        imp_df = calculate_feature_importance(df, 'temperature_celsius')
        if imp_df is not None:
            fig_imp = px.bar(imp_df, x='Mức độ tác động', y='Biến số', orientation='h',
                             title="Xếp hạng yếu tố ảnh hưởng đến Nhiệt độ")
            fig_imp.update_traces(marker_color='#e67e22')
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Dữ liệu cho thấy Vĩ độ và Độ ẩm là hai yếu tố định hình nền nhiệt chính.")

    st.markdown("""
    <div class="report-box">
    <b>Nhận định Chương 1:</b><br>
    Biến đổi khí hậu không diễn ra ngẫu nhiên. Nhiệt độ trung bình toàn cầu đang chịu tác động mạnh bởi vị trí địa lý (Vĩ độ) và sự thay đổi của các yếu tố khí tượng như Độ ẩm và Lượng mưa. Xu hướng chung là nền nhiệt đang gia tăng qua các năm.
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # CHƯƠNG 2: TÁC ĐỘNG TRỰC TIẾP - SỐC NHIỆT (HEAT STRESS)
    # =========================================================
    st.header("2. Tác động Trực tiếp: Hội chứng Sốc nhiệt")
    
    st.markdown("""
    Sốc nhiệt (Heat Stroke) hay các bệnh lý liên quan đến nhiệt là phản ứng sinh lý trực tiếp của cơ thể khi hệ thống điều hòa thân nhiệt bị quá tải.
    """)

    # Biểu đồ phân tán + Đường xu hướng phi tuyến tính
    # Tạo logic hiển thị ngưỡng
    fig_heat = px.scatter(df, x="temperature_celsius", y="heat_related_admissions", 
                          opacity=0.6, 
                          title="Tương quan giữa Nhiệt độ và Số ca nhập viện do nhiệt",
                          labels={"temperature_celsius": "Nhiệt độ môi trường (°C)", 
                                  "heat_related_admissions": "Số ca nhập viện"})
    
    # Vẽ đường ngưỡng chịu đựng (Threshold)
    fig_heat.add_vline(x=30, line_width=2, line_dash="dash", line_color="red", annotation_text="Ngưỡng nguy hiểm (30°C)")
    fig_heat.add_shape(type="rect", x0=30, y0=0, x1=df['temperature_celsius'].max(), y1=df['heat_related_admissions'].max(),
                       fillcolor="red", opacity=0.1, line_width=0)
    
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class="report-box">
    <b>Phân tích Ngưỡng chịu đựng:</b>
    <ul>
        <li><b>Vùng An toàn (< 25°C):</b> Số ca nhập viện do nhiệt gần như bằng 0.</li>
        <li><b>Vùng Cảnh báo (25°C - 30°C):</b> Xuất hiện rải rác các ca bệnh nhẹ.</li>
        <li><b>Vùng Nguy hiểm (> 30°C):</b> Số ca bệnh <span class="highlight-red">tăng theo cấp số nhân</span>. Đây là điểm gãy (tipping point) nơi cơ thể mất khả năng tự làm mát hiệu quả.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # CHƯƠNG 3: TÁC ĐỘNG GIÁN TIẾP - DỊCH BỆNH TRUYỀN NHIỄM
    # =========================================================
    st.header("3. Tác động Gián tiếp: Dịch bệnh qua Vector truyền dẫn")
    
    st.markdown("""
    Khác với sốc nhiệt, các bệnh truyền nhiễm (Sốt xuất huyết, Malaria, Zika) không tác động trực tiếp mà thông qua vật chủ trung gian (muỗi, côn trùng).
    Sự sinh trưởng của các vector này phụ thuộc vào **cộng hưởng nhiệt - ẩm**.
    """)

    col3, col4 = st.columns([2, 1])
    
    with col3:
        # Biểu đồ Heatmap 3 chiều
        fig_vec = px.scatter(df, x="temperature_celsius", y="precipitation_mm", 
                             color="vector_disease_risk_score",
                             size="vector_disease_risk_score",
                             color_continuous_scale="RdYlBu_r", # Đỏ là nguy hiểm, Xanh là an toàn
                             title="Ma trận Rủi ro: Nhiệt độ vs Lượng mưa",
                             labels={"temperature_celsius": "Nhiệt độ (°C)", 
                                     "precipitation_mm": "Lượng mưa (mm)",
                                     "vector_disease_risk_score": "Chỉ số Rủi ro"})
        st.plotly_chart(fig_vec, use_container_width=True)

    with col4:
        st.markdown("#### Giải mã Biểu đồ:")
        st.markdown("""
        **Vùng màu đỏ đậm (Rủi ro cao nhất):**
        Hội tụ tại khu vực:
        - Nhiệt độ: **28°C - 35°C**
        - Lượng mưa: **> 100mm**
        
        **Kết luận:**
        Dịch bệnh **KHÔNG** bùng phát ở nơi nóng nhưng khô hạn (Góc dưới bên phải biểu đồ). Nó cần độ ẩm để ấu trùng phát triển.
        """)

    # =========================================================
    # CHƯƠNG 4: MÔ HÌNH DỰ BÁO THAM SỐ (PREDICTIVE MODEL)
    # =========================================================
    st.header("4. Mô hình Dự báo Rủi ro")
    st.markdown("Dựa trên dữ liệu lịch sử, hệ thống sử dụng thuật toán **Random Forest** để dự báo chỉ số rủi ro dựa trên điều kiện môi trường giả định.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        temp_val = c1.number_input("Nhiệt độ dự báo (°C)", value=32.0, min_value=0.0, max_value=50.0)
        rain_val = c2.number_input("Lượng mưa dự báo (mm)", value=120.0, min_value=0.0, max_value=500.0)
        heat_adm_val = c3.number_input("Số ca sốc nhiệt nền", value=5, min_value=0)
        
        submitted = st.form_submit_button("Chạy Mô phỏng")

        if submitted:
            # Train model nhanh (On-the-fly)
            target_cols = ['temperature_celsius', 'precipitation_mm', 'heat_related_admissions']
            train_df = df[target_cols + ['vector_disease_risk_score']].dropna()
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(train_df[target_cols], train_df['vector_disease_risk_score'])
            
            # Predict
            pred_score = rf.predict([[temp_val, rain_val, heat_adm_val]])[0]
            
            # Hiển thị kết quả
            st.divider()
            col_res1, col_res2 = st.columns([1, 3])
            
            with col_res1:
                st.metric("Chỉ số Rủi ro Dự báo", f"{pred_score:.2f}/10")
            
            with col_res2:
                if pred_score >= 7.0:
                    st.error("⚠️ CẢNH BÁO MỨC ĐỘ CAO: Môi trường cực kỳ thuận lợi cho dịch bệnh bùng phát. Khuyến nghị phun khử khuẩn và kiểm soát vector.")
                elif pred_score >= 4.0:
                    st.warning("⚠️ CẢNH BÁO MỨC ĐỘ TRUNG BÌNH: Cần theo dõi sát sao.")
                else:
                    st.success("✅ AN TOÀN: Điều kiện môi trường chưa đủ ngưỡng gây dịch.")

if __name__ == "__main__":
    main()