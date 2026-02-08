import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="Phân tích Khí hậu & Sức khỏe",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# ===== HÀM TIỆN ÍCH =====
def load_health_data():
    """Tải dữ liệu khí hậu và sức khỏe"""
    try:
        df = pd.read_csv('data/global_climate_health_impact_tracker_2015_2025.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file dữ liệu sức khỏe!")
        return None

def analyze_correlations(df):
    """Phân tích tương quan chi tiết giữa các biến"""
    return {
        'pm25_respiratory': df[['pm25_ugm3', 'respiratory_disease_rate']].corr().iloc[0, 1],
        'temp_vector': df[['temperature_celsius', 'vector_disease_risk_score']].corr().iloc[0, 1],
        'heat_admission': df[['heat_wave_days', 'heat_related_admissions']].corr().iloc[0, 1],
        'aqi_cardio': df[['air_quality_index', 'cardio_mortality_rate']].corr().iloc[0, 1]
    }

def generate_research_findings(df):
    """Tạo các phát hiện nghiên cứu từ dữ liệu"""
    findings = []
    
    # 1. PM2.5 và bệnh hô hấp
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
    
    # 2. Nhiệt độ và bệnh sinh vật trung gian
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
    
    # 3. Nắng nóng và ca nhập viện
    heat_wave = df[df['heat_wave_days'] > 0]
    admission_ratio = heat_wave['heat_related_admissions'].mean() / df['heat_related_admissions'].mean()
    
    findings.append({
        'Danh mục': 'Nắng nóng',
        'Phát hiện': f'Ca nhập viện tăng {(admission_ratio - 1) * 100:.1f}% trong đợt nắng nóng',
        'Tác động': 'Rất cao' if admission_ratio > 2 else 'Cao',
        'Số mẫu': len(heat_wave),
        'Độ tin cậy': 'Cao'
    })
    
    return pd.DataFrame(findings)

def calculate_model_metrics(df):
    """Tính toán các chỉ số cho từng mô hình"""
    metrics = []
    
    # Model 1: Respiratory Disease
    X = df[['pm25_ugm3', 'air_quality_index']].dropna()
    y = df.loc[X.index, 'respiratory_disease_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
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
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
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
    model = LinearRegression().fit(X_train, y_train)
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

def train_model(df, features, target, model_type='linear'):
    """Hàm chung để huấn luyện mô hình"""
    X = df[features].dropna()
    y = df.loc[X.index, target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X_test, y_test, y_pred

# ===== GIAO DIỆN CHÍNH =====
def main():
    st.title("Phân tích Tác động Khí hậu lên Sức khỏe")
    
    # Sidebar
    st.markdown("""
        <style>
            /* Ẩn nút tròn (radio button) mặc định */
            [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
                display: none !important;
            }

            /* Định dạng từng dòng menu */
            [data-testid="stSidebar"] div[role="radiogroup"] > label {
                background-color: transparent;
                padding: 10px 15px;
                border-radius: 10px;
                margin-bottom: 5px;
                transition: all 0.3s;
                border: none;
                display: block;
                cursor: pointer;
            }

            /* Hiệu ứng khi di chuột qua (Hover) */
            [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
                background-color: rgba(151, 166, 195, 0.1);
            }

            /* Màu nền và chữ đậm cho mục đang được chọn (giống trong ảnh) */
            [data-testid="stSidebar"] div[role="radiogroup"] [data-checked="true"] {
                background-color: #e8eaed !important; /* Màu xám nhạt như ảnh */
                font-weight: bold;
                color: #1c1e21;
            }
            
            /* Chỉnh font chữ menu to và rõ hơn */
            [data-testid="stSidebar"] div[role="radiogroup"] label p {
                font-size: 18px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Menu Điều hướng")
        
        # Dùng st.radio nhưng CSS đã ẩn đi các nút tròn
        menu = st.radio(
            "Chọn chức năng:",
            ["Tổng quan", "Phân tích & Báo cáo", "Dự đoán Bệnh", "Hướng dẫn"],
            label_visibility="collapsed"
        )
    
    # ===== TỔNG QUAN =====
    if menu == "Tổng quan":
        st.header("Giới thiệu Dự án")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Mục tiêu")
            st.write("""
            Dự án phân tích và dự đoán tác động của biến đổi khí hậu đến sức khỏe:
            - **Bệnh hô hấp** từ PM2.5 và chỉ số chất lượng không khí
            - **Bệnh lây truyền qua sinh vật trung gian** từ nhiệt độ và lượng mưa
            - **Ca nhập viện** do nắng nóng
            """)
        
        with col2:
            st.subheader("📊 Dữ liệu")
            st.write("""
            Nguồn dữ liệu chính:
            - **Global Climate Health Impact Tracker (2015-2025)**: 14,100 bản ghi
            - Dữ liệu từ nhiều quốc gia
            
            Tổng cộng hơn **30 biến số** được phân tích
            """)
        
        health_df = load_health_data()
        
        if health_df is not None:
            st.divider()
            st.subheader("Thống kê Tổng quan")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Bản ghi", f"{len(health_df):,}")
            with col2:
                st.metric("Quốc gia", health_df['country_name'].nunique())
            with col3:
                st.metric("Khu vực", health_df['region'].nunique())
            with col4:
                st.metric("Năm", f"{health_df['year'].min()}-{health_df['year'].max()}")
    
    # ===== PHÂN TÍCH & BÁO CÁO =====
    elif menu == "Phân tích & Báo cáo":
        st.header("Phân tích Dữ liệu & Báo cáo Nghiên cứu")
        
        health_df = load_health_data()
        
        if health_df is not None:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Tổng quan",
                "📋 Báo cáo Nghiên cứu",
                "🔥 Tương quan",
                "📈 Hiệu suất Mô hình",
                "📉 Xu hướng Chi tiết"
            ])
            
            # TAB 1: TỔNG QUAN
            with tab1:
                st.subheader("Thông tin Dữ liệu")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng bản ghi", f"{len(health_df):,}")
                with col2:
                    st.metric("Số cột", len(health_df.columns))
                with col3:
                    st.metric("Số quốc gia", health_df['country_name'].nunique())
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dữ liệu mẫu:**")
                    st.dataframe(health_df.head(10), use_container_width=True, height=400)
                
                with col2:
                    st.write("**Thống kê Mô tả:**")
                    st.dataframe(health_df.describe().T, use_container_width=True, height=400)
                
                st.divider()
                st.write("**Thông tin Cột:**")
                
                col_info = [{
                    'Tên cột': col,
                    'Kiểu': str(health_df[col].dtype),
                    'Null': health_df[col].isnull().sum(),
                    '% Null': f"{(health_df[col].isnull().sum() / len(health_df) * 100):.2f}%",
                    'Unique': health_df[col].nunique()
                } for col in health_df.columns]
                
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, height=400)
            
            # TAB 2: BÁO CÁO NGHIÊN CỨU
            with tab2:
                st.subheader("Kết quả Nghiên cứu")
                
                with st.spinner("Đang phân tích dữ liệu..."):
                    correlations = analyze_correlations(health_df)
                
                st.info(f"""
                **Phân tích {len(health_df):,} bản ghi** từ **{health_df['country_name'].nunique()} quốc gia** 
                trong giai đoạn **{health_df['year'].min()}-{health_df['year'].max()}**
                """)
                
                st.divider()
                st.subheader("Các Phát hiện Chính")
                
                # Phát hiện 1: PM2.5
                st.write("**1. Chất lượng Không khí và Bệnh Hô hấp**")
                st.info(f"""
                ### **Phát hiện:** Tỷ lệ mắc bệnh hô hấp tương quan với lượng bụi mịn PM2.5
                
                **Cách thức tác động:**
                - PM2.5 xâm nhập sâu vào phổi, gây viêm đường hô hấp
                - AQI phản ánh tổng hợp các chất ô nhiễm, ảnh hưởng trực tiếp đến hệ hô hấp
                - Nguy cơ hen suyễn, viêm phế quản tăng đáng kể.
                """)
                
                pm25_data = health_df[['pm25_ugm3', 'respiratory_disease_rate']].dropna()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(pm25_data['pm25_ugm3'], pm25_data['respiratory_disease_rate'], alpha=0.5, s=20)
                ax.set_xlabel('PM2.5 (μg/m³)', fontsize=12)
                ax.set_ylabel('Tỷ lệ bệnh hô hấp (%)', fontsize=12)
                ax.set_title('PM2.5 vs Tỷ lệ bệnh hô hấp', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.divider()
                
                # Phát hiện 2: Nhiệt độ
                temp_high = health_df[health_df['temperature_celsius'] > 25]
                
                st.write("**2. Nhiệt độ và Bệnh lây truyền**")
                st.info(f"""
                ### **Phát hiện:** Rủi ro bệnh truyền nhiễm qua côn trùng tăng khi nhiệt độ > 20°C
                
                **Cách thức tác động:**
                - Nhiệt độ > 20°C tạo điều kiện cho muỗi, ruồi sinh sản nhanh
                - Lượng mưa tạo vũng nước - nơi sinh sản của muỗi sốt rét, sốt xuất huyết
                - Chu kỳ sinh trưởng muỗi diễn ra nhanh hơn khi nhiệt độ tăng
                """)
                
                temp_data = health_df[['temperature_celsius', 'vector_disease_risk_score']].dropna()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(temp_data['temperature_celsius'], temp_data['vector_disease_risk_score'], 
                          alpha=0.3, s=10, color='coral')
                ax.axvline(x=20, color='red', linestyle='--', linewidth=2, label='Ngưỡng = 20°C')
                ax.set_xlabel('Nhiệt độ (°C)', fontsize=12)
                ax.set_ylabel('Điểm rủi ro', fontsize=12)
                ax.set_title('Nhiệt độ - Rủi ro Bệnh', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.divider()
                
                # Phát hiện 3: Nắng nóng
                
                st.write("**3. Nắng nóng và Ca Nhập viện**")
                st.info(f"""
                ### **Phát hiện:** Ca nhập viện tăng trong các đợt nắng nóng dài ngày
                
                **Cách thức tác động:**
                - Cơ thể điều hòa nhiệt liên tục → mệt mỏi, suy giảm chức năng
                - Nhiệt độ cao gây mất nước, sốc nhiệt, đột quỵ nhiệt
                - Lượng mưa thấp tăng ô nhiễm không khí
                """)
                
                heat_grouped = health_df.groupby('heat_wave_days')['heat_related_admissions'].mean()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(heat_grouped.index, heat_grouped.values, 
                       marker='o', linewidth=2, markersize=8, color='orangered')
                ax.set_xlabel('Số ngày nắng nóng liên tục', fontsize=12)
                ax.set_ylabel('Ca nhập viện TB', fontsize=12)
                ax.set_title('Ca Nhập viện theo Nắng nóng', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.divider()
                st.subheader("Hệ số Tương quan")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("PM2.5 ↔ Hô hấp", f"{correlations['pm25_respiratory']:.3f}")
                    st.caption("PM2.5 ↑ → bệnh hô hấp ↑")
                    st.metric("Nhiệt độ ↔ Bệnh", f"{correlations['temp_vector']:.3f}")
                    st.caption("Nhiệt độ ↑ → rủi ro ↑")
                with col2:
                    st.metric("Nắng ↔ Nhập viện", f"{correlations['heat_admission']:.3f}")
                    st.caption("Nắng nóng ↑ → nhập viện ↑")
                
                st.divider()
                st.subheader("Phân tích theo Khu vực")
                
                region_stats = health_df.groupby('region').agg({
                    'respiratory_disease_rate': 'mean',
                    'vector_disease_risk_score': 'mean',
                    'heat_related_admissions': 'mean',
                    'temperature_celsius': 'mean',
                    'pm25_ugm3': 'mean'
                }).round(2)
                
                st.dataframe(region_stats, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                region_stats['respiratory_disease_rate'].plot(kind='barh', ax=ax, color='steelblue')
                ax.set_title('Tỷ lệ Bệnh Hô hấp theo Vùng', fontsize=14, fontweight='bold')
                ax.set_xlabel('Tỷ lệ (%)', fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                region_stats['vector_disease_risk_score'].plot(kind='barh', ax=ax, color='coral')
                ax.set_title('Rủi ro Bệnh theo Vùng', fontsize=14, fontweight='bold')
                ax.set_xlabel('Điểm rủi ro', fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.divider()
                st.subheader("Kết luận")
                st.success("""
                **Kết luận chính:**
                1. Chất lượng không khí tác động trực tiếp đến bệnh hô hấp
                2. Biến đổi khí hậu tăng rủi ro bệnh lây truyền
                3. Nắng nóng ngày càng nghiêm trọng
                """)
                
                st.warning("""
                **Khuyến nghị:**
                - Tăng cường giám sát chất lượng không khí
                - Chuẩn bị nguồn lực y tế cho vùng nguy cơ cao
                - Nâng cao nhận thức cộng đồng
                - Tiếp tục nghiên cứu và phát triển mô hình dự đoán
                """)
            
            # TAB 3: TƯƠNG QUAN
            with tab3:
                st.subheader("Ma trận Tương quan")
                
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
                    
                    with st.expander("Ma trận Số"):
                        st.dataframe(correlation.style.format("{:.3f}"), use_container_width=True)
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                               fmt='.2f', ax=ax, square=True, linewidths=0.5)
                    plt.title('Ma trận Tương quan', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.divider()
                    st.subheader("Tương quan Cao (|r| > 0.5)")
                    
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
                    st.warning("Vui lòng chọn ít nhất một biến")
            
            # TAB 4: HIỆU SUẤT
            with tab4:
                st.subheader("Đánh giá Mô hình")
                
                with st.spinner("Đang tính toán..."):
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
                - **R² Score**: 0-1, càng gần 1 càng tốt (> 0.7 = tốt)
                - **RMSE**: Sai số TB, càng thấp càng tốt
                """)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics_df.plot(x='Mô hình', y='R² Score', kind='barh', ax=ax, color='steelblue', legend=False)
                ax.set_xlabel('R² Score', fontsize=12)
                ax.set_title('So sánh R² Score', fontsize=14, fontweight='bold')
                ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics_df.plot(x='Mô hình', y='RMSE', kind='barh', ax=ax, color='coral', legend=False)
                ax.set_xlabel('RMSE', fontsize=12)
                ax.set_title('So sánh RMSE', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # TAB 5: XU HƯỚNG
            with tab5:
                st.subheader("Xu hướng Theo Thời gian")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    metric = st.selectbox(
                        "Chọn chỉ số:",
                        ['temperature_celsius', 'respiratory_disease_rate', 
                         'vector_disease_risk_score', 'heat_related_admissions', 
                         'pm25_ugm3', 'cardio_mortality_rate']
                    )
                
                with col2:
                    country = st.selectbox("Chọn quốc gia:", sorted(health_df['country_name'].unique()))
                
                country_data = health_df[health_df['country_name'] == country].sort_values('date')
                
                if len(country_data) > 0:
                    if metric == 'temperature_celsius':
                        st.subheader("🌡️ Xu hướng Nhiệt độ")
                        
                        yearly_stats = country_data.groupby('year')['temperature_celsius'].agg(['mean', 'min', 'max']).reset_index()
                        monthly_stats = country_data.groupby([country_data['date'].dt.to_period('M')])['temperature_celsius'].mean().reset_index()
                        monthly_stats['date'] = monthly_stats['date'].dt.to_timestamp()
                        
                        # Biểu đồ theo ngày
                        fig, ax = plt.subplots(figsize=(14, 6))
                        ax.plot(country_data['date'], country_data['temperature_celsius'], 
                               linewidth=1.5, color='steelblue', alpha=0.7, label='Thực tế')
                        
                        if len(country_data) > 12:
                            ma = country_data.set_index('date')['temperature_celsius'].rolling(window=12, center=True).mean()
                            ax.plot(ma.index, ma.values, linewidth=3, color='red', alpha=0.8, 
                                   label='Xu hướng (MA-12)', linestyle='-')
                        
                        ax.set_xlabel('Thời gian', fontsize=10)
                        ax.set_ylabel('Nhiệt độ (°C)', fontsize=10)
                        ax.set_title(f'Xu hướng Nhiệt độ - {country}', fontsize=12, fontweight='bold')
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Biểu đồ theo tháng
                        fig, ax = plt.subplots(figsize=(14, 6))
                        ax.plot(monthly_stats['date'], monthly_stats['temperature_celsius'], 
                               marker='o', linewidth=2, markersize=4, color='darkgreen')
                        ax.set_xlabel('Tháng/Năm', fontsize=10)
                        ax.set_ylabel('Nhiệt độ TB (°C)', fontsize=10)
                        ax.set_title('Xu hướng Theo Tháng', fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Thống kê
                        st.divider()
                        st.subheader("📊 Thống kê")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("TB", f"{country_data['temperature_celsius'].mean():.2f}°C")
                        with col2:
                            st.metric("Max", f"{country_data['temperature_celsius'].max():.2f}°C")
                        with col3:
                            st.metric("Min", f"{country_data['temperature_celsius'].min():.2f}°C")
                        with col4:
                            st.metric("Std", f"{country_data['temperature_celsius'].std():.2f}°C")
                        with col5:
                            temp_range = country_data['temperature_celsius'].max() - country_data['temperature_celsius'].min()
                            st.metric("Range", f"{temp_range:.2f}°C")
                    
                    else:
                        fig, ax = plt.subplots(figsize=(14, 6))
                        ax.plot(country_data['date'], country_data[metric], 
                               marker='o', linewidth=2, markersize=4, color='steelblue')
                        
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
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("TB", f"{country_data[metric].mean():.2f}")
                        with col2:
                            st.metric("Max", f"{country_data[metric].max():.2f}")
                        with col3:
                            st.metric("Min", f"{country_data[metric].min():.2f}")
                        with col4:
                            st.metric("Std", f"{country_data[metric].std():.2f}")
                else:
                    st.warning(f"⚠️ Không có dữ liệu cho {country}")
    
    # ===== DỰ ĐOÁN BỆNH =====
    elif menu == "Dự đoán Bệnh":
        st.header("Dự đoán Tác động Sức khỏe")
        
        health_df = load_health_data()
        
        if health_df is not None:
            model_type = st.selectbox(
                "Chọn loại dự đoán:",
                ["Bệnh hô hấp", "Bệnh lây truyền qua sinh vật trung gian", "Ca nhập viện do nắng nóng"]
            )
            
            # Khởi tạo trạng thái trong session_state nếu chưa có
            if 'model_trained' not in st.session_state:
                st.session_state.model_trained = False
                st.session_state.trained_model = None

            # Nút Huấn luyện
            if st.button("Huấn luyện Mô hình", type="primary"):
                with st.spinner("Đang huấn luyện..."):
                    if model_type == "Bệnh hô hấp":
                        model, rmse, r2, X_test, y_test, y_pred = train_model(
                            health_df, ['pm25_ugm3', 'air_quality_index'], 
                            'respiratory_disease_rate', 'linear'
                        )
                    elif model_type == "Bệnh lây truyền qua sinh vật trung gian":
                        model, rmse, r2, X_test, y_test, y_pred = train_model(
                            health_df, ['temperature_celsius', 'precipitation_mm', 'heat_related_admissions'],
                            'vector_disease_risk_score', 'forest'
                        )
                    else:
                        model, rmse, r2, X_test, y_test, y_pred = train_model(
                            health_df, ['temperature_celsius', 'precipitation_mm', 'heat_wave_days', 'extreme_weather_events'],
                            'heat_related_admissions', 'linear'
                        )
                    
                    # Lưu vào session state
                    st.session_state.model_trained = True
                    st.session_state.trained_model = model
                    st.session_state.model_metrics = (r2, rmse)
                    st.session_state.test_data = (y_test, y_pred)
                    st.session_state.current_model_type = model_type

            # Nếu đã huấn luyện xong, hiển thị kết quả và phần Dự đoán mới (NẰM NGOÀI IF BUTTON)
            if st.session_state.model_trained and st.session_state.current_model_type == model_type:
                r2, rmse = st.session_state.model_metrics
                y_test, y_pred = st.session_state.test_data
                
                st.success(f"✅ Mô hình đã sẵn sàng! (R² = {r2:.4f}, RMSE = {rmse:.4f})")
                
                # Hiển thị biểu đồ đánh giá
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_title('Đồ thị Dự đoán vs Thực tế')
                st.pyplot(fig)
                
                st.divider()
                st.subheader("Nhập thông số để Dự đoán")

                # Form dự đoán để tránh việc click vào input cũng gây reload
                with st.form("prediction_form"):
                    if model_type == "Bệnh hô hấp":
                        pm25_in = st.number_input("PM2.5 (μg/m³):", 0.0, 500.0, 50.0)
                        aqi_in = st.number_input("AQI:", 0, 500, 100)
                        inputs = pd.DataFrame([[pm25_in, aqi_in]], columns=['pm25_ugm3', 'air_quality_index'])
                    
                    elif model_type == "Bệnh lây truyền qua sinh vật trung gian":
                        temp_in = st.number_input("Nhiệt độ (°C):", -20.0, 50.0, 25.0)
                        precip_in = st.number_input("Lượng mưa (mm):", 0.0, 500.0, 100.0)
                        adm_in = st.number_input("Ca nhập viện hiện tại:", 0.0, 100.0, 10.0)
                        inputs = pd.DataFrame([[temp_in, precip_in, adm_in]], columns=['temperature_celsius', 'precipitation_mm', 'heat_related_admissions'])
                    
                    else:
                        temp_in = st.number_input("Nhiệt độ (°C):", -20.0, 50.0, 30.0)
                        precip_in = st.number_input("Lượng mưa (mm):", 0.0, 500.0, 50.0)
                        heat_in = st.number_input("Số ngày nắng nóng:", 0, 30, 5)
                        ext_in = st.number_input("Sự kiện thời tiết cực đoan:", 0, 10, 1)
                        inputs = pd.DataFrame([[temp_in, precip_in, heat_in, ext_in]], columns=['temperature_celsius', 'precipitation_mm', 'heat_wave_days', 'extreme_weather_events'])

                    submit_predict = st.form_submit_button("Tính toán kết quả")

                if submit_predict:
                    prediction = st.session_state.trained_model.predict(inputs)[0]
                    st.metric("Kết quả Dự đoán", f"{prediction:.2f}%")
    
    # ===== HƯỚNG DẪN =====
    else:
        st.header("Hướng dẫn sử dụng")
        
        st.markdown("""
        ### Cách sử dụng
        
        #### 1. Tổng quan
        - Thông tin về dự án và dữ liệu
        - Thống kê cơ bản
        
        #### 2. Phân tích & Báo cáo
        - **Tổng quan**: Khám phá dữ liệu
        - **Báo cáo**: Phát hiện chính và biểu đồ
        - **Tương quan**: Mối quan hệ giữa các biến
        - **Hiệu suất**: Đánh giá mô hình
        - **Xu hướng**: Xu hướng theo thời gian
        
        #### 3. Dự đoán Bệnh
        - Chọn loại bệnh
        - Huấn luyện mô hình ML
        - Dự đoán với dữ liệu mới
        
        ### Lưu ý
        - Dữ liệu được cập nhật định kỳ
        - Sử dụng Machine Learning
        - Kết quả mang tính tham khảo
        - Biểu đồ đã tách riêng
        """)

if __name__ == "__main__":
    main()
