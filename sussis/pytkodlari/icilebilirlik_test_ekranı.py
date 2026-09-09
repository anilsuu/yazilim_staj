# -*- coding: utf-8 -*-
"""
@author: nilsu
"""

import streamlit as st
import pandas as pd
import base64
import joblib
import time
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def guvenli_resim_goster(resim_yolu):
    try:
        st.image(resim_yolu, width='stretch')
    except TypeError:
        try:
            st.image(resim_yolu, use_container_width=True)
        except TypeError:
            st.image(resim_yolu, use_column_width=True)

try:
    model_path = os.path.join(BASE_DIR, "rf_kural_su_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler_kural.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model dosyaları (pkl) bulunamadı! Lütfen eğitim kodunuzu çalıştırdığınıza ve aynı klasörde olduğunuzdan emin olun.")

if 'analiz_tamamlandi' not in st.session_state:
    st.session_state.analiz_tamamlandi = False
    st.session_state.sonuc = 0

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

bg_path = os.path.join(BASE_DIR, "suweb2.webp")
img_base64 = get_base64_of_bin_file(bg_path)

if img_base64:
    custom_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Sidebar (Yan Menü) Blur ve Arka Plan Stili */
    [data-testid="stSidebar"] {{
        background-color: rgba(11, 61, 98, 0.75) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *, 
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] span {{
        color: #F8F8FF !important;
    }}

    [data-testid="stHorizontalBlock"]:has(#su_resmi):has(#su_formu) {{
        align-items: stretch !important;
    }}
    [data-testid="column"]:has(#su_resmi),
    [data-testid="stColumn"]:has(#su_resmi) {{
        display: flex;
        flex-direction: column;
        margin-top: 1rem; 
        margin-bottom: 1rem; 
    }}
    [data-testid="column"]:has(#su_resmi) > div,
    [data-testid="stColumn"]:has(#su_resmi) > div,
    [data-testid="column"]:has(#su_resmi) [data-testid="stImage"],
    [data-testid="stColumn"]:has(#su_resmi) [data-testid="stImage"] {{
        height: 100% !important;
        display: flex;
    }}
    [data-testid="column"]:has(#su_resmi) img,
    [data-testid="stColumn"]:has(#su_resmi) img {{
        height: 100% !important;
        object-fit: cover !important; 
        border-radius: 25px; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
    }}
    [data-testid="column"]:has(#su_formu),
    [data-testid="stColumn"]:has(#su_formu) {{
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(12px) !important;            
        -webkit-backdrop-filter: blur(12px) !important; 
        border-radius: 25px;                                        
        border: 1px solid rgba(255, 255, 255, 0.4); 
        padding: 2rem 3rem;  
        margin-top: 1rem; 
        margin-bottom: 1rem; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
        display: flex; 
        flex-direction: column;
    }}
    
    /* Header şeffaf yapıldı */
    [data-testid="stHeader"], .stApp > header {{
        background-color: transparent !important;
        background: transparent !important;
    }}

    /* Sağ üst menü ve deploy butonları gizlendi */
    .stDeployButton, [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, footer {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* Sidebar ok butonu */
    [data-testid="collapsedControl"] {{
        background: #0b3d62 !important;
        border-radius: 12px !important;
        margin: 10px !important;
        width: 46px !important;
        height: 46px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
        z-index: 999999 !important;
    }}

    [data-testid="collapsedControl"]:hover {{
        background: #145a86 !important;
        transform: scale(1.05);
    }}

    h1, h2, h4, p, label, .stMarkdown {{
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8); 
    }}
    .stSlider [data-testid="stThumbValue"] {{
        color: #ffffff !important;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
        color: #800000 !important;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        transform: scale(1.02);
        box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
    }}
    @media (max-width: 768px) {{
        [data-testid="stHorizontalBlock"]:has(#su_resmi):has(#su_formu) {{
            display: grid !important;
            grid-template-columns: 1fr !important;
        }}
        [data-testid="column"]:has(#su_resmi),
        [data-testid="stColumn"]:has(#su_resmi),
        [data-testid="column"]:has(#su_formu),
        [data-testid="stColumn"]:has(#su_formu) {{
            grid-column: 1 / 2 !important;
            grid-row: 1 / 2 !important;
            width: 100% !important; 
        }}
        [data-testid="column"]:has(#su_formu),
        [data-testid="stColumn"]:has(#su_formu) {{
            z-index: 10 !important; 
            width: 95% !important; 
            margin: auto !important; 
            padding: 1.5rem !important; 
        }}
        [data-testid="column"]:has(#su_resmi),
        [data-testid="stColumn"]:has(#su_resmi) {{
            z-index: 1 !important; 
        }}
        [data-testid="column"]:has(#su_resmi) img,
        [data-testid="stColumn"]:has(#su_resmi) img {{
            min-height: 90vh !important; 
        }}
        .block-container {{
            max-width: 95% !important;
            padding-top: 3rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

sol_kolon, sag_kolon = st.columns([1, 1])

with sol_kolon:
    st.markdown("<div id='su_resmi'></div>", unsafe_allow_html=True)
    try:
        su_resmi_path = os.path.join(BASE_DIR, "suweb.webp")
        guvenli_resim_goster(su_resmi_path)
    except FileNotFoundError:
        st.info("Sol tarafta gösterilecek resim bulunamadı. Lütfen dosya adını güncelleyin.")

with sag_kolon:
    st.markdown("<div id='su_formu'></div>", unsafe_allow_html=True)
    
    if not st.session_state.analiz_tamamlandi:
        st.markdown("<h3 style='color : #ADD8E6 !important; text-shadow : 1px 1px 3px rgba (0,0,0,0.8); margin-bottom : 0px;'>🚰 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
        st.markdown("### Suyun Bileşen Analizi")
        st.markdown("<hr style='border:1px solid white'>" , unsafe_allow_html=True)
        
        with st.form(key="su_analiz_formu", border=False):
            form_sol, form_sag = st.columns(2)
            
            with form_sol:
                ph = float(st.number_input("💧 pH Seviyesi", step=1.0, min_value=0.227, max_value=14.0)) 
                Solids = float(st.number_input("🧊 Solids (Katılar)", step=1.0, min_value=320.9, max_value=30000.0))
                Sulfate = float(st.number_input("🟣 Sulfate(Sülfat)", step=1.0, min_value=129.0, max_value=400.0))
                Organic_carbon = float(st.number_input("💬 Organic Carbon", step=1.0, min_value=2.2, max_value=8.0))
                Turbidity = float(st.number_input("🌀 Turbidity(Bulanıklık)", step=1.0, min_value=1.45, max_value=5.0))

            with form_sag:
                Hardness = float(st.number_input("🪨 Hardness(Sertlik)", step=1.0, min_value=73.4, max_value=300.0))
                Chloramines = float(st.number_input("🧪 Chloramines", step=1.0, min_value=1.39, max_value=10.0))
                Conductivity = float(st.number_input("⚡ Conductivity(İletkenlik)", step=1.0, min_value=201.6, max_value=500.0))
                Trihalomethanes = float(st.number_input("🟠 Trihalometanlar", step=1.0, min_value=8.57, max_value=100.0))
                
                st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            analiz_baslat = st.form_submit_button("Analiz Et 🚀", width='stretch')
            
        if analiz_baslat:
            with st.spinner("Su kalite verileri analiz ediliyor..."):
                time.sleep(3) 
                
                input_data = pd.DataFrame({
                    "ph": [ph], "Hardness": [Hardness], "Solids": [Solids],
                    "Chloramines": [Chloramines], "Sulfate": [Sulfate],
                    "Conductivity": [Conductivity], "Organic_carbon": [Organic_carbon],
                    "Trihalomethanes": [Trihalomethanes], "Turbidity": [Turbidity]
                })
                
                try:
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)
                    
                    st.session_state.sonuc = int(prediction[0])
                    st.session_state.analiz_tamamlandi = True
                    
                except Exception as e:
                    st.error(f"Tahmin sırasında bir hata oluştu: {e}")
            
            st.rerun()
                
    else:
        st.markdown("### 📊 ANALİZ SONUCU")
        st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
            
        if st.session_state.sonuc == 1: 
            st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
        else:
            st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        if st.button("Yeni Test Yap 🔄", key="yeni_test_buton", width='stretch'):
            st.session_state.analiz_tamamlandi = False
            st.rerun()