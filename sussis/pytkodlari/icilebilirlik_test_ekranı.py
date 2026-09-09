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

# Mevcut dosyanın bulunduğu tam dizin (Bulut ortamındaki yol hatalarını önler)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modeli medyan_puanlamalıdan ve scaleri yükledim 
try:
    model_path = os.path.join(BASE_DIR, "rf_kural_su_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler_kural.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model dosyaları (pkl) bulunamadı! Lütfen eğitim kodunuzu çalıştırdığınıza ve aynı klasörde olduğunuza emin olun.")

# session_state - Formun kaybolması için
if 'analiz_tamamlandi' not in st.session_state:
    st.session_state.analiz_tamamlandi = False
    st.session_state.sonuc = 0

# Web arka sayfasına su-bardak resmi yükleme 
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# Tüm sayfanın arka planı için kullanılacak resimi yükleme
bg_path = os.path.join(BASE_DIR, "suweb2.webp")
img_base64 = get_base64_of_bin_file(bg_path)

# CSS Özelleştirme kodları 
if img_base64:
    custom_css = f"""
    <style>
    /* Arka plan resmini tam ekran yapma */
    .stApp {{
        background-image: url("data:image/webp;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment:fixed;
    }}

    /* Sadece ana kolonları eşitlemek için özel ID'li seçiciler */
    [data-testid="stHorizontalBlock"]:has(#su_resmi):has(#su_formu) {{
        align-items: stretch !important;
    }}
     
        
    /* Su Bardağı Resmi (Sol) - Versiyon Uyumluluğu Eklendi */
    [data-testid="column"]:has(#su_resmi),
    [data-testid="stColumn"]:has(#su_resmi),
    [data-testid="stVerticalBlock"]:has(#su_resmi) {{
        display: flex;
        flex-direction: column;
        margin-top: 1rem; 
        margin-bottom: 1rem; 
    }}

    /* Resmi kırparak alanı doldurmasını sağla */
    [data-testid="column"]:has(#su_resmi) > div,
    [data-testid="stColumn"]:has(#su_resmi) > div,
    [data-testid="stVerticalBlock"]:has(#su_resmi) > div,
    [data-testid="column"]:has(#su_resmi) [data-testid="stImage"],
    [data-testid="stColumn"]:has(#su_resmi) [data-testid="stImage"],
    [data-testid="stVerticalBlock"]:has(#su_resmi) [data-testid="stImage"] {{
        height: 100% !important;
        display: flex;
    }}
        
    [data-testid="column"]:has(#su_resmi) img,
    [data-testid="stColumn"]:has(#su_resmi) img,
    [data-testid="stVerticalBlock"]:has(#su_resmi) img {{
        height: 100% !important;
        object-fit: cover !important; 
        border-radius: 25px; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
    }}
        
        
    /* Form Kolonu (Sağ) - Buzlu Cam Efekti - Versiyon Uyumluluğu */
    [data-testid="column"]:has(#su_formu),
    [data-testid="stColumn"]:has(#su_formu),
    [data-testid="stVerticalBlock"]:has(#su_formu) {{
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(12px) !important;            /*Kutunun buzlu olması 12px*/
        -webkit-backdrop-filter: blur(12px) !important; 
        border-radius: 25px;                               /*Kutunun köşelerini yuvarlatmak*/
        border: 1px solid rgba(255, 255, 255, 0.4); 
        padding: 2rem 3rem;  
        margin-top: 1rem; 
        margin-bottom: 1rem; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
        display:flex; 
        flex-direction: column;
    }}
    
    /* streamlit deploy , clear cache olan toolbarı kaldırma */

    [data-testid="stHeader"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
    }}

    .stApp > header {{
        display: none !important;
        visibility: hidden !important;
    }}

    .stAppToolbar {{
        display: none !important;
        visibility: hidden !important;
    }}

    [data-testid="stToolbar"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    .stDeployButton {{
        display: none !important;
        visibility: hidden !important;
    }}

    [data-testid="stDecoration"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    [data-testid="stStatusWidget"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    #MainMenu {{
        display: none !important;
        visibility: hidden !important;
        background: #6B8E23 !important;
        backdrop-filter: blur(12px) !important;
    }}

    footer {{
        display: none !important;
        visibility: hidden !important;
        
    }}
        
    /* Yazı renkleri ve gölgeleri */
    h1,h2, h4, p, label, .stMarkdown {{
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8); 
        
        /*Shadow + ile başlıyorsa sağa doğru,- ile başlıyorsa sola doğru*/
    }}
    
    .stSlider [data-testid="stThumbValue"] {{
        color: #ffffff !important;
    }}

    /* Buton Tasarımı */
    .stButton>button {{
        background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
        color: #800000 !important;
        font-weight: bold;
    }}
                                        
    .stButton>button:hover {{
        transform: scale(1.02);
        box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
    }}

    /* Display için css ayarları - Mobil görünüm güncellemeleri */
    @media (max-width: 768px) {{
        /* Sadece ana bloğu Grid yap (içerideki 2'li form kolonları etkilenmesin) */
        [data-testid="stHorizontalBlock"]:has(#su_resmi):has(#su_formu) {{
            display: grid !important;
            grid-template-columns: 1fr !important;
        }}
        
        [data-testid="column"]:has(#su_resmi),
        [data-testid="stColumn"]:has(#su_resmi),
        [data-testid="stVerticalBlock"]:has(#su_resmi),
        [data-testid="column"]:has(#su_formu),
        [data-testid="stColumn"]:has(#su_formu),
        [data-testid="stVerticalBlock"]:has(#su_formu) {{
            grid-column: 1 / 2 !important;
            grid-row: 1 / 2 !important;
            width: 100% !important; 
        }}

        /* Form üste gelecek şekilde ayarla */
        [data-testid="column"]:has(#su_formu),
        [data-testid="stColumn"]:has(#su_formu),
        [data-testid="stVerticalBlock"]:has(#su_formu) {{
            z-index: 10 !important; 
            width: 95% !important; 
            margin: auto !important; 
            padding: 1.5rem !important; 
        }}
        
        /* Soldaki resim kutusunu altta kalacak şekilde ayarla */
        [data-testid="column"]:has(#su_resmi),
        [data-testid="stColumn"]:has(#su_resmi),
        [data-testid="stVerticalBlock"]:has(#su_resmi) {{
            z-index: 1 !important; 
        }}
            
        [data-testid="column"]:has(#su_resmi) img,
        [data-testid="stColumn"]:has(#su_resmi) img,
        [data-testid="stVerticalBlock"]:has(#su_resmi) img {{
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


# Sayfanın yerleşimi (Sol kolon boşluk, sağ kolon veri girişi)
sol_kolon, sag_kolon = st.columns([1, 1])

# sol kolon (Su Bardağı Resmi) 
with sol_kolon:
    
    st.markdown("<div id='su_resmi'></div>", unsafe_allow_html=True)
    
    try:
        su_resmi_path = os.path.join(BASE_DIR, "suweb.webp")
        st.image(su_resmi_path, use_container_width=True)
        
    except FileNotFoundError:
        st.info("Sol tarafta gösterilecek resim bulunamadı. Lütfen dosya adını güncelleyin.")

# --- sağ kolon (Form ve Sonuçlar) ---
with sag_kolon:
    
    st.markdown("<div id='su_formu'></div>", unsafe_allow_html=True)
    
    # Analiz yapılmadıysa formu göster
    if not st.session_state.analiz_tamamlandi:
        st.markdown("<h3 style='color : #ADD8E6 !important; text-shadow : 1px 1px 3px rgba (0,0,0,0.8); margin-bottom : 0px;'>🚰 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
        st.markdown("### Suyun Bileşen Analizi")
        st.markdown("<hr style='border:1px solid white'>" , unsafe_allow_html=True)
        
        with st.form(key="su_analiz_formu", border=False):
            form_sol, form_sag = st.columns(2)
            
            with form_sol:
                ph = float(st.number_input("💧 pH Seviyesi", step=1.0, min_value=0.227, max_value=14.0)) 
                Solids = float(st.number_input("🧊 Solids (Katılar)", step=1.0, min_value=320.9,max_value=30000.0))
                Sulfate = float(st.number_input("🟣 Sulfate(Sülfat)", step=1.0, min_value=129.0,max_value=400.0))
                Organic_carbon = float(st.number_input("💬 Organic Carbon", step=1.0, min_value=2.2,max_value=8.0))
                Turbidity = float(st.number_input("🌀 Turbidity(Bulanıklık)", step=1.0, min_value=1.45,max_value=5.0))

            with form_sag:
                Hardness = float(st.number_input("🪨 Hardness(Sertlik)", step=1.0, min_value=73.4,max_value=300.0))
                Chloramines = float(st.number_input("🧪 Chloramines", step=1.0, min_value=1.39,max_value=10.0))
                Conductivity = float(st.number_input("⚡ Conductivity(İletkenlik)", step=1.0, min_value=201.6,max_value=500.0))
                Trihalomethanes = float(st.number_input("🟠 Trihalometanlar", step=1.0, min_value=8.57,max_value=100.0))
                
                st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            analiz_baslat = st.form_submit_button("Analiz Et 🚀", use_container_width=True)
            
        #  Bekleme ve analiz formu içindeyken yapılır
        if analiz_baslat:
            
            #  Önce analiz ediliyor yazısı çıkar, döner, sonra ekran değişir.
            with st.spinner("Su kalite verileri analiz ediliyor..."):
                time.sleep(3) 
                
                input_data = pd.DataFrame({
                    "ph": [ph], "Hardness": [Hardness], "Solids":[Solids],
                    "Chloramines":[Chloramines], "Sulfate":[Sulfate],
                    "Conductivity":[Conductivity], "Organic_carbon":[Organic_carbon],
                    "Trihalomethanes":[Trihalomethanes], "Turbidity":[Turbidity]
                })
                
                try:
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)
                    
                    # Verileri kaydet
                    st.session_state.sonuc = int(prediction[0])
                    st.session_state.analiz_tamamlandi = True
                    
                except Exception as e:
                    st.error(f"Tahmin sırasında bir hata oluştu: {e}")
            
            st.rerun()
                
    # Analiz bittikten sonra sonuç ekranı anında yüklenir
    else:
        st.markdown("### 📊 ANALİZ SONUCU")
        st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
            
        if st.session_state.sonuc == 1: 
            st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
        else:
            st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        if st.button("Yeni Test Yap 🔄", key="yeni_test_buton", use_container_width=True):
            st.session_state.analiz_tamamlandi = False
            st.rerun()