# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:10:52 2026

@author: nilsu
"""

import streamlit as st
import pandas as pd
import joblib
import streamlit as st

# Sayfa ayarlarını geniş modda ve modern bir başlıkla başlatın
st.set_page_config(page_title="Su Kalite Analizi", layout="wide")

# Modern ve koyu bir "su" teması CSS'i
page_bg_css = """
<style>
/* Arka plan degrade (gradient) rengi */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
}

/* Başlık ve metin renklerini beyaza çevirme */
h1, h2, h3, p, label {
    color: #ffffff !important;
}

/* Butonu modernleştirme */
.stButton>button {
    background-color: #00d2ff;
    color: #000000;
    border-radius: 20px;
    border: none;
    padding: 10px 24px;
    font-weight: bold;
    transition: all 0.3s ease 0s;
}
.stButton>button:hover {
    background-color: #3a7bd5;
    color: white;
    box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)


#icilebilirlik su testi web sitesinin iskeletini hazırlamak

model = joblib.load(r"C:\Users\nilsu\OneDrive\Masaüstü\yazilim_staj\sussis\pytkodlari\rf_kural_su_model.pkl")
# Sayfa ayarlarını geniş modda ve modern bir başlıkla başlatın

# st.set_page_config(
#     page_title="SASKİ Su İçilebilirlik Testi",layout="wide",
#     page_icon=r"C:\Users\nilsu\OneDrive\Masaüstü\yazilim_staj\sussis\pytkodlari\test_icon.png"
#     )

st.title=("SASKİ Su İçilebilirlik Testi")
st.write=("Lütfen tahmin için değerleri giriniz.")


#Girilecek Veri tiplerini belirtmek,minimum değerlerin girilmesi.Help (?) açıklamasının infonun verilmesi

# ph = float(st.number_input("ph",step=1.0,min_value=0.22749905,max_value=14.0,help="Suyun ph değerini giriniz.")) 
# Hardness=float(st.number_input("Hardness(Sertlik)",step=1.0,min_value=73.49223369,help="Sertlik(Hardness) değerini giriniz."))
# Solids=float(st.number_input("Solids (Katılar)",step=1.0,min_value=320.9426113,help="Solids(Katılar) değerini giriniz."))
# Chloramines=float(st.number_input("Chloramines(Kloramin)",step=1.0,min_value=1.390870905,help="Chloramines(Kloramin) değerini giriniz."))
# Sulfate=float(st.number_input("Sulfate(Sülfat)",step=1.0,min_value=129.0,help="Sulfate(Sülfat) değerini giriniz."))
# Conductivity=float(st.number_input("Conductivity(İletkenlik)",step=1.0,min_value=201.6197368,help="20°C Conductivity (İletkenlik) değerini giriniz."))
# Organic_carbon=float(st.number_input("Organic_carbon (Organik karbon)",step=1.0,min_value=2.2,help="Organic_carbon (Organik karbon) değerini giriniz."))
# Trihalomethanes=float(st.number_input("Trihalomethanes(Trihalometanlar)",step=1.0,min_value=8.577012933,help="Trihalomethanes(Trihalometanlar) değerini giriniz."))
# Turbidity=float(st.number_input("Turbidity(Bulanıklık)",step=1.0,min_value=1.45,help="Turbidity(Bulanıklık) değerini giriniz."))




# Modern Slider Kullanımları (Klavye yerine fare/dokunmatik ile kaydırma)
col1, col2 = st.columns(2) # Ekranı iki sütuna bölerek şık bir görünüm elde edin

with col1:
    ph = st.slider("💧 pH Seviyesi", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.slider("🪨 Sertlik (Hardness)", min_value=0.0, max_value=400.0, value=150.0, step=1.0)
    
with col2:
    chloramines = st.slider("🧪 Kloramin", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
    # Diğer parametrelerinizi buraya ekleyin...
   
# input_data = pd.DataFrame({
#     "ph": [ph],
#     "Hardness": [Hardness],
#     "Solids": [Solids],
#     "Chloramines": [Chloramines],
#     "Sulfate": [Sulfate],
#     "Conductivity": [Conductivity],
#     "Organic_carbon": [Organic_carbon],
#     "Trihalomethanes": [Trihalomethanes],
#     "Turbidity": [Turbidity],
   
# })

# if st.button("--Su İçilebilir mi Test Et--"):
#     prediction = model.predict(input_data)
#     sonuc = int(prediction[0])
    
#     if sonuc == 1:
#         st.success("✅ Tahmin: Su İçilebilir")
#     else:
#         st.error("❌ Tahmin: Su İçilemez (Güvenli Değil)")
#         tab1, tab2 = st.tabs(["📊 Parametre Girişi", "📈 Yapay Zeka Analizi"])



    tab1, tab2 = st.tabs(["📊 Parametre Girişi", "📈 Yapay Zeka Analizi"])

with tab1:
    st.markdown("### Lütfen su değerlerini kaydırarak belirleyin")
    # Yukarıdaki slider'ları buraya koyabilirsiniz
    
with tab2:
    st.markdown("### Sonuç Ekranı")
    if st.button("Analizi Başlat 🚀"):
        st.success("✅ Su İçilebilir!")