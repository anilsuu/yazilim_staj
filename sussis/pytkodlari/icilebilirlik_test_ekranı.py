# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:10:52 2026

@author: nilsu
"""

import streamlit as st
import pandas as pd
import joblib

#icilebilirlik su testi web sitesinin iskeletini hazırlamak

model = joblib.load(r"C:\Users\nilsu\OneDrive\Masaüstü\yazilim_staj\sussis\pytkodlari\rf_kural_su_model.pkl")

st.set_page_config(
    page_title="SASKİ Su İçilebilirlik Testi",
    page_icon=r"C:\Users\nilsu\OneDrive\Masaüstü\yazilim_staj\sussis\pytkodlari\test_icon.png"
    )

st.title=("SASKİ Su İçilebilirlik Testi")
st.write=("Lütfen tahmin için değerleri giriniz.")


#Girilecek Veri tiplerini belirtmek,minimum değerlerin girilmesi.Help (?) açıklamasının infonun verilmesi

ph = float(st.number_input("ph",step=1.0,min_value=0.22749905,max_value=14.0,help="Suyun ph değerini giriniz.")) 
Hardness=float(st.number_input("Hardness(Sertlik)",step=1.0,min_value=73.49223369,help="Sertlik(Hardness) değerini giriniz."))
Solids=float(st.number_input("Solids (Katılar)",step=1.0,min_value=320.9426113,help="Solids(Katılar) değerini giriniz."))
Chloramines=float(st.number_input("Chloramines(Kloramin)",step=1.0,min_value=1.390870905,help="Chloramines(Kloramin) değerini giriniz."))
Sulfate=float(st.number_input("Sulfate(Sülfat)",step=1.0,min_value=129.0,help="Sulfate(Sülfat) değerini giriniz."))
Conductivity=float(st.number_input("Conductivity(İletkenlik)",step=1.0,min_value=201.6197368,help="20°C Conductivity (İletkenlik) değerini giriniz."))
Organic_carbon=float(st.number_input("Organic_carbon (Organik karbon)",step=1.0,min_value=2.2,help="Organic_carbon (Organik karbon) değerini giriniz."))
Trihalomethanes=float(st.number_input("Trihalomethanes(Trihalometanlar)",step=1.0,min_value=8.577012933,help="Trihalomethanes(Trihalometanlar) değerini giriniz."))
Turbidity=float(st.number_input("Turbidity(Bulanıklık)",step=1.0,min_value=1.45,help="Turbidity(Bulanıklık) değerini giriniz."))


input_data = pd.DataFrame({
    "ph": [ph],
    "Hardness": [Hardness],
    "Solids": [Solids],
    "Chloramines": [Chloramines],
    "Sulfate": [Sulfate],
    "Conductivity": [Conductivity],
    "Organic_carbon": [Organic_carbon],
    "Trihalomethanes": [Trihalomethanes],
    "Turbidity": [Turbidity],
   
})
if st.button("--Su İçilebilir mi Test Et--"):
    prediction = model.predict(input_data)
    sonuc = int(prediction[0])
    
    if sonuc == 1:
        st.success("✅ Tahmin: Su İçilebilir")
    else:
        st.error("❌ Tahmin: Su İçilemez (Güvenli Değil)")
        