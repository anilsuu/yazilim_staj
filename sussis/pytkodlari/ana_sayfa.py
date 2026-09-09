# -*- coding: utf-8 -*-
"""
@author: nilsu
"""

import streamlit as st
import base64
import os

st.set_page_config(
    page_title="SASKİ Su Analizi",
    layout="wide",
    page_icon="🚰",
    initial_sidebar_state="collapsed"
)
st.set_option("client.toolbarMode", "viewer")

# Mevcut dosyanın bulunduğu tam dizini alma
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# İçilebilirlik test sayfası yolu (Dinamik yol kullanıldı)

test_sayfasi_yolu = os.path.join(BASE_DIR, "icilebilirlik_test_ekranı.py")
sayfa_test = st.Page(test_sayfasi_yolu, title="İçilebilirlik Testi", icon="💧")


def karsilama_sayfasi():
    
    # Arka plan resmi yükleme fonksiyonu
    def get_base64_of_bin_file(bin_file):
        try:
            with open(bin_file, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            return ""

    # Arka plan görseli için dinamik yol
    bg_path = os.path.join(BASE_DIR, "doga_kaynak.webp")
    img_base64 = get_base64_of_bin_file(bg_path)

    # CSS Özelleştirmeleri
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

        /* Sağ Kolon (İçerik) - Buzlu Cam Efekti */
        [data-testid="column"]:has(#ana_sayfa_icerik) {{
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(12px) !important; 
            -webkit-backdrop-filter: blur(12px) !important; 
            border-radius: 25px;                               
            border: 1px solid rgba(255, 255, 255, 0.4); 
            padding: 2rem 3rem;  
            margin-top: 1rem; 
            margin-bottom: 1rem; 
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
            display:flex; 
            flex-direction: column;
            justify-content: space-between; /* İçeriği yukarı, butonu aşağı iter */
        }}

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
        h1, h2, h3, h4, p, label, .stMarkdown {{
            color: #F8F8FF !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8); 
        }}

        /* Sol Kolon (Resimler) Tasarımı */
        [data-testid="column"]:has(#sol_gorseller) img {{
            width: 70% !important; /* Görsellerin boyutunu daralttım */
            margin: 0 auto 1.5rem auto !important; /* Ortala */
            display: block !important;
            border-radius: 15px; 
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            
        }}

        /* Buton Tasarımı */
        .stButton>button {{
            background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
            color: #191970 !important;
            font-weight: bold;
            padding: 1rem;
            border-radius: 15px;
        }}
                                        
        .stButton>button:hover {{
            transform: scale(1.02);
            box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
        }}
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    # Sayfa yerleşimi 
    # Sol kolon dar (1), sağ kolon geniş (2.5)
    sol_kolon, sag_kolon = st.columns([1, 2.5])

#  Sol taraf görseller
    with sol_kolon:
        st.markdown("<div id='sol_gorseller'></div>", unsafe_allow_html=True)
        try:
            # use_container_width yerine use_column_width olarak düzeltildi
            st.image(os.path.join(BASE_DIR, "slider_1.jpg"), use_container_width=True)
            st.image(os.path.join(BASE_DIR, "saski.jpg"), use_container_width=True)
            st.image(os.path.join(BASE_DIR, "sakaryabelediye.jpg"), use_container_width=True)
            
        except FileNotFoundError:
            st.info("Lütfen görsellerin doğru klasörde olduğundan emin olun.")
    #  Sağ kolon (Tanıtım Metni ve Buton)
    with sag_kolon:
        st.markdown("<div id='ana_sayfa_icerik'></div>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: #ADD8E6 !important;'>💧SASKİ Su Analiz Sistemi💧</h1>", unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
        
        st.markdown("""
        ### Su Analiz Sistemine Hoş Geldiniz
        İçilebilir suyun belirli özelliklerde olması gerekir:
        * Berrak ve renksiz olmalıdır.
        * Kokusuz ve tatsız olmalıdır.
        * İçinde hastalık yapıcı bakteriler ve toksinler bulunmamalıdır.
        * İçinde zararlı kimyasallar, ağır metaller ve diğer kirleticiler bulunmamalıdır.
          """)
        st.markdown("""Test ekranında gireceğiniz parametreler ile suyun içilebilirlik testi yapılabilir.""")
        
        st.markdown(""" :blue-background[Güvenilir Sonuçlar:] Model algoritmaları, laboratuvar verileriyle eğitilmiştir.""")
        st.markdown(""" :blue-background[Hızlı Analiz:] Değerlerini su içilebilirlik analiz formuna girerek saniyeler içinde analiz sonucuna ulaşabilirsiniz.""")
       
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        # Son görsel için düzeltme
        try:
            st.image(os.path.join(BASE_DIR, "ai_nedensuicmeliyiz.webp"), use_container_width=True)
        except FileNotFoundError:
            pass
        
        # Test Sayfasına Yönlendiren Buton 
        if st.button("Hemen Analiz Testine Başla 🔍 ", use_container_width=True):
            st.switch_page(sayfa_test)


# sidebar bilgileri
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">

        <h4>📞 SASKİ İletişim</h4>
        <p>
        <b>Sakarya Büyükşehir Belediyesi</b><br>
        SASKİ Genel Müdürlüğü
        </p>

        <hr>

        <h4>📝 Görüş ve Öneriler</h4>
        <p>
        Görüş ve önerileriniz için<br>
        <a href="mailto:su.analizi.sistemi@gmail.com">
        su.analizi.sistemi@gmail.com
        </a>
        </p>

        <hr>

        <p style="font-size: 13px;">
        💻 <b>Bilişim Mühendisi</b><br>
        Azra Nilsu tarafından geliştirilmiştir.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

sayfa_ana = st.Page(karsilama_sayfasi, title="Ana Sayfa", icon="🏠")

pg = st.navigation([sayfa_ana, sayfa_test])
pg.run()