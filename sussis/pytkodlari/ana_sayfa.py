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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def guvenli_resim_goster(resim_yolu):
    try:
        st.image(resim_yolu, width='stretch') # Yeni sürüm
    except TypeError:
        try:
            st.image(resim_yolu, use_container_width=True) # Geçiş sürümü
        except TypeError:
            st.image(resim_yolu, use_column_width=True) # Eski sürüm

test_sayfasi_yolu = os.path.join(BASE_DIR, "icilebilirlik_test_ekranı.py")
sayfa_test = st.Page(test_sayfasi_yolu, title="İçilebilirlik Testi", icon="💧")

def karsilama_sayfasi():
    def get_base64_of_bin_file(bin_file):
        try:
            with open(bin_file, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            return ""

    bg_path = os.path.join(BASE_DIR, "doga_kaynak.webp")
    img_base64 = get_base64_of_bin_file(bg_path)

    if img_base64:
        custom_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment:fixed;
        }}
        [data-testid="column"]:has(#ana_sayfa_icerik),
        [data-testid="stColumn"]:has(#ana_sayfa_icerik) {{
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
            justify-content: space-between; 
        }}
        
        /* Header (Üst Şerit) Kutusunu gizlemeden, sadece şeffaf yapalım */
        [data-testid="stHeader"], .stApp > header {{
            background-color: transparent !important;
            background: transparent !important;
        }}
        
        /* Sağ üstteki Menü ve Deploy butonlarını giliyorum. */
        .stDeployButton, [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu {{
            display: none !important;
            visibility: hidden !important;
        }}
        
        [data-testid="collapsedControl"] {{
            display: flex !important;
            visibility: visible !important;
            background-color: rgba(0, 0, 0, 0.5) !important;
            border-radius: 50% !important;
            margin: 10px !important;
            z-index: 999999 !important; /* Her şeyin en üstünde olmasını garantiler */
        }}
        [data-testid="collapsedControl"] svg {{
            fill: #ffffff !important;
            color: #ffffff !important;
            stroke: #ffffff !important;
            width: 24px !important;
            height: 24px !important;
        }}

        footer {{
            display: none !important;
        }}
        
        h1, h2, h3, h4, p, label, .stMarkdown {{
            color: #F8F8FF !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8); 
        }}
        [data-testid="column"]:has(#sol_gorseller) img,
        [data-testid="stColumn"]:has(#sol_gorseller) img {{
            width: 70% !important; 
            margin: 0 auto 1.5rem auto !important; 
            display: block !important;
            border-radius: 15px; 
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
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
        
        /* Mobil ve Küçük Ekranlar İçin Responsive Ayarları */
        @media (max-width: 768px) {{
        [data-testid="stHorizontalBlock"]:has(#sol_gorseller):has(#ana_sayfa_icerik) {{
            display: grid !important;
            grid-template-columns: 1fr !important;
        }}
        
        [data-testid="column"]:has(#sol_gorseller),
        [data-testid="stColumn"]:has(#sol_gorseller),
        [data-testid="column"]:has(#ana_sayfa_icerik),
        [data-testid="stColumn"]:has(#ana_sayfa_icerik) {{
            width: 100% !important;
            max-width: 100% !important;
        }}

        /* Metin kutusunun mobildeki boşluklarını ve kenar paylarını rahatlatıyoruz */
        [data-testid="column"]:has(#ana_sayfa_icerik),
        [data-testid="stColumn"]:has(#ana_sayfa_icerik) {{
            padding: 1.5rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }}

       /* Mobilde resimlerin devasa olmasını engellemek için sınırlandırıyoruz ve ortalıyoruz */
        [data-testid="column"]:has(#sol_gorseller) img,
        [data-testid="stColumn"]:has(#sol_gorseller) img {{
            width: 60% !important;
            max-width: 250px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: 1rem !important;
            display: block !important;
        }}

        /* Sayfa dış kenar boşluklarını mobile uyumlu hale getiriyoruz */
        .block-container {{
            max-width: 100% !important;
            padding-top: 3rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    sol_kolon, sag_kolon = st.columns([1, 2.5])

    with sol_kolon:
        st.markdown("<div id='sol_gorseller'></div>", unsafe_allow_html=True)
        try:
            guvenli_resim_goster(os.path.join(BASE_DIR, "slider_1.jpg"))
            guvenli_resim_goster(os.path.join(BASE_DIR, "saski.jpg"))
            guvenli_resim_goster(os.path.join(BASE_DIR, "sakaryabelediye.jpg"))
        except FileNotFoundError:
            st.info("Lütfen görsellerin doğru klasörde olduğundan emin olun.")
            
    with sag_kolon:
        st.markdown("<div id='ana_sayfa_icerik'></div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #ADD8E6 !important;'>SASKİ Su Analiz Sistemi💧</h1>", unsafe_allow_html=True)
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
        
        try:
            guvenli_resim_goster(os.path.join(BASE_DIR, "ai_nedensuicmeliyiz.webp"))
        except FileNotFoundError:
            pass
        
        if st.button("Hemen Analiz Testine Başla 🔍 ", width='stretch'):
            st.switch_page(sayfa_test)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
        <h4>📞 SASKİ İletişim</h4>
        <p><b>Sakarya Büyükşehir Belediyesi</b><br>SASKİ Genel Müdürlüğü</p>
        <hr>
        <h4>📝 Görüş ve Öneriler</h4>
        <p>Görüş ve önerileriniz için<br><a href="mailto:su.analizi.sistemi@gmail.com">su.analizi.sistemi@gmail.com</a></p>
        <hr>
        <p style="font-size: 13px;">💻 <b>Bilişim Mühendisi</b><br>Azra Nilsu tarafından geliştirilmiştir.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

sayfa_ana = st.Page(karsilama_sayfasi, title="Ana Sayfa", icon="🏠")
pg = st.navigation([sayfa_ana, sayfa_test])
pg.run()