# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:25:51 2026

@author: nilsu
"""

# import streamlit as st
# import pandas as pd
# import base64
# import joblib
# import time
# import numpy as np


# # Geniş mod - layout="wide" ekranın daha büyük

# st.set_page_config(page_title="Su İçilebilirlik Analizi", layout="wide", page_icon="🚰")


# #  Sayfanın durum yönetimi - Formun kaybolması için

# if 'analiz_tamamlandi' not in st.session_state:
#     st.session_state.analiz_tamamlandi = False
#     st.session_state.sonuc = 0



# #  Web arka sayfasına su-bardak resmi yükleme 

# def get_base64_of_bin_file(bin_file):
#     try:
#         with open(bin_file, 'rb') as f:
#             return base64.b64encode(f.read()).decode()
#     except FileNotFoundError:
#         return ""



# # Tüm sayfanın arka planı için kullanılacak resim

# img_base64 = get_base64_of_bin_file("doga_kaynak.webp")




# #  CSS Özelleştirme kodları

# #  rgba(255, 255, 255, 0.4); ana renklerin ne oranda karıştırılacağını gösteriyor.

# if img_base64:
#     custom_css = f"""
#     <style>
#     /* Arka plan resmini tam ekran yapma */
#     .stApp {{
#         background-image: url("data:image/webp;base64,{img_base64}");
#         background-size: cover;
#         background-position: center;
#         background-attachment:fixed;
#     }}

#     # /* Sadece ana kolonları eşitlemek için özel ID'li seçiciler */
#     # [data-testid="stHorizontalBlock"]:has(#ana_sayfa):has(#slider) {{
#     #     align-items: stretch !important;
#     # }}
        
#     # # /*  Sonuç ekranında buzlu boyutu küçültüp ortalama */
#     # #  [data-testid="stHorizontalBlock"]:has(#su_sonuc) {{
#     # #      align-items: center !important; 
#     # #  }}
        
        
#     # /* Su Bardağı Resmi (Sol) */
#     # [data-testid="column"]:has(#ana_sayfa) {{
#     #     display: flex;
#     #     flex-direction: column;
#     #     margin-top: 1rem; 
#     #     margin-bottom: 1rem; 
#     # }}

#     # /* Resmi kırparak alanı doldurmasını sağla */
#     # [data-testid="column"]:has(#ana_sayfa) > div,
#     # [data-testid="column"]:has(#ana_sayfa) [data-testid="stImage"] {{
#     #     height: 100% !important;
#     #     display: flex;
#     # }}
        
    
#     [data-testid="column"]:has(#ana_sayfa) img {{
#         height: 80% !important;
#         object-fit: cover !important; 
#         border-radius: 25px; 
#         box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
#     }}
        
        

#     # /* Form Kolonu (Sağ) - Buzlu  */
#     # [data-testid="column"]:has(#slider){{
#     #     background: rgba(255, 255, 255, 0.15) !important;
#     #     backdrop-filter: blur(12px) !important;            /*Kutunun buzlu olması 12px*/
#     #     -webkit-backdrop-filter: blur(12px) !important; 
#     #     border-radius: 25px;                               /*Kutunun köşelerini yuvarlatmak*/
#     #     border: 1px solid rgba(255, 255, 255, 0.4); 
#     #     padding: 2rem 3rem;  
#     #     margin-top: 1rem; 
#     #     margin-bottom: 1rem; 
#     #     box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
#     #     display:flex; 
#     #     flex-direction: column;
#     # }}

#     /* Yazı renkleri ve gölgeleri */
#     h1,h2, h4, p, label, .stMarkdown {{
#         color: #ffffff !important;
#         text-shadow: 1px 1px 3px rgba(0,0,0,0.8); 
#                                                   /*Shadow + ile başlıyorsa sağa doğru,- ile başlıyorsa sola doğru*/
    
#     }}
    
#     .stSlider [data-testid="stThumbValue"] {{
#         color: #ffffff !important;
#     }}

#     /* Buton Tasarımı */
#     .stButton>button {{
#         background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
#         color: #800000 !important;
#         font-weight: bold;
#     }}
                                    
#     .stButton>button:hover {{
#         transform: scale(1.02);
#         box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
#     }}

#     /* Display için css ayarları */
#     @media (max-width: 768px) {{
#         /* Sadece ana bloğu Grid yap (içerideki 2'li form kolonları etkilenmesin) */
#         [data-testid="stHorizontalBlock"]:has(#ana_sayfa):has(#slider) {{
#             display: grid !important;
#             grid-template-columns: 1fr !important;
#         }}
        
#         [data-testid="column"]:has(#ana_sayfa),
#         [data-testid="column"]:has(#slider) {{
#             grid-column: 1 / 2 !important;
#             grid-row: 1 / 2 !important;
#             width: 100% !important; 
#         }}

#         /* Form üste gelecek şekilde ayarla */
#         [data-testid="column"]:has(#slider) {{
#             z-index: 10 !important; 
#             width: 95% !important; 
#             margin: auto !important; 
#             padding: 1.5rem !important; 
#         }}
        
#         /* Soldaki resim kutusunu altta kalacak şekilde ayarla */
#         [data-testid="column"]:has(#ana_sayfa) {{
#             z-index: 1 !important; 
#         }}
            
#         [data-testid="column"]:has(#ana_sayfa) img {{
#             min-height: 90vh !important; 
#         }}
            
#     }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)




# #  Sayfanın yerleşimi (Sol kolon boşluk, sağ kolon slider)
# slider, ana_sayfa = st.columns([1, 1])


# # --- sol kolon bilgilendirme ---


# with slider:
    
#     # CSS'in sadece bu kolonu tanıması için id 
#     # Burada slider=.. gibi tanımlanmıyor 
 
    
#     st.markdown("<div id='slider'></div>", unsafe_allow_html=True)
    
#     try:
#         st.image("saski.jpg", use_column_width=True)
       
#     except FileNotFoundError:
#         st.info("Sol tarafta gösterilecek resim bulunamadı. Lütfen dosya adını güncelleyin.")



# # --- sağ kolon (Form ve Sonuçlar) ---


# with ana_sayfa:
    
#     # CSS'in sadece bu kolonu tanıması için id
#     st.markdown("<div id='ana_sayfa'></div>", unsafe_allow_html=True)
    
#     # Analiz yapılmadıysa formu
#     if not st.session_state.analiz_tamamlandi:
#         st.markdown("<h3 style='color : #ADD8E6' !important; text-shadow : 1px 1px 3px rgba (0,0,0,0.8); margin-bottom : 0px;'>💧 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
#         st.markdown("### Ana Sayfa ")
#         st.markdown("<hr style='border:1px solid white'>" , unsafe_allow_html=True)
        
#         # --- Formu 2 kolona bölme, parametrelerin girildiği ---
#         form_sol, form_sag = st.columns(2)
        
#         with form_sol:
#             st.image("sask_mobil.jpg")
            
            
#         with form_sag:
#            st.image("slider_1.jpg")
           
           
#             # Sağ kolonun sol kolonla (5 kutu vs 4 kutu) hizalı görünmesi için alt tarafa şeffaf bir boşluk 
#         st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
            



import streamlit as st
import pandas as pd
import base64
import joblib  

st.set_page_config(page_title="Su Kalite Analizi", layout="wide", page_icon="🚰")

if 'teste_gec' not in st.session_state:
    st.session_state.teste_gec = False
    st.session_state.sonuc = 0

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# Tüm sayfanın arka planı için kullanılacak resim
img_base64 = get_base64_of_bin_file("doga_kaynak.webp")


#date input (Tarih seçimi için bir kutu ekler)
import datetime   
today=st.date_input("Today is" , datetime.datetime.now())  # now() ile bugünün tarihini ekler

if img_base64:
    custom_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Sadece sağ kolondaki forma (2. kolon) BUZLU CAM efekti uyguladım. */
    [data-testid="column"]:nth-of-type(1) {{
        align-items=center;
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
    }}

    /* .stMarkdown kuralı kaldırdım ki Mavi renk (ADD8E6) çalışabilsin */
    h1, h2, h4, p, label {{
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
        font-size: 18px;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.3s ease;
        
    }}
    
    .stButton>button:hover {{
        transform: scale(1.02);
        box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

if st.button("Analize Git 🚀", type="primary"):
    # st.Page içinde tanımladığınız dosya adını birebir aynı şekilde yazın
    sayfa_test = st.Page("icilebilirlik_test_ekranı.py")
    st.switch_page(sayfa_test)

# Ekranı iki eşit parçaya bölüyoruz (Sol kolon resim için, Sağ kolon form için)
sol_kolon, sag_kolon = st.columns([1, 1])

# # --- SOL KOLON (Su Bardağı Resmi) ---
# with sol_kolon:
#     if not st.session_state.teste_gec:
#         # Renk kodu artık sorunsuz çalışacaktır
#         st.markdown("<h3 style='color: #ADD8E6 !important; text-shadow: 1px 1px 3px rgba(0,0,0,0.8); margin-bottom: 0px;'>💧 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
#         st.markdown("#### GİRİŞ PARAMETRELERİ")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
#         st.button("Analiz Et 🚀", key="analiz_butonu_1")
    
#         if st.session_state.sonuc == 0: 
#             st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
#         else:
#             st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
#             st.markdown("<br>", unsafe_allow_html=True)
            
#         if st.button("Yeni Test Yap 🔄", key="yeni_test_buton"):
#             st.session_state.analiz_tamamlandi = False
#             st.rerun()
