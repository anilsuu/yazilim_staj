# -*- coding: utf-8 -*-
"""
@author: nilsu
"""
            
import streamlit as st
import pandas as pd
import base64



# Geniş mod - layout="wide" ekranın daha büyük

st.set_page_config(page_title="Su İçilebilirlik Analizi", layout="wide", page_icon="🚰")



# Modeli medyan_puanlamalıdan ve scaleri yükledim 

try:
    model = joblib.load("rf_kural_su_model.pkl")
    scaler = joblib.load("scaler_kural.pkl")
except FileNotFoundError:
    st.error("Model dosyaları (pkl) bulunamadı! Lütfen eğitim kodunuzu çalıştırdığınıza ve aynı klasörde olduğunuza emin olun.")



#  Sayfanın durum yönetimi - Formun kaybolması için

if 'analiz_tamamlandi' not in st.session_state:
    st.session_state.analiz_tamamlandi = False
    st.session_state.sonuc = 0



#  Web arka sayfasına su-bardak resmi yükleme 

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""



# Tüm sayfanın arka planı için kullanılacak resim

img_base64 = get_base64_of_bin_file("suweb2.webp")



#  CSS Özelleştirme kodları

#  rgba(255, 255, 255, 0.4); ana renklerin ne oranda karıştırılacağını gösteriyor.

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
        
   # /*  Sonuç ekranında buzlu boyutu küçültüp ortalama */
   #  [data-testid="stHorizontalBlock"]:has(#su_sonuc) {{
   #      align-items: center !important; 
   #  }}
        
        
    /* Su Bardağı Resmi (Sol) */
    [data-testid="column"]:has(#su_resmi) {{
        display: flex;
        flex-direction: column;
        margin-top: 1rem; 
        margin-bottom: 1rem; 
    }}

    /* Resmi kırparak alanı doldurmasını sağla */
    [data-testid="column"]:has(#su_resmi) > div,
    [data-testid="column"]:has(#su_resmi) [data-testid="stImage"] {{
        height: 100% !important;
        display: flex;
    }}
        
    [data-testid="column"]:has(#su_resmi) img {{
        height: 100% !important;
        object-fit: cover !important; 
        border-radius: 25px; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
    }}

    /* Form Kolonu (Sağ) - Buzlu  */
    [data-testid="column"]:has(#su_formu){{
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

    /* Display için css ayarları */
    @media (max-width: 768px) {{
        /* Sadece ana bloğu Grid yap (içerideki 2'li form kolonları etkilenmesin) */
        [data-testid="stHorizontalBlock"]:has(#su_resmi):has(#su_formu) {{
            display: grid !important;
            grid-template-columns: 1fr !important;
        }}
        
        [data-testid="column"]:has(#su_resmi),
        [data-testid="column"]:has(#su_formu) {{
            grid-column: 1 / 2 !important;
            grid-row: 1 / 2 !important;
            width: 100% !important; 
        }}

        /* Form üste gelecek şekilde ayarla */
        [data-testid="column"]:has(#su_formu) {{
            z-index: 10 !important; 
            width: 95% !important; 
            margin: auto !important; 
            padding: 1.5rem !important; 
        }}
        
        /* Soldaki resim kutusunu altta kalacak şekilde ayarla */
        [data-testid="column"]:has(#su_resmi) {{
            z-index: 1 !important; 
        }}
            
        [data-testid="column"]:has(#su_resmi) img {{
            min-height: 90vh !important; 
        }}
            
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)



#  Sayfanın yerleşimi (Sol kolon veri girişi, sağ kolon boşluk)
sol_kolon, sag_kolon = st.columns([1, 1])


# --- sol kolon (Su Bardağı Resmi) ---


with sol_kolon:
    
    # CSS'in sadece bu kolonu tanıması için id 
    # Burada su_resmi=.. gibi tanımlanmıyor 
    # Streamlitte Kolonların içine HTMLden (<div>) yazdım ve bunlara id='su_resmi', id='su_formu'
    
    
    st.markdown("<div id='su_resmi'></div>", unsafe_allow_html=True)
    
    try:
        st.image("suweb.webp", use_column_width=True)
        
    except FileNotFoundError:
        st.info("Sol tarafta gösterilecek resim bulunamadı. Lütfen dosya adını güncelleyin.")



# --- sağ kolon (Form ve Sonuçlar) ---


with sag_kolon:
    
    # CSS'in sadece bu kolonu tanıması için id
    st.markdown("<div id='su_formu'></div>", unsafe_allow_html=True)
    
    # Analiz yapılmadıysa formu
    if not st.session_state.analiz_tamamlandi:
        st.markdown("<h3 style='color : #ADD8E6' !important; text-shadow : 1px 1px 3px rgba (0,0,0,0.8); margin-bottom : 0px;'>💧 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
        st.markdown("### Suyun Bileşen Analizi")
        st.markdown("<hr style='border:1px solid white'>" , unsafe_allow_html=True)
        
        # --- Formu 2 kolona bölme, parametrelerin girildiği ---
        form_sol, form_sag = st.columns(2)
        
        with form_sol:
            ph = float(st.number_input("💧 pH Seviyesi", step=1.0, min_value=0.22749905, max_value=14.0, help="ph bilgisi 0-14 değerleri arasında olmalıdır.")) 
            Solids = float(st.number_input("🧊 Solids (Katılar)", step=1.0, min_value=320.9426113, help="Suyun içinde çözünmüş halde bulunan mineral, tuz ve iyonların toplam miktarını ifade eder."))
            Sulfate = float(st.number_input("🟣 Sulfate(Sülfat)", step=1.0, min_value=129.0, help="Suyun kalitesini değerlendirmek için ölçülür."))
            Organic_carbon = float(st.number_input("💬 Organic Carbon", step=1.0, min_value=2.2, help="(Organik karbon) değerini giriniz."))
            Turbidity = float(st.number_input("🌀 Turbidity(Bulanıklık)", step=1.0, min_value=1.45, help="Suyun içindeki askıda katı maddelerin ışığı dağıtmasıyla suyun berraklığının azalmasıdır."))

        with form_sag:
            Hardness = float(st.number_input("🪨 Hardness(Sertlik)", step=1.0, min_value=73.49223369, help="Suyun bir yüzeye temas etmeye karşı gösterdiği dirençtir."))
            Chloramines = float(st.number_input("🧪 Chloramines", step=1.0, min_value=1.390870905, help="Suyun dezenfeksiyon aşamasında klor kullanılınca oluşur"))
            Conductivity = float(st.number_input("⚡ Conductivity(İletkenlik)", step=1.0, min_value=201.6197368, help="20°C İletkenlik genellikle 50-500 değerleri arasında olur."))
            Trihalomethanes = float(st.number_input("🟠 Trihalometanlar", step=1.0, min_value=8.577012933, help="Trihalometanlar için sınır değer,100 µg/L olarak belirlenmiştir."))
            
            # Sağ kolonun sol kolonla (5 kutu vs 4 kutu) hizalı görünmesi için alt tarafa şeffaf bir boşluk 
            st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True) # Buton öncesi küçük boşluk

        # Butonu tüm alana yaymak için: use_container_width=True
        if st.button("Analiz Et 🚀", key="analiz_butonu1", use_container_width=True):
            input_data = pd.DataFrame({
                "ph": [ph],
                "Hardness": [Hardness],
                "Solids":[Solids],
                "Chloramines":[Chloramines],
                "Sulfate":[Sulfate],
                "Conductivity":[Conductivity],
                "Organic_carbon":[Organic_carbon],
                "Trihalomethanes":[Trihalomethanes],
                "Turbidity":[Turbidity]
            })
            
            try:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                
                st.session_state.sonuc = int(prediction[0])
                st.session_state.analiz_tamamlandi = True
                st.rerun()
                
            except Exception as e:
                st.error(f"Tahmin sırasında bir hata oluştu: {e}")
                
    # Analiz yapıldıysa sonuç ekranı gelir.
    else:
        st.markdown("### 📊 ANALİZ SONUCU")
        st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
        
        if st.session_state.sonuc == 0: 
            st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
        else:
            st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
            st.markdown("<br>", unsafe_allow_html=True)
            
        if st.button("Yeni Test Yap 🔄", key="yeni_test_buton", use_container_width=True):
            st.session_state.analiz_tamamlandi = False
            st.rerun()






















































# import streamlit as st
# import pandas as pd
# import base64
# import joblib  # Modeli yüklemek için gerekli


# # Geniş mod - layout="wide" ekranın daha büyük
# st.set_page_config(page_title="Su İçilebilirlik Analizi", layout="wide", page_icon="🚰")

# # Modeli medyan_puanlamalıdan  ve Scaler'ı yükledim 
# try:
#     model = joblib.load("rf_kural_su_model.pkl")
#     scaler = joblib.load("scaler_kural.pkl")
# except FileNotFoundError:
#     st.error("Model dosyaları (pkl) bulunamadı! Lütfen eğitim kodunuzu çalıştırdığınıza ve aynı klasörde olduğunuza emin olun.")


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
# img_base64 = get_base64_of_bin_file("suweb2.webp")



# #  CSS Özelleştirme kodları , buzlu görüntü , display ayarı vb.
# if img_base64:
#     custom_css = f"""
#     <style>
#     /* Arka plan resmini tam ekran yapma */
#     .stApp {{
#         background-image: url("data:image/webp;base64,{img_base64}");
#         background-size: cover;
#         background-position: center;
#         background-attachment:fixed;
#         }}
# /* EŞİTLEME İŞLEMİ 1: Tüm yatay kolon bloğunu esnemeye (stretch) zorla */
#     [data-testid="stHorizontalBlock"] {{
#         align-items: stretch !important;
#     }}

#     /* EŞİTLEME İŞLEMİ 2: Sol kolon (Resim) ayarları ve konumlandırması */
#     [data-testid="column"]:nth-of-type(1) {{
#         display: flex;
#         flex-direction: column;
#         margin-top: 1rem; /* Sağdaki formla aynı boşluk */
#         margin-bottom: 1rem; /* Sağdaki formla aynı boşluk */
#     }}

#     /* Resmin içindeki kapsayıcı div'leri tam yüksekliğe zorla */
#     [data-testid="column"]:nth-of-type(1) > div,
#     [data-testid="column"]:nth-of-type(1) [data-testid="stImage"] {{
#         height: 100% !important;
#         display: flex;
#     }}

#     /* Resmi sündürmeden kırparak alanı doldurmasını sağla */
#     [data-testid="column"]:nth-of-type(1) img {{
#         height: 100% !important;
#         object-fit: cover !important; 
#         border-radius: 25px; /* Sağdaki forma görsel uyum için */
#         box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); /* Forma uyumlu gölge */
#         display:flex; /* flex özelliğe sahip bir kutu içerisinde öğeler uygun boyutlandırılır.*/
#         flex-direction: column;
    
#     }}
    
    
#     /* Ortadaki veri giriş kartına BUZLU CAM  */
#     [data-testid="column"]:nth-of-type(2) {{
#         background: rgba(255, 255, 255, 0.15) !important;
#         backdrop-filter: blur(12px) !important; /*Kutunun buzlu olması 12px*/
#         -webkit-backdrop-filter: blur(12px) !important; /*Aykırı arka tarafı blurlaması*/
#         border-radius: 25px; /*Kutunun köşelerini yuvarlatmak*/
#         border: 1px solid rgba(255, 255, 255, 0.4); 
#         padding: 2rem 3rem;  /*Sayfa da kenarlara boşluk bırakmak için*/
#         margin-top: 1rem; /* Sayfa üstüne boşluk */
#         margin-bottom: 1rem; /* Sayfa altına boşluk */
#         box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
#         display:flex; /* flex özelliğe sahip bir kutu içerisinde öğeler uygun boyutlandırılır.*/
#         flex-direction: column;
        
    
#     }}

#     /* Yazıları okunabilir yapmak için beyaz renk ve gölge */
#     h1,h2, h4, p, label, .stMarkdown {{
#         color: #ffffff !important;
#         text-shadow: 1px 1px 3px rgba(0,0,0,0.8); 
#         /*Shadow +ile başlıyorsa sağa doğru,- ile başlıyorsa sola doğru*/
#     }}
    

#     /* Slider (Kaydırma Çubuğu) değer yazılarının rengi */
#     .stSlider [data-testid="stThumbValue"] {{
#         color: #ffffff !important;
#     }}

#     /* Buton Tasarımı */
#     .stButton>button {{
#         background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
#         color: #800000 !important;
#         # font-weight: bold;
#         # font-size: 18px;
#         # border-radius: 20px;
#         # border: none;
#         # padding: 10px 24px;
#         # width: 100%;
#         # transition: all 0.3s ease;
#     }}
                            
    
#     .stButton>button:hover {{
#         transform: scale(1.02);
#         box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
#     }}
    
    
#     @media (max-width: 768px) {{
#         /* Kolonların yatay bloğunu CSS Grid'e çevir */
#         [data-testid="stHorizontalBlock"] {{
#             display: grid !important;
#             grid-template-columns: 1fr !important; /* Tek sütun */
#         }}
        
#         /* Her iki kolonu da aynı ızgara (grid) hücresine üst üste yerleştir */
#         [data-testid="column"]:nth-of-type(1),
#         [data-testid="column"]:nth-of-type(2) {{
#             grid-column: 1 / 2 !important;
#             grid-row: 1 / 2 !important;
#             width: 100% !important; /* Streamlit'in varsayılan genişliğini ez */
#         }}

#         /* FORM KUTUSU (Sağ Kolon) -  Üste gelsin */
#         [data-testid="column"]:nth-of-type(2) {{
#             z-index: 10 !important; /* Derinlik verip öne alma */
#             width: 90% !important; /* Mobilde resmin kenarlarından biraz görünmesi */
#             margin: auto !important; /* Kutuyu mobilde ortala */
#             padding: 1.5rem !important; /* Mobilde iç boşluğu biraz daralt */
#         }}
        
#         /* RESİM KUTUSU (Sol Kolon) - ALTTA KALSIN */
#         [data-testid="column"]:nth-of-type(1) {{
#             z-index: 1 !important; /* Altta kalması için z-index düşük */
#         }}
        
#         /* Mobilde resmin formun altında yeterince uzun görünmesi için */
#         [data-testid="column"]:nth-of-type(1) img {{
#             min-height: 85vh !important; /* Ekranın %85'ini kaplasın */
#             object-fit: cover !important;
#         }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# #  Sayfanın yerleşimi (Sol kolon veri girişi, sağ kolon boşluk)
# sol_kolon, sag_kolon = st.columns([1, 1])


# #Sadece sağ kolonun içine içerik ekleme
# # rgba (0,0,0,0.8) demek renklerin oranlarıyla karışım yapıyor.


# # --- SOL KOLON (Su Bardağı Resmi) ---
# with sol_kolon:
#     try:
#         st.image("suweb.webp", use_column_width=True)
        
#     except FileNotFoundError:
#         st.info("Sol tarafta gösterilecek resim bulunamadı. Lütfen dosya adını güncelleyin.")




# with sag_kolon:
#     # 1. DURUM: Analiz YAPILMADIYSA sadece formu ve "Analiz Et" butonunu göster
#     if not st.session_state.analiz_tamamlandi:
#         st.markdown("<h3 style='color : #ADD8E6' !important; text-shadow : 1px 1px 3px rgba (0,0,0,0.8); margin-bottom : 0px;'>💧 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
#         st.markdown("### Suyun Analiz Edilecek Bileşenleri")
#         st.markdown("<hr style='border:1px solid white'>" , unsafe_allow_html=True)
        
#         ph = float(st.number_input("💧 pH Seviyesi", step=1.0, min_value=0.22749905, max_value=14.0, help="ph bilgisi 0-14 değerleri arasında olmalıdır.")) 
#         Hardness = float(st.number_input("🪨 (Hardness) Sertlik ", step=1.0, min_value=73.49223369, help="Sertlik(Hardness):Suyun bir yüzeye temas etmeye karşı gösterdiği dirençtir."))
#         Solids = float(st.number_input("🧊 Solids (Katılar)", step=1.0, min_value=320.9426113, help="Solids(Katılar):Suyun içinde çözünmüş halde bulunan mineral, tuz ve iyonların toplam miktarını ifade eder."))
#         Chloramines = float(st.number_input("🧪 Chloramines(Kloramin)", step=1.0, min_value=1.390870905, help="Chloramines(Kloramin):Suyun dezenfeksiyon aşamasında klor kullanılınca oluşur"))
#         Sulfate = float(st.number_input("🟣 Sulfate(Sülfat)", step=1.0, min_value=129.0, help="Sulfate(Sülfat):Suyun kalitesini değerlendirmek için ölçülür."))
#         Conductivity = float(st.number_input("⚡ Conductivity(İletkenlik)", step=1.0, min_value=201.6197368, help="20°C Conductivity (İletkenlik) genellikle 50-500 değerleri arasında olur."))
#         Organic_carbon = float(st.number_input("💬 Organic_carbon (Organik karbon)", step=1.0, min_value=2.2, help="Organic_carbon (Organik karbon) değerini giriniz."))
#         Trihalomethanes = float(st.number_input("🟠 Trihalomethanes(Trihalometanlar)", step=1.0, min_value=8.577012933, help="Trihalomethanes(Trihalometanlar) için sınır değer, 2005 yılından beri gereği 100 µg/L olarak belirlenmiştir."))
#         Turbidity = float(st.number_input("🌀 Turbidity(Bulanıklık)", step=1.0, min_value=1.45, help="Turbidity(Bulanıklık):Suyun içindeki askıda katı maddelerin ışığı dağıtmasıyla suyun berraklığının azalmasına denir."))

#         if st.button("Analiz Et 🚀", key="analiz_butonu1"):
#             input_data = pd.DataFrame({
#                 "ph": [ph],
#                 "Hardness": [Hardness],
#                 "Solids":[Solids],
#                 "Chloramines":[Chloramines],
#                 "Sulfate":[Sulfate],
#                 "Conductivity":[Conductivity],
#                 "Organic_carbon":[Organic_carbon],
#                 "Trihalomethanes":[Trihalomethanes],
#                 "Turbidity":[Turbidity]
#             })
            
#             try:
#                 input_scaled = scaler.transform(input_data)
#                 prediction = model.predict(input_scaled)
                
#                 st.session_state.sonuc = int(prediction[0])
#                 st.session_state.analiz_tamamlandi = True
#                 st.rerun()
                
#             except Exception as e:
#                 st.error(f"Tahmin sırasında bir hata oluştu: {e}")
                
#     # 2. DURUM: Analiz YAPILDIYSA formu tamamen gizle, sadece sonucu ve "Yeni Test Yap" butonunu göster
#     else:
#         st.markdown("### 📊 ANALİZ SONUCU")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
        
#         if st.session_state.sonuc == 0: 
#             st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
#         else:
#             st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
#             st.markdown("<br>", unsafe_allow_html=True)
            
#         if st.button("Yeni Test Yap 🔄", key="yeni_test_buton"):
#             st.session_state.analiz_tamamlandi = False
#             st.rerun()






















            
            
            
            
            

























   
    
   
    
   