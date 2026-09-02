# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:10:52 2026

@author: nilsu
"""





# #SAYFA DA 2 RESİM VAR , FORM KAYBOLUYOR.
            
import streamlit as st
import pandas as pd
import base64
import joblib  # Modeli yüklemek için gerekli


# Geniş mod - layout="wide" ekranın daha büyük
st.set_page_config(page_title="Su İçilebilirlik Analizi", layout="wide", page_icon="🚰")

# Modeli medyan_puanlamalıdan  ve Scaler'ı yükledim 
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

img_base64 = get_base64_of_bin_file("suweb2.webp")



#  CSS Özelleştirme kodları , buzlu görüntü , display ayarı vb.
if img_base64:
    custom_css = """
    <style>
    /* Arka plan resmini tam ekran yapma */
    .stApp {{
        background-image: url("data:image/webp;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: scrool,fixed;
        }}

    /* Ortadaki veri giriş kartına BUZLU CAM  */
    [data-testid="column"]:nth-of-type(2) {{
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(12px) !important; #Kutunun buzlu olması 12px
        -webkit-backdrop-filter: blur(12px) !important; #Aykırı arka tarafı blurlaması
        border-radius: 25px; #Kutunun köşelerini yuvarlatmak
        border: 1px solid rgba(255, 255, 255, 0.4); 
        padding: 2rem 3rem;  #Sayfa da kenarlara boşluk bırakmak için
        margin-top: 3rem; # Sayfa üstüne boşluk
        margin-bottom: 3rem; # Sayfa altına boşluk
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); 
        display:flex; # flex özelliğe sahip bir kutu içerisinde öğeler uygun boyutlandırılır.
        flex-direction: column;
    }}

    /* Yazıları okunabilir yapmak için beyaz renk ve gölge */
    h1,h2, h4, p, label, .stMarkdown {{
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8); 
        #Shadow +ile başlıyorsa sağa doğru,- ile başlıyorsa sola doğru
    }}
    

    /* Slider (Kaydırma Çubuğu) değer yazılarının rengi */
    .stSlider [data-testid="stThumbValue"] {{
        color: #ffffff !important;
    }}

    /* Buton Tasarımı */
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

#  Sayfanın yerleşimi (Sol kolon veri girişi, sağ kolon boşluk)
sol_kolon, sag_kolon = st.columns([1, 2])


#Sadece sağ kolonun içine içerik ekleme
# rgba (0,0,0,0.8) demek renklerin oranlarıyla karışım yapıyor.
with sag_kolon:
    #Analiz yapıladıysa formu göstermeye devam eder.
    if not st.session_state.analiz_tamamlandi:
        st.markdown("<h3 style='color : lightblue' !important; text-shadow : 1px 1px 3px rgba (0,0,0,0.8); margin-bottom : 0px;'>💧 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
        st.markdown("### GİRİLMESİ GEREKEN DEĞERLER")
        st.markdown("<hr style='border:1px solid white'>" , unsafe_allow_html=True)
        
        
# # Sadece sağ kolonun içine içerik ekleme
# with sag_kolon:
#     # Analiz yapılmadıysa formu göstermeye devam eder
#     if not st.session_state.analiz_tamamlandi:
#         st.markdown("<h3 style='color: lightblue !important; text-shadow: 1px 1px 3px rgba(0,0,0,0.8); margin-bottom: 0px;'>💧 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
#         st.markdown("#### GİRİŞ PARAMETRELERİ")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)



#Girilecek parametrelerin .input olarak veri tipini belirterek,help kutusundaki mesaj açılması.
        ph = float(st.number_input("💧 pH Seviyesi", step=1.0, min_value=0.22749905, max_value=14.0, help="ph bilgisi 0-14 değerleri arasında olmalıdır.")) 
        Hardness = float(st.number_input("🪨 (Hardness) Sertlik ", step=1.0, min_value=73.49223369, help="Sertlik(Hardness):Suyun bir yüzeye temas etmeye karşı gösterdiği dirençtir."))
        Solids = float(st.number_input("🧊 Solids (Katılar)", step=1.0, min_value=320.9426113, help="Solids(Katılar):Suyun içinde çözünmüş halde bulunan mineral, tuz ve iyonların toplam miktarını ifade eder."))
        Chloramines = float(st.number_input("🧪 Chloramines(Kloramin)", step=1.0, min_value=1.390870905, help="Chloramines(Kloramin):Suyun dezenfeksiyon aşamasında klor kullanılınca oluşur"))
        Sulfate = float(st.number_input("🟣 Sulfate(Sülfat)", step=1.0, min_value=129.0, help="Sulfate(Sülfat):Suyun kalitesini değerlendirmek için ölçülür."))
        Conductivity = float(st.number_input("⚡ Conductivity(İletkenlik)", step=1.0, min_value=201.6197368, help="20°C Conductivity (İletkenlik) genellikle 50-500 değerleri arasında olur."))
        Organic_carbon = float(st.number_input("💬 Organic_carbon (Organik karbon)", step=1.0, min_value=2.2, help="Organic_carbon (Organik karbon) değerini giriniz."))
        Trihalomethanes = float(st.number_input("🟠 Trihalomethanes(Trihalometanlar)", step=1.0, min_value=8.577012933, help="Trihalomethanes(Trihalometanlar) için sınır değer, 2005 yılından beri gereği 100 µg/L olarak belirlenmiştir."))
        Turbidity = float(st.number_input("🌀 Turbidity(Bulanıklık)", step=1.0, min_value=1.45, help="Turbidity(Bulanıklık):Suyun içindeki askıda katı maddelerin ışığı dağıtmasıyla suyun berraklığının azalmasına denir."))

if st.button("Analiz Et 🚀", key="yeni_test_butonu"):
    #Arayüzden gelen tüm değerler Pandas ile DataFrame'e dönüştürülüyor.
    #Değişken isimleri (Hardness,Solids vs.) yukarıda tanımlandığı gibi DataFrame ekledim.
    
    input_data=pd.DataFrame({
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
        # Girilen verileri , eğitilmiş scaler ile ölçeklendirme
        input_scaled=scaler.transform(input_data)
        
        #Eğitilen modelden tahmini alma
        prediction=model.predict(input_scaled)
        
        #Çıkan sonucu (1-0) oturum durumuna (session_state) kaydeder.
        st.session_state.sonuc=int(prediction[0])
        st.session_state.analiz_tamamlandi=True
        
        #Sayfayı yenileme ve sonuç kartını gösterme
        st.rerun()
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu:{e}")
        
    
#         if st.button("Analiz Et 🚀",key="yeni_test_butonu"):
#             # 1. Arayüzden gelen tüm değerler Pandas DataFrame'e dönüştürülüyor
#             # Değişken isimleri (Hardness, Solids vs.) yukarıda tanımlandığı gibi dataframe eklendi.
#             input_data = pd.DataFrame({
#                 "ph": [ph],
#                 "Hardness": [Hardness],
#                 "Solids": [Solids],                 
#                 "Chloramines": [Chloramines],
#                 "Sulfate": [Sulfate],                
#                 "Conductivity": [Conductivity],      
#                 "Organic_carbon": [Organic_carbon], 
#                 "Trihalomethanes": [Trihalomethanes],
#                 "Turbidity": [Turbidity]            
#             })
            
#             try:
#                 # Veriyi, eğitilmiş scaler ile ölçeklendirir
#                 input_scaled = scaler.transform(input_data)
                
#                 #  Modelden tahmini al
#                 prediction = model.predict(input_scaled)
                
#                 #  Çıkan sonucu (1 veya 0) oturum durumuna (session_state) kaydeder.
#                 st.session_state.sonuc = int(prediction[0])
#                 st.session_state.analiz_tamamlandi = True
                
#                 # Sayfayı yenile ve sonuç kartını göster
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Tahmin sırasında bir hata oluştu: {e}")


            
#     # Eğer analiz yapıldıysa sayfada formu kaldırır sadece sonuç kartını gösterir.
#     else:
#         st.markdown("### 📊 ANALİZ SONUCU")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
        
#         # if st.session_state.sonuc == 1:
#         #     st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
#         # else:
#         #     st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
            
        
        
#         #  1'i 0 yaptım
#         if st.session_state.sonuc == 0: 
#             st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
#         else:
#             st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
#             st.markdown("<br>", unsafe_allow_html=True)
            
            
#         # Yeni Hali:
#         if st.button("Yeni Test Yap 🔄", key="yeni_test_buton"):
#             st.session_state.analiz_tamamlandi = False
#             st.rerun() # Sayfayı yenile ve formu geri getir

























































# import streamlit as st
# import pandas as pd
# import base64
# import joblib  

# st.set_page_config(page_title="Su Kalite Analizi", layout="wide", page_icon="🚰")

# try:
#     model = joblib.load("rf_kural_su_model.pkl")
#     scaler = joblib.load("scaler_kural.pkl")
# except FileNotFoundError:
#     st.error("Model dosyaları (pkl) bulunamadı! Lütfen aynı klasörde olduğunuza emin olun.")

# if 'analiz_tamamlandi' not in st.session_state:
#     st.session_state.analiz_tamamlandi = False
#     st.session_state.sonuc = 0

# def get_base64_of_bin_file(bin_file):
#     try:
#         with open(bin_file, 'rb') as f:
#             return base64.b64encode(f.read()).decode()
#     except FileNotFoundError:
#         return ""

# # Tüm sayfanın arka planı için kullanılacak resim
# img_base64 = get_base64_of_bin_file("suweb2.webp")

# if img_base64:
#     custom_css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/webp;base64,{img_base64}");
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#     }}

#     /* Sadece sağ kolondaki forma (2. kolon) BUZLU CAM efekti uyguladım. */
#     [data-testid="column"]:nth-of-type(2) {{
#         background: rgba(255, 255, 255, 0.15) !important;
#         backdrop-filter: blur(12px) !important;
#         -webkit-backdrop-filter: blur(12px) !important;
#         border-radius: 25px;
#         border: 1px solid rgba(255, 255, 255, 0.4);
#         padding: 2rem 3rem;
#         margin-top: 1rem;
#         margin-bottom: 1rem;
#         box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
#         display:flex;
#         flex-direction: column;
#     }}

#     /* .stMarkdown kuralı kaldırdım ki Mavi renk (ADD8E6) çalışabilsin */
#     h1, h2, h4, p, label {{
#         color: #ffffff !important;
#         text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
#     }}
    
#     .stSlider [data-testid="stThumbValue"] {{
#         color: #ffffff !important;
#     }}

#     .stButton>button {{
#         background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
#         color: #800000 !important;
#         font-weight: bold;
#         font-size: 18px;
#         border-radius: 20px;
#         border: none;
#         padding: 10px 24px;
#         width: 100%;
#         transition: all 0.3s ease;
#     }}
    
#     .stButton>button:hover {{
#         transform: scale(1.02);
#         box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
#     }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# # Ekranı iki eşit parçaya bölüyoruz (Sol kolon resim için, Sağ kolon form için)
# sol_kolon, sag_kolon = st.columns([1, 1])

# # --- SOL KOLON (Su Bardağı Resmi) ---
# with sol_kolon:
#     try:
        
#         st.image("suweb.webp", use_column_width=True)
#     except FileNotFoundError:
#         st.info("Sol tarafta gösterilecek resim bulunamadı. Lütfen dosya adını güncelleyin.")

# # --- SAĞ KOLON (Buzlu Cam Form ve Test) ---
# with sag_kolon:
#     if not st.session_state.analiz_tamamlandi:
#         # Renk kodu artık sorunsuz çalışacaktır
#         st.markdown("<h3 style='color: #ADD8E6 !important; text-shadow: 1px 1px 3px rgba(0,0,0,0.8); margin-bottom: 0px;'>💧 SU KALİTE ANALİZİ 🚰</h3>", unsafe_allow_html=True)
#         st.markdown("#### GİRİŞ PARAMETRELERİ")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)

#         ph = float(st.number_input("💧 pH Seviyesi", step=1.0, min_value=0.22749905, max_value=14.0)) 
#         Hardness = float(st.number_input("🪨 (Hardness) Sertlik ", step=1.0, min_value=73.49223369))
#         Solids = float(st.number_input("🧊 Solids (Katılar)", step=1.0, min_value=320.9426113))
#         Chloramines = float(st.number_input("🧪 Chloramines(Kloramin)", step=1.0, min_value=1.390870905))
#         Sulfate = float(st.number_input("🟣 Sulfate(Sülfat)", step=1.0, min_value=129.0))
#         Conductivity = float(st.number_input("⚡ Conductivity(İletkenlik)", step=1.0, min_value=201.6197368))
#         Organic_carbon = float(st.number_input("💬 Organic_carbon (Organik karbon)", step=1.0, min_value=2.2))
#         Trihalomethanes = float(st.number_input("🟠 Trihalomethanes(Trihalometanlar)", step=1.0, min_value=8.577012933))
#         Turbidity = float(st.number_input("🌀 Turbidity(Bulanıklık)", step=1.0, min_value=1.45))

#         if st.button("Analiz Et 🚀", key="analiz_butonu_1"):
#             input_data = pd.DataFrame({
#                 "ph": [ph], "Hardness": [Hardness], "Solids": [Solids],                 
#                 "Chloramines": [Chloramines], "Sulfate": [Sulfate],                
#                 "Conductivity": [Conductivity], "Organic_carbon": [Organic_carbon], 
#                 "Trihalomethanes": [Trihalomethanes], "Turbidity": [Turbidity]            
#             })
            
#             try:
#                 input_scaled = scaler.transform(input_data)
#                 prediction = model.predict(input_scaled)
#                 st.session_state.sonuc = int(prediction[0])
#                 st.session_state.analiz_tamamlandi = True
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Tahmin sırasında bir hata oluştu: {e}")
            
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
            
            
            
            
            


























# with sol_kolon:
#     # EĞER ANALİZ YAPILMADIYSA FORMU GÖSTER
#     if not st.session_state.analiz_tamamlandi:
#         st.markdown("### 💧 SU KALİTE ANALİZİ 🚰")
#         st.markdown("#### GİRİŞ PARAMETRELERİ")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)

#         ph = float(st.number_input("💧 pH Seviyesi", step=1.0, min_value=0.22749905, max_value=14.0, help="Suyun ph değerini giriniz.")) 
#         Hardness = float(st.number_input("🪨 (Hardness) Sertlik ", step=1.0, min_value=73.49223369, help="Sertlik(Hardness) değerini giriniz."))
#         Solids = float(st.number_input("🧊 Solids (Katılar)", step=1.0, min_value=320.9426113, help="Solids(Katılar) değerini giriniz."))
#         Chloramines = float(st.number_input("🧪 Chloramines(Kloramin)", step=1.0, min_value=1.390870905, help="Chloramines(Kloramin) değerini giriniz."))
#         Sulfate = float(st.number_input("🟣 Sulfate(Sülfat)", step=1.0, min_value=129.0, help="Sulfate(Sülfat) değerini giriniz."))
#         Conductivity = float(st.number_input("⚡ Conductivity(İletkenlik)", step=1.0, min_value=201.6197368, help="20°C Conductivity (İletkenlik) değerini giriniz."))
#         Organic_carbon = float(st.number_input("💬 Organic_carbon (Organik karbon)", step=1.0, min_value=2.2, help="Organic_carbon (Organik karbon) değerini giriniz."))
#         Trihalomethanes = float(st.number_input("🟠 Trihalomethanes(Trihalometanlar)", step=1.0, min_value=8.577012933, help="Trihalomethanes(Trihalometanlar) değerini giriniz."))
#         Turbidity = float(st.number_input("🌀 Turbidity(Bulanıklık)", step=1.0, min_value=1.45, help="Turbidity(Bulanıklık) değerini giriniz."))

#         if st.button("Analiz Et 🚀"):
#             # 1. Arayüzden gelen tüm değerler Pandas DataFrame'e dönüştürülüyor
#             # Değişken isimleri (Hardness, Solids vs.) tam olarak yukarıda tanımlandığı gibi düzeltildi.
#             input_data = pd.DataFrame({
#                 "ph": [ph],
#                 "Hardness": [Hardness],
#                 "Solids": [Solids],                 
#                 "Chloramines": [Chloramines],
#                 "Sulfate": [Sulfate],                
#                 "Conductivity": [Conductivity],      
#                 "Organic_carbon": [Organic_carbon], 
#                 "Trihalomethanes": [Trihalomethanes],
#                 "Turbidity": [Turbidity]            
#             })
            
#             try:
#                 # 2. Veriyi, eğitilmiş scaler ile ölçeklendir
#                 input_scaled = scaler.transform(input_data)
                
#                 # 3. Modelden tahmini al
#                 prediction = model.predict(input_scaled)
                
#                 # 4. Çıkan sonucu (1 veya 0) oturum durumuna (session_state) kaydet
#                 st.session_state.sonuc = int(prediction[0])
#                 st.session_state.analiz_tamamlandi = True
                
#                 # Sayfayı yenile ve sonuç kartını göster
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Tahmin sırasında bir hata oluştu: {e}")
            
#     # EĞER ANALİZ YAPILDIYSA SADECE SONUÇ KARTINI GÖSTER
#     else:
#         st.markdown("### 📊 ANALİZ SONUCU")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
        
#         if st.session_state.sonuc == 1:
#             st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
#         else:
#             st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
            
#         st.markdown("<br>", unsafe_allow_html=True)
        
#         if st.button("Yeni Test Yap 🔄"):
#             st.session_state.analiz_tamamlandi = False
#             st.rerun() # Sayfayı yenile ve formu geri getir




#         # # Değişken isimleri (küçük harflerle) ve max_value sınırları düzeltildi
#         # ph = st.slider("💧 pH Seviyesi", min_value=0.0, max_value=14.0, value=7.0, step=0.1, help="Suyun ph değerini giriniz.")
#         # hardness = st.slider("🪨 Sertlik (Hardness)", min_value=70.0, max_value=400.0, value=150.0, step=1.0, help="Sertlik değerini giriniz.")
#         # solids = st.slider("🧊 Katılar (Solids)", min_value=300.0, max_value=65000.0, value=20000.0, step=10.0, help="Katı madde miktarını giriniz.")
#         # chloramines = st.slider("🧪 Kloramin (Chloramines)", min_value=0.0, max_value=15.0, value=7.0, step=0.1, help="Kloramin değerini giriniz.")
#         # sulfate = st.slider("🟣 Sülfat (Sulfate)", min_value=120.0, max_value=500.0, value=330.0, step=1.0, help="Sülfat değerini giriniz.")
#         # conductivity = st.slider("⚡ İletkenlik (Conductivity)", min_value=180.0, max_value=800.0, value=400.0, step=1.0, help="İletkenlik değerini giriniz.")
#         # organic_carbon = st.slider("💬 Organik Karbon", min_value=2.0, max_value=30.0, value=14.0, step=0.1, help="Organik karbon değerini giriniz.")
#         # trihalomethanes = st.slider("🟠 Trihalometanlar", min_value=5.0, max_value=130.0, value=65.0, step=1.0, help="Trihalometan değerini giriniz.")
#         # turbidity = st.slider("🌀 Bulanıklık (Turbidity)", min_value=1.0, max_value=7.0, value=4.0, step=0.1, help="Bulanıklık değerini giriniz.")

#         # st.markdown("<br>", unsafe_allow_html=True)

#         if st.button("Analiz Et 🚀"):
#             # 1. Arayüzden gelen tüm değerler Pandas DataFrame'e dönüştürülüyor
#             # Sözlük anahtarları modelinizin eğitimindeki sütun isimleriyle birebir aynı olmalı.
#             input_data = pd.DataFrame({
#                 "ph": [ph],
#                 "Hardness": [hardness],
#                 "Solids": [solids],                 
#                 "Chloramines": [chloramines],
#                 "Sulfate": [sulfate],               
#                 "Conductivity": [conductivity],     
#                 "Organic_carbon": [organic_carbon], 
#                 "Trihalomethanes": [trihalomethanes],
#                 "Turbidity": [turbidity]            
#             })
            
#             try:
#                 # 2. Veriyi, eğitilmiş scaler ile ölçeklendir
#                 input_scaled = scaler.transform(input_data)
                
#                 # 3. Modelden tahmini al
#                 prediction = model.predict(input_scaled)
                
#                 # 4. Çıkan sonucu (1 veya 0) oturum durumuna (session_state) kaydet
#                 st.session_state.sonuc = int(prediction[0])
#                 st.session_state.analiz_tamamlandi = True
                
#                 # Sayfayı yenile ve sonuç kartını göster
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Tahmin sırasında bir hata oluştu: {e}")
            
#     # EĞER ANALİZ YAPILDIYSA SADECE SONUÇ KARTINI GÖSTER
#     else:
#         st.markdown("### 📊 ANALİZ SONUCU")
#         st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
        
#         if st.session_state.sonuc == 1:
#             st.success("✅ Afiyet olsun, su **İÇİLEBİLİR!**")
#         else:
#             st.error("❌ Dikkat! Su **İÇİLEMEZ** (Güvenli Değil).")
            
#         st.markdown("<br>", unsafe_allow_html=True)
        
#         if st.button("Yeni Test Yap 🔄"):
#             st.session_state.analiz_tamamlandi = False
#             st.rerun() # Sayfayı yenile ve formu geri getir



















    
# # 1. SAYFA AYARLARI (Geniş mod - layout="wide")
# st.set_page_config(page_title="Su Kalite Analizi", layout="centered", page_icon="💧")

# # 2. STATE (DURUM) YÖNETİMİ - Formun kaybolması için
# if 'analiz_tamamlandi' not in st.session_state:
#     st.session_state.analiz_tamamlandi = False
#     st.session_state.sonuc = 0

# # 3. ARKA PLAN RESMİ YÜKLEME
# def get_base64_of_bin_file(bin_file):
#     try:
#         with open(bin_file, 'rb') as f:
#             return base64.b64encode(f.read()).decode()
#     except FileNotFoundError:
#         return ""

# img_base64 = get_base64_of_bin_file("suweb2.webp")

# #  MODERN CSS (BUZLU CAM EFEKTİ VE ARKA PLAN)
# if img_base64:
#     custom_css = f"""
#     <style>
#     /* Arka plan resmini tam ekran yapma */
#     .stApp {{
#         background-image: url("data:image/webp;base64,{img_base64}");
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#     }}

#     /* Ortadaki veri giriş kartına BUZLU CAM (Glassmorphism) efekti */
#     .data-testid="column"]:nth-of-type(1) {{
#         background: rgba(255, 255, 255, 0.15) !important; /* Yarı saydamlık */
#         backdrop-filter: blur(12px) !important;          /* Arkasını bulanıklaştırma */
#         -webkit-backdrop-filter: blur(12px) !important;
#         border-radius: 25px;                             /* Köşeleri yuvarlatma */
#         border: 1px solid rgba(255, 255, 255, 0.4);      /* Hafif beyaz çerçeve */
#         padding: 2rem 3rem;
#         margin-top: 3rem;
#         margin-bottom: 3rem;
#         box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);    /* Derinlik gölgesi */
#     }}

#     /* Yazıları okunabilir yapmak için beyaz renk ve gölge */
#     h1, h2, h3, h4, p, label, .stMarkdown {{
#         color: #ffffff !important;
#         text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
#     }}

#     /* Slider (Kaydırma Çubuğu) değer yazılarının rengi */
#     .stSlider [data-testid="stThumbValue"] {{
#         color: #ffffff !important;
#     }}

#     /* Modern Buton Tasarımı */
#     .stButton>button {{
#         background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
#         color: #000000 !important;
#         font-weight: bold;
#         font-size: 18px;
#         border-radius: 20px;
#         border: none;
#         padding: 10px 24px;
#         width: 100%; /* Butonu yatayda tam genişlik yapar */
#         transition: all 0.3s ease;
#     }}
    
#     .stButton>button:hover {{
#         transform: scale(1.02); /* Üzerine gelince hafif büyür */
#         box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
#     }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
# import streamlit as st
# import pandas as pd
# import joblib
# import base64
# import os

# # SAYFA AYARLARI (Mutlaka en üstte olmalı)
# # 'centered' layout, buzlu cam kartımızın ortada şık durmasını sağlar.
# st.set_page_config(page_title="Su Kalite Analizi", layout="centered", page_icon="🚰")


# #  STATE (DURUM) YÖNETİMİ - Formun kaybolması için
# if 'analiz_tamamlandi' not in st.session_state:
#     st.session_state.analiz_tamamlandi = False
#     st.session_state.sonuc = 0
    
    
# # ARKA PLAN RESMİNİ YÜKLEME FONKSİYONU
# def get_base64_of_bin_file(bin_file):
#     try:
#         with open(bin_file, 'rb') as f:
#             data = f.read()
#         return base64.b64encode(data).decode()
#     except FileNotFoundError:
#         st.error(f"HATA: '{bin_file}' adlı resim dosyası bulunamadı! Lütfen resmi Python dosyasıyla aynı klasöre koyun.")
#         return ""


# img_base64 = get_base64_of_bin_file("suweb2.webp")

# #  MODERN CSS (BUZLU CAM EFEKTİ VE ARKA PLAN)
# if img_base64:
#     custom_css = f"""
#     <style>
#     /* Arka plan resmini tam ekran yapma */
#     .stApp {{
#         background-image: url("data:image/webp;base64,{img_base64}");
#         background-size: cover;
#         background-position: center;
#         background-attachment: fixed;
#     }}

#     /* Ortadaki veri giriş kartına BUZLU CAM (Glassmorphism) efekti */
#     .block-container {{
#         background: rgba(255, 255, 255, 0.15) !important; /* Yarı saydamlık */
#         backdrop-filter: blur(12px) !important;          /* Arkasını bulanıklaştırma */
#         -webkit-backdrop-filter: blur(12px) !important;
#         border-radius: 25px;                             /* Köşeleri yuvarlatma */
#         border: 1px solid rgba(255, 255, 255, 0.4);      /* Hafif beyaz çerçeve */
#         padding: 2rem 3rem;
#         margin-top: 3rem;
#         margin-bottom: 3rem;
#         box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);    /* Derinlik gölgesi */
#     }}

#     /* Yazıları okunabilir yapmak için beyaz renk ve gölge */
#     h1, h2, h3, h4, p, label, .stMarkdown {{
#         color: #ffffff !important;
#         text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
#     }}

#     /* Slider (Kaydırma Çubuğu) değer yazılarının rengi */
#     .stSlider [data-testid="stThumbValue"] {{
#         color: #ffffff !important;
#     }}

#     /* Modern Buton Tasarımı */
#     .stButton>button {{
#         background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
#         color: #000000 !important;
#         font-weight: bold;
#         font-size: 18px;
#         border-radius: 20px;
#         border: none;
#         padding: 10px 24px;
#         width: 100%; /* Butonu yatayda tam genişlik yapar */
#         transition: all 0.3s ease;
#     }}
    
#     .stButton>button:hover {{
#         transform: scale(1.02); /* Üzerine gelince hafif büyür */
#         box-shadow: 0px 5px 15px rgba(255, 255, 255, 0.5);
#     }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

# #  UYGULAMA ARAYÜZÜ (KULLANICI GİRİŞLERİ)
# st.title("💧 SU KALİTE ANALİZİ 🚰")
# st.markdown("#### GİRİŞ PARAMETRELERİ")
# st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)



# ph = float(st.number_input("💧 pH Seviyesi",step=1.0,min_value=0.22749905,max_value=14.0,help="Suyun ph değerini giriniz.")) 
# Hardness=float(st.number_input("🪨 (Hardness) Sertlik ",step=1.0,min_value=73.49223369,help="Sertlik(Hardness) değerini giriniz."))
# Solids=float(st.number_input("🧊 Solids (Katılar)",step=1.0,min_value=320.9426113,help="Solids(Katılar) değerini giriniz."))
# Chloramines=float(st.number_input("🧪 Chloramines(Kloramin)",step=1.0,min_value=1.390870905,help="Chloramines(Kloramin) değerini giriniz."))
# Sulfate=float(st.number_input("🟣 Sulfate(Sülfat)",step=1.0,min_value=129.0,help="Sulfate(Sülfat) değerini giriniz."))
# Conductivity=float(st.number_input("⚡ Conductivity(İletkenlik)",step=1.0,min_value=201.6197368,help="20°C Conductivity (İletkenlik) değerini giriniz."))
# Organic_carbon=float(st.number_input("💬 Organic_carbon (Organik karbon)",step=1.0,min_value=2.2,help="Organic_carbon (Organik karbon) değerini giriniz."))
# Trihalomethanes=float(st.number_input("🟠 Trihalomethanes(Trihalometanlar)",step=1.0,min_value=8.577012933,help="Trihalomethanes(Trihalometanlar) değerini giriniz."))
# Turbidity=float(st.number_input("🌀 Turbidity(Bulanıklık)",step=1.0,min_value=1.45,help="Turbidity(Bulanıklık) değerini giriniz."))


# st.markdown("<br>", unsafe_allow_html=True)

# # BUTON VE MODEL TAHMİNİ
# if st.button("Analiz Et 🚀"):
    
#     # Kendi modelinize göre burayı düzenleyebilirsiniz
#     # input_data = pd.DataFrame({"ph": [ph], "Hardness": [hardness], "Chloramines": [chloramines], ...})
#     # prediction = model.predict(input_data)
    
#     # Örnek Sonuç (Model entegre edilene kadar test amaçlı):
#     st.success("📊 Analiz tamamlandı! Su İçilebilir.✅🚰")
#     # st.error("❌ Dikkat! Su İçilemez (Güvenli Değil)")
    
    








































# # Sayfa ayarlarını geniş modda ve modern bir başlıkla başlatın
# st.set_page_config(page_title="Su Kalite Analizi", layout="wide")

# # Modern ve koyu bir "su" teması CSS'i
# page_bg_css = """
# <style>
# /* Arka plan degrade (gradient) rengi */
# .stApp {
#     background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
# }

# /* Başlık ve metin renklerini beyaza çevirme */
# h1, h2, h3, p, label {
#     color: #ffffff !important;
# }

# /* Butonu modernleştirme */
# .stButton>button {
#     background-color: #00d2ff;
#     color: #000000;
#     border-radius: 20px;
#     border: none;
#     padding: 10px 24px;
#     font-weight: bold;
#     transition: all 0.3s ease 0s;
# }
# .stButton>button:hover {
#     background-color: #3a7bd5;
#     color: white;
#     box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
# }
# </style>
# """
# st.markdown(page_bg_css, unsafe_allow_html=True)


# #icilebilirlik su testi web sitesinin iskeletini hazırlamak

# model = joblib.load(r"C:\Users\nilsu\OneDrive\Masaüstü\yazilim_staj\sussis\pytkodlari\rf_kural_su_model.pkl")
# # Sayfa ayarlarını geniş modda ve modern bir başlıkla başlatın

# # st.set_page_config(
# #     page_title="SASKİ Su İçilebilirlik Testi",layout="wide",
# #     page_icon=r"C:\Users\nilsu\OneDrive\Masaüstü\yazilim_staj\sussis\pytkodlari\test_icon.png"
# #     )

# st.title=("SASKİ Su İçilebilirlik Testi")
# st.write=("Lütfen tahmin için değerleri giriniz.")


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




# # Modern Slider Kullanımları (Klavye yerine fare/dokunmatik ile kaydırma)
# col1, col2 = st.columns(2) # Ekranı iki sütuna bölerek şık bir görünüm elde edin

# with col1:
#     ph = st.slider("💧 pH Seviyesi", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
#     hardness = st.slider("🪨 Sertlik (Hardness)", min_value=0.0, max_value=400.0, value=150.0, step=1.0)
    
# with col2:
#     chloramines = st.slider("🧪 Kloramin", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
#     # Diğer parametrelerinizi buraya ekleyin...
   
    
   
    
   
    
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







#     tab1, tab2 = st.tabs(["📊 Parametre Girişi", "📈 Yapay Zeka Analizi"])

# with tab1:
#     st.markdown("### Lütfen su değerlerini kaydırarak belirleyin")
#     # Yukarıdaki slider'ları buraya koyabilirsiniz
    
# with tab2:
#     st.markdown("### Sonuç Ekranı")
#     if st.button("Analizi Başlat 🚀"):
#         st.success("✅ Su İçilebilir!")