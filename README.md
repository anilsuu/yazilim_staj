# 🚰 SASKİ Su Kalitesi Analizi ve Makine Öğrenmesi Sistemi

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://suicmetesti.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

Bu proje, **Sakarya Su ve Kanalizasyon İdaresi (SASKİ)** bünyesinde gerçekleştirdiğim yazılım stajım kapsamında geliştirdiğim uçtan uca bir veri bilimi ve makine öğrenmesi uygulamasıdır . 
Projenin temel amacı, suyun içerdiği kimyasal bileşenleri analiz ederek içilebilirlik durumunu (potability) başarıyla tahmin eden yapay zeka modelleri geliştirmek,
Bu modelleri son kullanıcıların kolayca erişebileceği interaktif bir web arayüzü (`https://suicmetesti.streamlit.app/`) ile sunmaktır.

---

## 🌟 Öne Çıkan Özellikler

- **İnteraktif Web Arayüzü (Streamlit):** Özel CSS tasarımları (buzlu cam / glassmorphism efekti), şık arka plan görselleri ve mobil uyumlu tasarımıyla kullanıcı dostu test ekranı .
- **Dinamik Veri Etiketleme (Kural Tabanlı Puanlama):** Etiketsiz veri seti üzerinde su kalite parametrelerinin medyan değerleri baz alınarak 9 kriter üzerinden dinamik puanlama yapılmış ve veriler 'İçilebilir / İçilemez' sınıflarına ayrılmıştır.
- **Gözetimli Öğrenme (Supervised Learning):** Etiketlenen veriler kullanılarak `RandomForestClassifier` (Random Forest Sınıflandırıcı) eğitilmiş; sınıflar arası dengesizlikler `class_weight='balanced'` parametresiyle optimize edilmiştir.
- **Gözetimsiz Öğrenme (Unsupervised Learning):** `K-Means` kümeleme algoritması ile veri setindeki kalite grupları incelenmiş; en uygun küme sayısı Elbow Metodu ve Silhouette Skorları ile belirlenip sonuçlar PCA (Temel Bileşen Analizi) ile 2 boyutta görselleştirilmiştir.

---

## 📊 Model Performansı

- **Random Forest Classifier (Kural Tabanlı):** Geliştirilen gözetimli öğrenme modeli üstün bir başarı göstererek **0.9860 ROC-AUC Skoru** elde etmiştir.
- Modelin doğrulama ve öğrenme süreçleri **Öğrenme Eğrisi (Learning Curve)** grafikleriyle desteklenmiş, karmaşıklık matrisi (confusion matrix) ile detaylı performans analizleri yapılmıştır.

---

## 🔬 Analiz Edilen Su Parametreleri

Geliştirdiğim Model, suyun kalitesini ve içilebilirliğini belirlemek için aşağıdaki 9 temel parametreyi işlemektedir:
1. **pH Seviyesi (pH):** Suyun asitlik ve bazlık derecesi.
2. **Sertlik (Hardness):** Kalsiyum ve magnezyum iyonlarının yoğunluğu.
3. **Katılar (Solids):** Toplam çözünmüş katı maddeler (TDS).
4. **Kloramin (Chloramines):** Dezenfeksiyon için kullanılan klor bileşenleri.
5. **Sülfat (Sulfate):** Suda doğal ya da kimyasal yolla bulunan sülfat iyonları.
6. **İletkenlik (Conductivity):** Suyun elektrik akımını iletme kapasitesi.
7. **Organik Karbon (Organic Carbon):** Sudaki toplam organik karbon miktarı.
8. **Trihalometanlar (Trihalomethanes):** Klorlama yan ürünleri.
9. **Bulanıklık (Turbidity):** Suyun berraklık ve ışık geçirgenlik seviyesi.

---

## 📁 Proje Dosya Yapısı

* **`ana_sayfa.py`:** SASKİ kurumsal iletişim bilgilerini, karşılama metinlerini ve özel CSS arkaplan tasarımlarını içeren Streamlit ana navigasyon sayfası.
* **`icilebilirlik_test_ekranı.py`:** Eğitilmiş Random Forest modelini (`rf_kural_su_model.pkl`) ve ölçekleyiciyi (`scaler_kural.pkl`) kullanarak kullanıcıdan alınan parametrelere göre anlık içilebilirlik tahmini yapan interaktif test ekranı.
* **`medyan_puanlamalı.py`:** Veriseti üzerinde dinamik medyan tabanlı kural puanlaması yapan, Random Forest modelini eğiten ve Öğrenme Eğrisi grafiğini oluşturan makine öğrenmesi betiği.
* **`kmeans_model_buyuk.py`:** K-Means kümeleme algoritmalarını çalıştıran, Elbow, Silhouette ve PCA görselleştirmelerini gerçekleştiren analiz betiği.
* **`temizlik_buyukdata.py`:** Verisetinin eksik,aykırı değer analizlerini yapan,Scatter-Histogram gibi grafiklerle görselleştiren veri ön işleme betiği.
---

## 🚀 Kurulum ve Yerel Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için şu adımları izleyebilirsiniz:

**1. Repoyu Klonlayın**
```bash
git clone https://github.com/anilsuu/yazilim_staj.git
cd yazilim_staj
```

**2. Gerekli Kütüphaneleri Yükleyin**
```bash
pip install -r requirements.txt
```

**3. Modelleri Yeniden Eğitin (Opsiyonel)**
```bash
python medyan_puanlamalı.py
python kmeans_model_buyuk.py
```

**4. Streamlit Uygulamasını Başlatın**
```bash
streamlit run ana_sayfa.py
```

---

##  👩‍💻 Geliştirici

**Azra Nilsu**  
*Bilişim Sistemleri Mühendisi*

📩 İletişim, görüş ve önerileriniz için: [su.analizi.sistemi@gmail.com](mailto:su.analizi.sistemi@gmail.com)
