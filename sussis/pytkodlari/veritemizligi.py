# -*- coding: utf-8 -*-
"""
@author: nilsu
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer
import pyodbc
from sqlalchemy import create_engine
import urllib
import missingno as msno
from sklearn.impute import KNNImputer
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

#Server ve database bilgisi
server = 'LAPTOP-MNJN06EU\\NILSS'
database = 'water_usability'

#Bağlantı metni -> güvenilirliği için TrustedConnections=yes olmalı 
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'


# Bağlantı metnini URL formatına çevirip motoru (engine) oluştur
params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Sql'den tablo Python'a 
sql_query = "SELECT * FROM Drinking_water1"
df = pd.read_sql(sql_query, engine)

print(df.head()) # İlk 5 satırı yazdır

print("Veri boyutu:", df.shape)

df=df.drop(["Carcinogenics","medical_waste"],axis=1)

# msno.matrix(df.sample(len(df)))

# DUPLICATE(TEKRAR EDEN VERİ) KONTROLÜ
print("Toplam duplicate kayıt:", df.duplicated().sum())


def hesaplamainfo(column):
    print(f"Ortalama:{column}",df[column].mean())
    print(f"Medyan:{column}",df[column].median())
    print(f"Standart Sapma:{column}",df[column].std())
    print(f"Eksik değerler:{column}",df[column].isna().sum())
    
def medyanladoldurma(column):
    df[column] = df[column].fillna(df[column].median())
    
    
def nansızsütuninfo(column) : 
    print(f"Doldurma sonrası Ortalama:{column}",df[column].mean())
    print(f"Doldurma sonrası Medyan:{column}",df[column].median())
    print(f"Doldurma sonrası Standart Sapma:{column}",df[column].std())
    print(f"Doldurma sonrası Eksik değerler:{column}",df[column].isna().sum())
    

# Tüm sütunları baz alarak tamamen aynı olan satırları tespit edip,siler.
print("Duplicate gruplarındaki toplam satır:",
      df.duplicated(keep=False).sum())
df = df.drop_duplicates()


print("ph eksik oranı:", df["ph"].isna().mean() * 100)
print("Sulfate eksik oranı:", df["Sulfate"].isna().mean() * 100)
print("Trihalomethanes eksik oranı:", df["Trihalomethanes"].isna().mean() * 100)


print("-----------------------------------------------------")

# print("Tüm sütunların standart sapması:")
# std_values = df.std(numeric_only=True)
# print(std_values)

print("Sütunların standart sapması:")
std_values = df.drop(columns=["deney_id"]).std(numeric_only=True)
print(std_values)


print("-----------------------------------------------------")

hesaplamainfo("ph")

medyanladoldurma("ph")

nansızsütuninfo("ph")

#Histogram Grafiği
plt.hist(df["ph"], bins=20)
plt.xlabel("ph")
plt.ylabel("Frekans")
plt.title("ph Değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Hardness")

#Histogram Grafiği
plt.hist(df["Hardness"],bins=20)
plt.xlabel("Hardness")
plt.ylabel("frekans")
plt.title("Hardness değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Solids")

#Histogram Grafiği
plt.hist(df["Solids"], bins=20)
plt.xlabel("Solids")
plt.ylabel("frekans")
plt.title("Solids Değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Chloramines")

#Histogram Grafiği
plt.hist(df["Chloramines"], bins=20)
plt.xlabel("Chloramines")
plt.ylabel("frekans")
plt.title("Chloramines Değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Sulfate")

#Sulfate sütununu kNN ile doldurmak için eğitecek 2 başka sütun seçtim.
#Kopya bir dataframe oluşturuldu.
df_knn=df.filter(["Sulfate","ph","Hardness"],axis=1).copy()


scaler=MinMaxScaler()
df_knn = pd.DataFrame(scaler.fit_transform(df_knn), columns = df_knn.columns)

# Daha doğru sonuç için d
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)

# print(df_knn_imputed)
df_imputed = pd.DataFrame(scaler.inverse_transform(df_knn_imputed), columns=df_knn.columns)

# Doldurulmuş Sulfate sütununu, tekrar main dataframe eklendi.
df["Sulfate"] = df_imputed["Sulfate"]


# medyanladoldurma("Sulfate")
nansızsütuninfo("Sulfate")

#Histogram Grafiği
plt.hist(df["Sulfate"],bins=20)
plt.xlabel("Sulfate")
plt.ylabel("frekans")
plt.title("Sulfate değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Conductivity")

#Histogram Grafiği
plt.hist(df["Conductivity"],bins=20)
plt.xlabel("Conductivity")
plt.ylabel("frekans")
plt.title("Conductivity değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Organic_carbon")

#Histogram Grafiği
plt.hist(df["Organic_carbon"],bins=20)
plt.xlabel("Organic_carbon")
plt.ylabel("frekans")
plt.title("Organic_carbon değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Trihalomethanes")

medyanladoldurma("Trihalomethanes")

nansızsütuninfo("Trihalomethanes")

#Histogram Grafiği
plt.hist(df["Trihalomethanes"].dropna(), bins=20)
plt.xlabel("Trihalomethanes")
plt.ylabel("Frekans")
plt.title("Trihalomethanes Değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

hesaplamainfo("Turbidity")
#Histogram Grafiği
plt.hist(df["Turbidity"], bins=20)
plt.xlabel("Turbidity")
plt.ylabel("frekans")
plt.title("Turbidity Değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

# # ─────────────────────────────────────────────────────
# # 4. GELİŞMİŞ EKSİK VERİ DOLDURMA (MICE / IterativeImputer)
# # ─────────────────────────────────────────────────────
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# print("\nEksik veriler makine öğrenmesi (MICE) ile dolduruluyor...")
# # Modelin öğreneceği (X) özellikleri belirliyoruz. "Potability" ve "deney_id" hariç.
# x_sutunlari = [col for col in df.columns if col not in ['Potability', 'deney_id']]

# X_raw = df[x_sutunlari]
# y = df['Potability']

# # KNN veya Median yerine diğer sütunlardan tahmin yaparak eksikleri doldurur
# mice_imputer = IterativeImputer(max_iter=15, random_state=42)
# X_imputed = pd.DataFrame(mice_imputer.fit_transform(X_raw), columns=x_sutunlari)


# ─────────────────────────────────────────────────────
# ADIM 5 — AYKIRI DEĞER (IQR ve Z-SCORE)
# ─────────────────────────────────────────────────────

def iqr_aykirianalizi(df, column):
    Q1  = df[column].quantile(0.25)
    Q3  = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Sadece string yerine df[column] üzerinden karşılaştırma yapar
    # Sonucu direkt bir DataFrame olarak almak için df[...] içine aldım
    
    iqr_outliers = df[(df[column] < lower) | (df[column] > upper)]
    
    return iqr_outliers


def zscore_aykirianalizi(df, column):
    ortalama = df[column].mean()
    std_sapma = df[column].std()
    z_skorlari = (df[column] - ortalama) / std_sapma
    zscore_outliers = df[z_skorlari.abs() > 3]
    
    return zscore_outliers


#  İki Sonucu Karşılaştıran Fonksiyon

def karsilastir_aykiri_degerler(iqr_sonuclari, zscore_sonuclari):
    print(f"IQR yöntemi {len(iqr_sonuclari)} adet aykırı değer buldu.")
    print(f"Z-Skoru yöntemi {len(zscore_sonuclari)} adet aykırı değer buldu.")
    
    # Her iki yöntemin de ortak (kesişim) bulduğu aykırı değerlerin indexlerini alır.
    
    ortak_indexler = iqr_sonuclari.index.intersection(zscore_sonuclari.index)
    
    print(f"Her iki yöntemin ORTAK bulduğu aykırı değer sayısı: {len(ortak_indexler)}")
    
    

    print("-----------------------------------------------------")
    
    # Ortak bulunan bu satırların indexlerini döndürür
    return ortak_indexler


# Tüm analiz sürecini tek kodda yapan  fonksiyon

def tam_aykiri_analizi(df, column):
    
    #Sütunu büyük harfe dönüştürüp ekrana basıyor.
    print(f"\n--- {column.upper()} SÜTUNU ANALİZİ ---")
    
    # IQR Analizi
    bulunan_iqr = iqr_aykirianalizi(df, column)
    
    # Z-Skoru Analizi
    bulunan_zscore = zscore_aykirianalizi(df, column)
    
    # Karşılaştırma
    ortak_indexler = karsilastir_aykiri_degerler(bulunan_iqr, bulunan_zscore)
    
    # İleride silmek veya değiştirmek için ortak indexleri geri döndürür.
    return ortak_indexler

sutunlar = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# Tüm sütunları tek tek yeni fonksiyona işler.
for col in sutunlar:
    tam_aykiri_analizi(df, col)


# Silinecek indexlerin toplanacağı küme 
# (Aynı satırı iki kere silmeye çalışınca hata aldım o yüzden, 'set' )

silinecek_indexler = set()

for col in sutunlar:
    
    zscore_outliers = zscore_aykirianalizi(df, col)
    
    # Bulunan aykırı satırların index numaraları kümeye eklenir.
    silinecek_indexler.update(zscore_outliers.index)

# Toplanan tüm aykırı satırları veri setinden silinir
df_temiz = df.drop(index=list(silinecek_indexler)).copy()

print("-----------------------------------------------------")

print(f"Orijinal veri seti satır sayısı: {len(df)}")

print(f"Silinen toplam aykırı satır sayısı: {len(silinecek_indexler)}")

print(f"Temizlenmiş yeni veri seti satır sayısı: {len(df_temiz)}")

print("-----------------------------------------------------")


kayip_orani=(len(df)-len(df_temiz))/len(df)*100
print("Veri setindeki kayıp oranı:" ,kayip_orani)

# Eski kopuk indeksleri silip baştan 0,1,2,3... diye numaralandırma
df_temiz = df_temiz.reset_index(drop=True)


# =====================================================
# FEATURE ENGINEERING ADIMI
# =====================================================
df_fe = df_temiz.copy()

# ideal ph aralığı 6.5-8.5 olarak yeni ph_ideal üretmek
df_fe['ph_ideal'] = df_fe['ph'].between(6.5, 8.5).astype(int)

# ideal bulanıklık 5 değerinin altında olmalı
df_fe["Turbidity_safe"]=(df["Turbidity"]<5.0).astype(int)

# ideal Sülfatın değerinin 400den küçük olması gerekiyor 
df_fe["Sulfate_safe"]=(df["Sulfate"]<400).astype(int)

# Kimyasal etki Trihalometan oluşumu kloramin ve organik karbonun reaksiyonuyla artar
df_fe["Chemistry_reactivity"]=df_fe["Chloramines"]*df_fe["Organic_carbon"]

# İletkenlik ve Çözünmüş Katı Madde (Solids/Conductivity) ilişkisi
# 0'a bölme hatasını önlemek için küçük bir epsilon (1e-6) eklenir
df_fe['solids_to_cond_ratio'] = df_fe['Solids'] / (df_fe['Conductivity'] + 1e-6)

# Sertlik ve Sülfat oranı (Mineral yoğunluk dengesi)
df_fe['hardness_sulfate_ratio'] = df_fe['Hardness'] / (df_fe['Sulfate'] + 1e-6)

# 3. İdeal pH'tan Uzaklık (Non-linear Sapma)
# Suyun ideal pH olan 7.0'dan ne kadar saptığını ölçen mutlak sapma
df_fe['ph_dev_from_neutral'] = np.abs(df_fe['ph'] - 7.0)


# 4. Genel Kirletici İndeksi (Agregasyon)
# Organik karbon ve bulanıklığın birleşik kirlilik etkisi
df_fe['pollution_index'] = df_fe['Organic_carbon'] * df_fe['Turbidity']


# 5. Güvenlik İhlal Skoru (Ne kadar çok kural ihlal edilirse o kadar içilemez)
df_fe['safety_violations'] = (
    (1 - df_fe['ph_ideal']) + 
    (1 - df_fe['Turbidity_safe']) + 
    (1 - df_fe['Sulfate_safe'])
)

print(f"Eski Özellik Sayısı: {df_temiz.shape[1] - 1}")
print(f"Yeni Özellik Sayısı: {df_fe.shape[1] - 1}")



# # Temizlenmiş veriyi dışarı aktarma (İsteğe bağlı)
# df_fe = X_imputed.copy()
# df_fe['Potability'] = y
# df_fe.to_csv("temizlenmis_su_kalitesi_final.csv", index=False)
# print("Temiz ve doldurulmuş veri 'temizlenmis_su_kalitesi_final.csv' olarak kaydedildi.")


# # ─────────────────────────────────────────────────────
# # 4. GELİŞMİŞ EKSİK VERİ DOLDURMA (MICE / IterativeImputer)
# # ─────────────────────────────────────────────────────
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# print("\nEksik veriler makine öğrenmesi (MICE) ile dolduruluyor...")
# # Modelin öğreneceği (X) özellikleri belirliyoruz. "Potability" ve "deney_id" hariç.
# x_sutunlari = [col for col in df_fe.columns if col not in ['Potability', 'deney_id']]

# X_raw = df_fe[x_sutunlari]
# y = df_fe['Potability']

# # KNN veya Median yerine diğer sütunlardan tahmin yaparak eksikleri doldurur
# mice_imputer = IterativeImputer(max_iter=15, random_state=42)
# X_imputed = pd.DataFrame(mice_imputer.fit_transform(X_raw), columns=x_sutunlari)

# # Temizlenmiş veriyi dışarı aktarma (İsteğe bağlı)
# df_son = X_imputed.copy()
# df_son['Potability'] = y
# df_son.to_csv("temizlenmis_su_kalitesi_final.csv", index=False)
# print("Temiz ve doldurulmuş veri 'temizlenmis_su_kalitesi_final.csv' olarak kaydedildi.")







# Temizlenmiş veriyi CSV olarak dışa aktar
# -----------------------------------------------------
df_fe.to_csv("temizlenmis_su_kalitesi.csv", index=False)
print("\nVeri temizleme tamamlandı ve 'temizlenmis_su_kalitesi.csv' olarak kaydedildi!")





# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# # Ölçeklendirme 
# sc = StandardScaler() 

# # Train setinde öğren (fit) ve uygula (transform)
# x_train = sc.fit_transform(x_train)

# # Test setinde SADECE uygula (transform)
# x_test = sc.transform(x_test) 


# # BASE MODEL + CROSS VALIDATION

# print(" =====================================================")

# rf = RandomForestClassifier(random_state=42)

# kf = KFold(n_splits=10, shuffle=True, random_state=42)
# cv_scores = cross_val_score(rf, x, y, cv=kf, scoring='accuracy')

# rf.fit(x_train, y_train)
# base_accuracy = accuracy_score(y_test, rf.predict(x_test))

# print(f"CV Ortalama Accuracy : {np.mean(cv_scores):.4f}")
# print(f"Base Model Accuracy : {base_accuracy:.4f}")
# print("-" * 50)


# # Temel XGBoost Sınıflandırıcısını Tanımla
# xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# # Aşırı öğrenmeyi engelleyecek parametre ızgarası (Grid)
# # max_depth düşük tutularak ezberleme önlenir.
# # subsample ve colsample_bytree ile modelin her adımda verinin/sütunların sadece bir kısmını görmesi sağlanır.
# param_grid = {
#     'n_estimators': [100, 200, 300],        # Ağaç sayısı
#     'max_depth': [3, 5, 9],                 # Ağaç derinliği (Düşük overfitting'i engeller)
#     'learning_rate': [0.01, 0.05, 0.1],     # Öğrenme oranı
#     'subsample': [0.8, 1.0],                # Her ağaç için kullanılacak satır oranı
#     'colsample_bytree': [0.8, 1.0]          # Her ağaç için kullanılacak sütun oranı
# }

# print("GridSearchCV ile en iyi parametreler aranıyor... (Bu işlem birkaç dakika sürebilir)")

# # GridSearchCV'yi Başlat (5 katlı Çapraz Doğrulama ile)
# grid_search = GridSearchCV(
#     estimator=xgb_model, 
#     param_grid=param_grid, 
#     scoring='accuracy', 
#     cv=5, 
#     n_jobs=-1, # İşlemcinin tüm çekirdeklerini kullanır (Hızlandırır)
#     verbose=1
# )

# # Modeli Eğit
# grid_search.fit(x_train, y_train)

# # En İyi Parametreleri ve Çapraz Doğrulama Skorunu Yazdır
# print("\n--- OPTİMİZASYON SONUÇLARI ---")
# print(f"En İyi Parametreler: {grid_search.best_params_}")
# print(f"En İyi CV Accuracy Skoru: {grid_search.best_score_:.4f}")

# # Test Seti Üzerinde Tahmin ve Değerlendirme
# best_xgb = grid_search.best_estimator_
# y_pred = best_xgb.predict(x_test)

# test_accuracy = accuracy_score(y_test, y_pred)
# print(f"\nOptimize Edilmiş Test Accuracy Skoru: {test_accuracy:.4f}")
# print("\nSınıflandırma Raporu (Precision, Recall, F1-Score):")
# print(classification_report(y_test, y_pred))


# print("\n=====================================================")
# print("--- 1. VERİ SIZINTISI (TARGET LEAKAGE) KONTROLÜ ---")
# print("=====================================================")

# # x_train mi X_train mi kullanıldığını otomatik algıla (hata almamak için)

# if 'x_train' in locals():
#     aktif_x_train = x_train
#     aktif_x_test = x_test
# else:
#     raise ValueError("Eğitim verisi (x_train veya X_train) bulunamadı!")

# # NumPy dizisi ise (StandardScaler uygulandıysa) DataFrame'e çevir

# if isinstance(aktif_x_train, np.ndarray):
#     num_cols = aktif_x_train.shape[1]
#     print(f"Eğitim setindeki sütun (özellik) sayısı: {num_cols}")
    
#     # Sütun sayısına göre isimleri belirle
#     # Eğer 10 sütun varsa 'deney_id' hala içeride demektir.
    
#     if num_cols == 9:
#         cols = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
#                 "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
#     else:
#         # Ne olduğu bilinmiyorsa geçici isim ver
#         cols = [f"Sutun_{i}" for i in range(num_cols)]
        
#     df_check = pd.DataFrame(aktif_x_train, columns=cols)
# else:
#     df_check = aktif_x_train.copy()
#     print("Eğitim setindeki sütunlar:", df_check.columns.tolist())

# # Hedef değişkeni güvenle tabloya ekle
# df_check['HEDEF_POTABILITY'] = np.array(y_train).flatten()

# # Korelasyon hesabı (Hedef değişkenle diğer sütunlar arasındaki matematiksel ilişki)
# korelasyonlar = df_check.corr()['HEDEF_POTABILITY'].drop('HEDEF_POTABILITY').sort_values(ascending=False)
# print("\nÖzelliklerin Hedef Değişkenle (Potability) Korelasyonu:")
# print(korelasyonlar)

# print("\n🔍 ANALİZ İPUCU:")
# print("Korelasyon değerlerinde 0.20'nin veya -0.20'nin üzerinde aşırı yüksek bir sütun var mı?")
# print("Özellikle 'deney_id' veya 'Sutun_0' gibi bir değişkenin korelasyonu kol geziyorsa, %98'lik skorun sırrı odur!")

# print("\n=====================================================")
# print("--- 2. VERİ AYIRMA (DATA SPLIT) KONTROLÜ ---")
# print("=====================================================")

# print(f"Eğitim Seti Satır Sayısı: {aktif_x_train.shape[0]}")
# print(f"Test Seti Satır Sayısı:   {aktif_x_test.shape[0]}")

# print("\nEğitim Seti (y_train) Sınıf Dağılımı:")
# print(pd.Series(np.array(y_train).flatten()).value_counts(normalize=True))

# print("\nTest Seti (y_test) Sınıf Dağılımı:")
# print(pd.Series(np.array(y_test).flatten()).value_counts(normalize=True))
