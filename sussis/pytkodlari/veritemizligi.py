# -*- coding: utf-8 -*-
"""
@author: nilsu
"""
import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer
import pyodbc
from sqlalchemy import create_engine
import urllib
import missingno as msno
from sklearn.impute import KNNImputer
import scipy.stats 
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

#Sulfate sütununu kNN ile doldurmak için eğitecek 2 başka sütunda seçildi.
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


# 3. İki Sonucu Karşılaştıran Fonksiyon

def karsilastir_aykiri_degerler(iqr_sonuclari, zscore_sonuclari):
    print(f"IQR yöntemi {len(iqr_sonuclari)} adet aykırı değer buldu.")
    print(f"Z-Skoru yöntemi {len(zscore_sonuclari)} adet aykırı değer buldu.")
    
    # Her iki yöntemin de ortak (kesişim) bulduğu aykırı değerlerin indexlerini al
    
    ortak_indexler = iqr_sonuclari.index.intersection(zscore_sonuclari.index)
    
    print(f"Her iki yöntemin ORTAK bulduğu aykırı değer sayısı: {len(ortak_indexler)}")
    
    

    print("-----------------------------------------------------")
    
    # Ortak bulunan bu satırların indexlerini döndürür
    
    return ortak_indexler

# Tüm analiz sürecini tek kalemde yapan birleştirici fonksiyon

def tam_aykiri_analizi(df, column):
    print(f"\n--- {column.upper()} SÜTUNU ANALİZİ ---")
    
    # 1. IQR Analizi
    bulunan_iqr = iqr_aykirianalizi(df, column)
    
    # 2. Z-Skoru Analizi
    bulunan_zscore = zscore_aykirianalizi(df, column)
    
    # 3. Karşılaştırma
    ortak_indexler = karsilastir_aykiri_degerler(bulunan_iqr, bulunan_zscore)
    
    # İleride silmek veya değiştirmek için ortak indexleri geri döndürür.
    return ortak_indexler

sutunlar = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]


# Tüm sütunları tek tek yeni fonksiyona gönderiyor
for col in sutunlar:
    tam_aykiri_analizi(df, col)


# Silinecek indexlerin toplanacağı küme 
# (Aynı satırı iki kere silmeye çalışıp hata almamak için 'set' )

silinecek_indexler = set()

for col in sutunlar:
    
    zscore_outliers = zscore_aykirianalizi(df, col)
    
    # Bulunan aykırı satırların index numaraları kümeye eklenir.
    silinecek_indexler.update(zscore_outliers.index)

# Toplanan tüm aykırı satırları veri setinden silinir
df_temiz = df.drop(index=list(silinecek_indexler)).copy()

# df = df.drop(index=list(silinecek_indexler))


# def iqr_winsorize(df, column):
#     # Çeyreklikler ve IQR hesaplama
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
    
#     alt_sinir = Q1 - 1.5 * IQR
#     ust_sinir = Q3 + 1.5 * IQR
    
#     # clip() metodu: alt_sinir'dan küçükleri alt_sinir'a, ust_sinir'dan büyükleri ust_sinir'a eşitler.
#     df[column] = df[column].clip(lower=alt_sinir, upper=ust_sinir)
    
#     return df

# # Sütun listemiz (hedef değişken olan Potability hariç tüm sayısal sütunlar)
# sutunlar = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
#             "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# # df_temiz veri seti üzerinde döngü ile tüm sütunlara baskılama uyguluyoruz
# for col in sutunlar:
#     df_temiz = iqr_winsorize(df_temiz, col)

# print("Winsorization (Baskılama) işlemi tüm sütunlar için başarıyla tamamlandı.")


print("-----------------------------------------------------")

print(f"Orijinal veri seti satır sayısı: {len(df)}")

print(f"Silinen toplam aykırı satır sayısı: {len(silinecek_indexler)}")

print(f"Temizlenmiş yeni veri seti satır sayısı: {len(df_temiz)}")

print("-----------------------------------------------------")


kayip_orani=(len(df)-len(df_temiz))/len(df)*100
print("Veri setindeki kayıp oranı:" ,kayip_orani)

# Eski kopuk indeksleri silip baştan 0,1,2,3... diye numaralandırır
df_temiz = df_temiz.reset_index(drop=True)


# # #SCALİNG ISLEMLERİ

# #x ve y değişkenlerini tanımlama
# #'hedef_sutun' isimli sütunu y yapıp, geri kalanları x yapar
# x = df_temiz.drop('Potability', axis=1) 
# y = df_temiz['Potability']              


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# # Ölçeklendirme 
# sc = StandardScaler() 

# # Train setinde öğren (fit) ve uygula (transform)
# x_train = sc.fit_transform(x_train)

# # Test setinde SADECE uygula (transform)
# x_test = sc.transform(x_test) 

# # Model Eğitimi
# model = LogisticRegression()
# model.fit(x_train, y_train)

# tahmin=model.predict(x_test)

# # x_test'in içinde birden fazla özellik olduğu için görselleştirme adına 
# # X ekseninde göstermek üzere sadece 0. indeksteki ilk sütunu (özelliği) seç:
# x_gorsel = x_test[:,4]

# # plt.plot yerine plt.scatter kullanın
# plt.scatter(x_gorsel, y_test, color='pink', label='Gerçek Veriler')
# plt.scatter(x_gorsel, tahmin, color='blue', alpha=0.5, label='Model Tahminleri')

# plt.title("Gerçek Değerler ve Tahminler")
# plt.xlabel("Ölçeklendirilmiş Özellik")
# plt.ylabel("Potability (İçilebilirlik)")
# plt.legend()
# plt.show()

# =====================================================
# 1. VERİ BÖLME VE ÖLÇEKLEME (Yorum satırları kaldırıldı ve deney_id atıldı)
# =====================================================

# x değişkeninden hem hedef değişkeni hem de gereksiz 'deney_id' sütununu atıyoruz
x = df_temiz.drop(['Potability', 'deney_id'], axis=1) 
y = df_temiz['Potability']              

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Ölçeklendirme 
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 


# =====================================================
# 2. XGBOOST MODEL EĞİTİMİ VE OPTİMİZASYON
# =====================================================
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

param_grid = {
    'n_estimators': [100, 200, 300],        
    'max_depth': [3, 5, 7],                 
    'learning_rate': [0.01, 0.05, 0.1],     
    'subsample': [0.8, 1.0],                
    'colsample_bytree': [0.8, 1.0]          
}

print("GridSearchCV ile en iyi parametreler aranıyor...")

grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=5, 
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(x_train, y_train)

print("\n--- OPTİMİZASYON SONUÇLARI ---")
print(f"En İyi Parametreler: {grid_search.best_params_}")
print(f"En İyi CV Accuracy Skoru: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(x_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nOptimize Edilmiş Test Accuracy Skoru: {test_accuracy:.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


# =====================================================
# 3. KONTROL AŞAMASI (Küçük harf 'x' ile düzeltildi)
# =====================================================
import pandas as pd
import numpy as np

print("\n--- 1. VERİ SIZINTISI (TARGET LEAKAGE) KONTROLÜ ---")

# x_train artık StandardScaler'dan çıktığı için bir NumPy array. DataFrame'e çeviriyoruz.
if isinstance(x_train, np.ndarray):
    print("x_train bir NumPy dizisi olarak algılandı. DataFrame'e dönüştürülüyor...")
    # x değişkenini tanımlarken kullandığımız sütun isimlerini alıyoruz
    sutun_isimleri = x.columns.tolist() 
    df_check = pd.DataFrame(x_train, columns=sutun_isimleri)
else:
    df_check = x_train.copy()

# Hedef değişkeni ekliyoruz
df_check['HEDEF_y'] = np.array(y_train)

# Korelasyon hesabı
korelasyonlar = df_check.corr()['HEDEF_y'].drop('HEDEF_y').sort_values(ascending=False)
print("\nÖzelliklerin Hedef Değişkenle Korelasyonu:")
print(korelasyonlar)

print("\n-----------------------------------------------------")
print("--- 2. VERİ AYIRMA KONTROLÜ ---")
print(f"Eğitim Seti (x_train) Satır Sayısı: {x_train.shape[0]}")
print(f"Test Seti (x_test) Satır Sayısı: {x_test.shape[0]}")

print("\nEğitim Seti (y_train) Sınıf Dağılımı (Oransal):")
print(pd.Series(np.array(y_train).flatten()).value_counts(normalize=True))

print("\nTest Seti (y_test) Sınıf Dağılımı (Oransal):")
print(pd.Series(np.array(y_test).flatten()).value_counts(normalize=True))


# 4. BASE MODEL + CROSS VALIDATION
# =====================================================

rf = RandomForestClassifier(random_state=42)

# 1. Base Model Eğitimi (Train setiyle eğitilir, Test setiyle ölçülür)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
base_accuracy = accuracy_score(y_test, y_pred_rf)

# 2. Çapraz Doğrulama (Sadece eğitim setinde genelleme gücünü ölçmek için)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, x_train, y_train, cv=kf, scoring='accuracy')

print(f"Random Forest - Base Test Accuracy : {base_accuracy:.4f}")
print(f"Random Forest - CV Ortalama Accuracy: {np.mean(cv_scores):.4f}")
print(f"Random Forest - CV Standart Sapma   : {np.std(cv_scores):.4f}")
print("-" * 50)
print(f"CV Ortalama Accuracy : {np.mean(cv_scores):.4f}")
print(f"Base Model Accuracy : {base_accuracy:.4f}")
print("-" * 50)