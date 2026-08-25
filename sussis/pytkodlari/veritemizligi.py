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
import scipy.stats 
from scipy.stats import zscore


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




print("-----------------------------------------------------")

# print("Tüm sütunların standart sapması:")
# std_values = df.std(numeric_only=True)
# print(std_values)

print("Sütunların standart sapması:")
std_values = df.drop(columns=["deney_id"]).std(numeric_only=True)
print(std_values)


print("-----------------------------------------------------")

hesaplamainfo("ph")
# # ph sütunundaki null değerleri 'ph'sütununu medyanı ile doldurur.
# df["ph"] = df["ph"].fillna(df["ph"].median())
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


print("-----------------------------------------------------")

print(f"Orijinal veri seti satır sayısı: {len(df)}")

print(f"Silinen toplam aykırı satır sayısı: {len(silinecek_indexler)}")

print(f"Temizlenmiş yeni veri seti satır sayısı: {len(df_temiz)}")

print("-----------------------------------------------------")


kayip_orani=(len(df)-len(df_temiz))/len(df)*100
print("Veri setindeki kayıp oranı:" ,kayip_orani)

# Eski kopuk indeksleri silip baştan 0,1,2,3... diye numaralandırır
df_temiz = df_temiz.reset_index(drop=True)


# #ENCODED  VERİ SETİNİN YARISINI YARIN YAPACAĞIM.

