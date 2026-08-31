# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:17:00 2026

@author: nilsu
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer


# ======================================================
# DOSYA YOLU (1 KLASÖR YUKARI)
# ======================================================

print("Çalışılan dizin:", os.getcwd())

# ─────────────────────────────────────────────────────
#  VERİ YÜKLEME
# ─────────────────────────────────────────────────────

df_full = pd.read_excel("water_quality_potability1.xlsx")

COLS =  ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

df = df_full[COLS].copy()

print("Veri boyutu:", df.shape)
print(df.head())

# ─────────────────────────────────────────────────────
#  TEKRAR EDEN VERİ (DUPLICATE) KONTROLÜ
# ─────────────────────────────────────────────────────
# Tüm sütunları baz alarak tamamen aynı olan satırları tespit eder

print("Toplam duplicate kayıt:", df.duplicated().sum())

print("Duplicate gruplarındaki toplam satır:",
      df.duplicated(keep=False).sum())
df = df.drop_duplicates()


def hesaplamainfo(column):
    print(f"Ortalama:{column}",df[column].mean())
    print(f"Medyan:{column}",df[column].median())
    print(f"Standart Sapma:{column}",df[column].std())
    print(f"Eksik değerler:{column}",df[column].isna().sum())
    print(f"Minimum değer:{column}",df[column].min())
    print(f"Maximum değer:{column}",df[column].max())
def medyanladoldurma(column):
    df[column] = df[column].fillna(df[column].median())
    
    
def nansızsütuninfo(column) : 
    print(f"Doldurma sonrası Ortalama:{column}",df[column].mean())
    print(f"Doldurma sonrası Medyan:{column}",df[column].median())
    print(f"Doldurma sonrası Standart Sapma:{column}",df[column].std())
    print(f"Doldurma sonrası Eksik değerler:{column}",df[column].isna().sum())
    
print("-----------------------------------------------------")

# print("Tüm sütunların standart sapması:")
# std_values = df.std(numeric_only=True)
# print(std_values)

print("Sütunların standart sapması:")
std_values = df.std(numeric_only=True)
print(std_values)

print("-----------------------------------------------------")

print("-----------------------------------------------------")

hesaplamainfo("ph")

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


# -----------------------------------------------------

print("-----------------------------------------------------")

print(f"Orijinal veri seti satır sayısı: {len(df)}")

print(f"Silinen toplam aykırı satır sayısı: {len(silinecek_indexler)}")

print(f"Temizlenmiş yeni veri seti satır sayısı: {len(df_temiz)}")

print("-----------------------------------------------------")


kayip_orani=(len(df)-len(df_temiz))/len(df)*100
print("Veri setindeki kayıp oranı:" ,kayip_orani)

# Eski kopuk indeksleri silip baştan 0,1,2,3... diye numaralandırma
df_temiz = df_temiz.reset_index(drop=True)



df_temiz.to_csv("buyuk_veri_su.csv", index=False)
print("\nVeri temizleme tamamlandı ve 'buyuk_veri_su.csv' olarak kaydedildi!")


