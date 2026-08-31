# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:39:56 2026

@author: nilsu
"""
# -*- coding: utf-8 -*-

import os #Dosya sistemi yollarınnı yönetmek için kullanılır
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =====================================================
# Veriyi Yükleme
# =====================================================
df = pd.read_csv("buyuk_veri_su.csv")

tum_kimyasal_sutunlar = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

X = df[tum_kimyasal_sutunlar].copy()

# K-Means mesafe bazlı çalıştığı için ölçeklendirme kesin lazım
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# En uygun küme sayısını bulma (ELBOW & SILHOUETTE)
# =====================================================
k_degerleri = range(2, 7)
wcss = []
silhouette_degerleri = []

for k in k_degerleri:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette_degerleri.append(score)

# Grafik: Elbow ve Silhouette Skoru
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(k_degerleri, wcss, marker='o', color='b')
ax[0].set_title("Elbow Metodu (WCSS)")
ax[0].set_xlabel("Küme Sayısı (k)")
ax[0].set_ylabel("Inertia")
ax[0].grid(True)

ax[1].plot(k_degerleri, silhouette_degerleri, marker='s', color='g')
ax[1].set_title("Silhouette Skorları")
ax[1].set_xlabel("Küme Sayısı (k)")
ax[1].set_ylabel("Silhouette Skoru")
ax[1].grid(True)

plt.tight_layout()
plt.show()

# =====================================================
# K-MEANS MODELİNİ KURMA VE ETİKET ATAMA (k=2)
# =====================================================
# 2 küme (İçilebilir / İçilemez veya Kalite Seviyeleri)
kmeans_final = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

print("--- K-Means Küme Dağılımı ---")
print(df['Cluster'].value_counts())

# Küme ortalama değerlerini inceleme
kume_ortalamalari = df.groupby('Cluster')[tum_kimyasal_sutunlar].mean()
print("\n--- Kümelerin Kimyasal Özellik Ortalamaları ---")
print(kume_ortalamalari.T)

# =====================================================
# Scatter Grafik (PCA ile)
# =====================================================
pca = PCA(n_components=2)
pca_sonuc = pca.fit_transform(X_scaled)
df['PCA1'] = pca_sonuc[:, 0]
df['PCA2'] = pca_sonuc[:, 1]

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, x='PCA1', y='PCA2', 
    hue='Cluster', palette=['#1f77b4', '#ff7f0e'], alpha=0.7
)
plt.title("K-Means Kümelerinin PCA ile Görselleştirilmesi")
plt.xlabel("Temel Bileşen 1")
plt.ylabel("Temel Bileşen 2")
plt.grid(True)
plt.show()

# =====================================================
# Web tarafı için kaydetme dosyası
# =====================================================
joblib.dump(scaler, "scaler_kmeans.pkl")
joblib.dump(kmeans_final, "kmeans_su_model.pkl")
print("\nScaler ve K-Means modeli başarıyla kaydedildi!")






















































# df_son = pd.read_csv("buyuk_veri_su.csv")



# # (X) - 9 kimyasal parametre
# tum_kimyasal_sutunlar = [
#     "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
#     "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
# ]
# X = df_son[tum_kimyasal_sutunlar]

# # Hedef Değişken (y) - Sadece içilebilirlik sütunu (0 ve 1)

# y = df_son["Potability"] 

# # Güvenlik Kontrolü
# print("--- GÜVENLİK KONTROLÜ ---")
# print(f"X Boyutu: {X.shape}")        # Örn: (N, 9)
# print(f"y Boyutu: {y.shape}")        # Örn: (N,)
# print(f"Sınıf Dağılımı:\n{y.value_counts()}")
# print("-------------------------\n")

# # Denetimsiz öğrenme için train_test_split (stratify olmadan)
# x_train, x_test = train_test_split(df_son, test_size=0.2, random_state=42)

# sc = StandardScaler()
# x_train_sc = sc.fit_transform(x_train)
# x_test_sc = sc.transform(x_test)

# print("\n=====================================================")
# print("--- 1. VERİ SIZINTISI (TARGET LEAKAGE) KONTROLÜ ---")
# print("=====================================================")


# # =====================================================
# # Sızıntı engellemek için sütunları tekrar filtreleyerk devam ediyorum.
# # =====================================================


# import pandas as pd
# import joblib
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier

# # 1. Veriyi Oku ve Ölçeklendir
# df = pd.read_csv("buyuk_veri_su.csv")
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df)

# # 2. K-Means ile 2 Küme Oluştur
# kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
# df['Cluster'] = kmeans.fit_predict(X_scaled)

# # 3. Küme Merkezlerini İnceleyip Etiket Ata
# # (Ortalamalara göre daha kaliteli olan kümeye 1, diğerine 0 diyebilirsiniz)
# # Örnek: Cluster 0 -> İçilebilir (1), Cluster 1 -> İçilemez (0)
# df['Potability_Tahmin'] = df['Cluster'].map({0: 1, 1: 0})

# # 4. Web Arayüzü İçin Sınıflandırıcı Eğit
# X = df.drop(columns=['Cluster', 'Potability_Tahmin'])
# y = df['Potability_Tahmin']

# clf = XGBClassifier(random_state=42)
# clf.fit(scaler.transform(X), y)

# # 5. Modeli ve Scaler'ı Kaydet
# joblib.dump(clf, "k_means_su_model.joblib")
# joblib.dump(scaler, "su_scaler.joblib")
# print("K-Means tabanlı model hazır ve kaydedildi!")

