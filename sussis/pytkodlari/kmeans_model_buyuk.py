# -*- coding: utf-8 -*-
"""
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

# K-Means mesafe bazlı çalıştığı için ölçeklendirme zorunlu

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



# Modeli ve Scaler'ı Kaydet
joblib.dump(kmeans_final, "k_means_su_model.joblib")
joblib.dump(scaler, "su_scaler.joblib")
print("K-Means tabanlı model hazır ve kaydedildi!")

