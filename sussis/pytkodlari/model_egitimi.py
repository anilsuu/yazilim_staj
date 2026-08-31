                          <# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:53:06 2026

@author: nilsu
"""
import os #Dosya sistemi yollarınnı yönetmek için kullanılır
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer
import pyodbc
from sqlalchemy import create_engine
import urllib
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

df_son = pd.read_csv("temizlenmis_su_kalitesi.csv")



# modelin öğrenmesi için deney_id ve potability sütunu hariç 9 sütunu seçtim.
tum_kimyasal_sutunlar = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",'ph_ideal', 'Turbidity_safe', 'Sulfate_safe', 'Chemistry_reactivity', 
    'solids_to_cond_ratio', 'hardness_sulfate_ratio', 'ph_dev_from_neutral', 
    'pollution_index', 'safety_violations'
]


# Grafikte en altta kalan özellikleri buraya yazdım.Karar vermesinde en az yardımcı olanlar.
silinecek_sutunlar = ['ph_ideal', 'safety_violations', 'Chemistry_reactivity', 'pollution_index'] 
secili_sutunlar = [col for col in tum_kimyasal_sutunlar if col not in silinecek_sutunlar]

# X ve y değişkenlerini tanımlama
X_sade = df_son[secili_sutunlar]
y_sutun_adi = [col for col in df_son.columns if col.lower() == 'potability'][0]
y_sade = df_son[y_sutun_adi]

print("\n--- GÜVENLİK KONTROLÜ ---")
print(f"Kullanılan Özellik Sayısı: {X_sade.shape[1]}")
print("-------------------------\n")



#  Veriyi Bölme (Eğitim ve Test)
# SMOTE KULLANMADAN veriyi böldüm.
x_train, x_test, y_train, y_test = train_test_split(
    X_sade, y_sade, test_size=0.30, random_state=42, stratify=y_sade
)


# Ölçeklendirme (Standardization)
sc = StandardScaler() 

# Train setinde öğren (fit) ve uygula (transform)
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Sınıf dengesizliğini algoritmaların kendi içine ağırlık verdim.
# (Negatif / Pozitif) sınıf oranı. XGBoost için gereklidir.
scale_weight = (y_train == 0).sum() / (y_train == 1).sum() 



print("\n=====================================================")
print("--- 1. VERİ SIZINTISI (TARGET LEAKAGE) KONTROLÜ ---")
print("=====================================================")

# x_train mi X_train mi kullanıldığını otomatik algıla (hata almamak için)

if 'x_train' in locals():
    aktif_x_train = x_train
    aktif_x_test = x_test
else:
    raise ValueError("Eğitim verisi (x_train veya X_train) bulunamadı!")

# NumPy dizisi ise (StandardScaler uygulandıysa) DataFrame'e çevir

if isinstance(aktif_x_train, np.ndarray):
    num_cols = aktif_x_train.shape[1]
    print(f"Eğitim setindeki sütun (özellik) sayısı: {num_cols}")
    
    # Sütun sayısına göre isimleri belirle
    # Eğer 10 sütun varsa 'deney_id' hala içeride demektir.
    
    if num_cols == 9:
        cols = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
                "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
    else:
        # Ne olduğu bilinmiyorsa geçici isim ver
        cols = [f"Sutun_{i}" for i in range(num_cols)]
        
    df_check = pd.DataFrame(aktif_x_train, columns=cols)
else:
    df_check = aktif_x_train.copy()
    print("Eğitim setindeki sütunlar:", df_check.columns.tolist())

# Hedef değişkeni güvenle tabloya ekle
df_check['HEDEF_POTABILITY'] = np.array(y_train).flatten()

# Korelasyon hesabı (Hedef değişkenle diğer sütunlar arasındaki matematiksel ilişki)
korelasyonlar = df_check.corr()['HEDEF_POTABILITY'].drop('HEDEF_POTABILITY').sort_values(ascending=False)
print("\nÖzelliklerin Hedef Değişkenle (Potability) Korelasyonu:")
print(korelasyonlar)

# =====================================================
# Sızıntı engellemek için sütunları tekrar filtreleyerk devam ediyorum.
# =====================================================


# =====================================================
# GRID SEARCH (SINIFLANDIRMA İÇİN GÜNCELLENDİ)
# =====================================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'ccp_alpha': [0.0, 0.01]
}

print("\nGridSearch çalışıyor...")

# RandomForestRegressor yerine RandomForestClassifier kullanıldı
grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                    param_grid,
                    cv=3,
                    n_jobs=-1,
                    scoring='accuracy')

grid.fit(x_train_sc, y_train)

best_rf = grid.best_estimator_
y_pred = best_rf.predict(x_test_sc)
best_acc = accuracy_score(y_test, y_pred)

print("En iyi parametreler:", grid.best_params_)
print(f"Optimize Doğruluk (Accuracy): {best_acc:.4f}")
print("-" * 50)

# =====================================================
#  ÖZELLİK ÖNEM SIRALAMASI
# =====================================================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(10, 6))
plt.barh(np.array(secili_sutunlar)[indices], importances[indices])
plt.title("Özellik Önem Sıralaması")
plt.xlabel("Önem")
plt.tight_layout()
plt.show()




# =====================================================
# MODEL EĞİTİMLERİ VE TEST (CLASS WEIGHTS İLE)
# =====================================================

# --- A. CATBOOST MODELİ ---
print("\n=====================================================")
print("1. CATBOOST MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")


cat_model = CatBoostClassifier(
    auto_class_weights='Balanced',  # Dengesizliği, SMOTE yerine matematiksel cezayla çözer
    iterations=500,
    learning_rate=0.02,
    depth=4,
    verbose=False,   # Eğitilirken ekrana binlerce satır yazı basmasını engeller
    random_state=42
)


#Modeli Eğit ve Test Et
print("CatBoost eğitiliyor (SMOTE YOK)...")
cat_model.fit(x_train, y_train) # CatBoost ölçeklendirilmemiş veriyle de çok iyi çalışır
y_pred_cat = cat_model.predict(x_test)

print(f"CatBoost Test Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_cat))

# Özellik önem düzeylerini al ve bir DataFrame'e çevir
feature_importances = pd.DataFrame({
    'Feature': secili_sutunlar,
    'Importance': cat_model.get_feature_importance()
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
# Gelecekte hata vermemesi için hue eklendi
sns.barplot(x='Importance', y='Feature', data=feature_importances, hue='Feature', palette='viridis', legend=False)
plt.title('CatBoost - Özellik Önem Düzeyleri (Feature Importance)')
plt.xlabel('Önem Skoru')
plt.ylabel('Özellikler')
plt.tight_layout()
plt.show()


# --- LIGHTGBM MODELİ ---
print("\n=====================================================")
print("2. LIGHTGBM MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")


lgbm_model = LGBMClassifier(
    class_weight='balanced', # Dengesizliği çözen kritik parametre
    n_estimators=100,
    learning_rate=0.02,
    max_depth=4,
    random_state=42,
    verbose=-1
)

lgbm_model.fit(x_train_sc, y_train)
y_pred_lgbm = lgbm_model.predict(x_test_sc)


#Modeli Eğit ve Test Et
print("LightGBM eğitiliyor (SMOTE YOK)...")
lgbm_model.fit(x_train_sc, y_train) # LightGBM ölçeklendirilmemiş veriyle de çok iyi çalışır
y_pred_lgbm = lgbm_model.predict(x_test_sc)

# Özellik önem düzeylerini al ve bir DataFrame'e çevir
feature_importances_lgbm = pd.DataFrame({
    'Feature': secili_sutunlar,
    'Importance': lgbm_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
# Gelecekte hata vermemesi için hue eklendi
sns.barplot(x='Importance', y='Feature', data=feature_importances_lgbm, hue='Feature', palette='viridis', legend=False)
plt.title('LightGBM - Özellik Önem Düzeyleri (Feature Importance)')
plt.xlabel('Önem Skoru')
plt.ylabel('Özellikler')
plt.tight_layout()
plt.show()

print(f"LightGBM Test Accuracy: {accuracy_score(y_test, y_pred_lgbm):.4f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_lgbm))



# Aşırı öğrenmeyi engelleyecek parametre ızgarası (Grid)
# max_depth düşük tuttum ,ezberlemeyi önlemek için..
# subsample ve colsample_bytree ile modelin her adımda verinin/sütunların sadece bir kısmını görmesi sağlanır.

# ---  XGBOOST MODELİ ---
print("\n=====================================================")
print("3. XGBOOST MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_weight, # Dengesizliği çözen kritik parametre
    n_estimators=100,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,             
    min_child_weight=20,      # (Aşırı dallanmayı engeller)
    gamma=1.5,               # (Gereksiz dalları budar)
    eval_metric='logloss'
)
xgb_model.fit(x_train_sc, y_train)
y_pred_xgb = xgb_model.predict(x_test_sc)

print(f"XGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_xgb))

# =====================================================
# GÖRSELLEŞTİRME (EN İYİ MODELİN ÖZELLİK ÖNEM DÜZEYİ)
# =====================================================
# CatBoost genellikle bu tarz verilerde en iyi sonucu verir, onun grafiğini çizdirdim


print("\n--- MODEL SÜRECİ BAŞARIYLA TAMAMLANDI! ---")


#BURADAKİ EGİTİM DOĞRULUĞU:0.7201 , TEST DOĞRULUĞU:0.6153 , ARADAKİ FARK:0.1048
import xgboost as xgb
from sklearn.metrics import accuracy_score

print("\n=====================================================")
print("--- DÜZENLİLEŞTİRİLMİŞ OVERFITTING KONTROLÜ ---")
print("=====================================================")

# Modeli aşırı ezberden kaçınacak şekilde sert kurallarla kuruyoruz
xgb_saglam_kontrol = xgb.XGBClassifier(
    scale_pos_weight=scale_weight, 
    n_estimators=100,            # Ağaç sayısını 300'den 100'e düşürdük
    learning_rate=0.02,          # Öğrenmeyi yavaşlattık
    max_depth=4,                 # Derinliği 6'dan 4'e indirdik (Ezberi engeller)
    min_child_weight=20,         # Bir yaprakta en az 10 örnek olmasını şart koştuk
    gamma=2.0,                   # Yeni dallar açılmasına yüksek ceza getirdik
    subsample=0.7,               # Her ağaç verinin %70'ini görsün
    colsample_bytree=0.7,        # Her ağaç özelliklerin %70'ini görsün
    reg_alpha=1.0,               # L1 Regularization (Gereksiz ağırlıkları sıfırlar)
    reg_lambda=1.0,              # L2 Regularization (Aşırı büyük katsayıları engeller)
    random_state=42,
    eval_metric='logloss'
)

# Modeli Eğitme Aşaması
xgb_saglam_kontrol.fit(x_train_sc, y_train)

# Eğitim (Train) Skoru
y_train_pred = xgb_saglam_kontrol.predict(x_train_sc)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test Skoru
y_test_pred = xgb_saglam_kontrol.predict(x_test_sc)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Sonuçları Yazdır
print(f"EĞİTİM (Train) Doğruluğu : {train_accuracy:.4f}")
print(f"TEST Doğruluğu           : {test_accuracy:.4f}")
print("-" * 50)
print(f"Yeni Uçurum (Fark)       : {(train_accuracy - test_accuracy):.4f}")



# =====================================================
# LEARNING CURVE (SINIFLANDIRMA SKORU İLE)
# =====================================================
train_sizes, train_scores, test_scores = learning_curve(
    best_rf, x_train_sc, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation")
plt.title("Learning Curve")
plt.xlabel("Veri Miktarı")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()



