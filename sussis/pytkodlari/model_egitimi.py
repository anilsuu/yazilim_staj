# -*- coding: utf-8 -*-
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



df_fe = pd.read_csv("temizlenmis_su_kalitesi.csv")



# =====================================================
# Sızıntı engellemek için sütunları tekrar filtreleyerk devam ediyorum.
# =====================================================

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

# X ve y değişkenlerini tanımlama,güncelliyoruz
X_sade = df_fe[secili_sutunlar]
y_sutun_adi = [col for col in df_fe.columns if col.lower() == 'potability'][0]
y_sade = df_fe[y_sutun_adi]

print(f"\n--- GÜVENLİK KONTROLÜ ---")
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

# =====================================================
# 3. MODEL EĞİTİMLERİ VE TEST (CLASS WEIGHTS İLE)
# =====================================================

# --- A. CATBOOST MODELİ ---
print("\n=====================================================")
print("1. CATBOOST MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")
cat_model = CatBoostClassifier(
    auto_class_weights='Balanced',  # Dengesizliği, SMOTE yerine matematiksel cezayla çözer
    iterations=500,
    learning_rate=0.05,
    depth=6,
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


# --- LIGHTGBM MODELİ ---
print("\n=====================================================")
print("2. LIGHTGBM MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")
lgbm_model = LGBMClassifier(
    class_weight='balanced', # Dengesizliği çözen kritik parametre
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(x_train_sc, y_train)
y_pred_lgbm = lgbm_model.predict(x_test_sc)

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
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,             
    min_child_weight=5,      # (Aşırı dallanmayı engeller)
    gamma=1.5,               # (Gereksiz dalları budar)
    eval_metric='logloss'
)
xgb_model.fit(x_train_sc, y_train)
y_pred_xgb = xgb_model.predict(x_test_sc)

print(f"XGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_xgb))

# =====================================================
# 4. GÖRSELLEŞTİRME (EN İYİ MODELİN ÖZELLİK ÖNEM DÜZEYİ)
# =====================================================
# CatBoost genellikle bu tarz verilerde en iyi sonucu verir, onun grafiğini çizdiriyoruz


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

print("\n--- MODEL SÜRECİ BAŞARIYLA TAMAMLANDI! ---")


import xgboost as xgb
from sklearn.metrics import accuracy_score

print("\n=====================================================")
print("--- OVERFITTING (EZBERLEME) KONTROLÜ (max_depth=15) ---")
print("=====================================================")

# Modeli max_depth=15 olacak şekilde kuruyoruz
xgb_ezber_test = xgb.XGBClassifier(
    scale_pos_weight=scale_weight, # Dengesizlik çözümü (Önceki kodda hesaplamıştık)
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,                  # DİKKAT: Derinliği 15'e çıkardık
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

# Modeli Eğitiyoruz
xgb_ezber_test.fit(x_train_sc, y_train)

# 1. Eğitim (Train) Skoru: Modelin daha önce gördüğü veriler
y_train_pred = xgb_ezber_test.predict(x_train_sc)
train_accuracy = accuracy_score(y_train, y_train_pred)

# 2. Test Skoru: Modelin ilk defa karşılaştığı veriler
y_test_pred = xgb_ezber_test.predict(x_test_sc)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Sonuçları Yazdır
print(f"EĞİTİM (Train) Doğruluğu : {train_accuracy:.4f}")
print(f"TEST Doğruluğu           : {test_accuracy:.4f}")
print("-" * 50)
print(f"Aradaki Uçurum (Fark)    : {(train_accuracy - test_accuracy):.4f}")


