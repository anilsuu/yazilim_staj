# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:53:06 2026

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
from imblearn.over_sampling import SMOTE

# Temizlenmiş Veriyi Oku
df = pd.read_csv("temizlenmis_su_kalitesi.csv")

# X ve Y Değişkenlerini Oluşturma (Beyaz Liste Yöntemi)
kimyasal_sutunlar = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

x = df[kimyasal_sutunlar]
y_sutun_adi = [col for col in df.columns if col.lower() == 'potability'][0]
y = df[y_sutun_adi]

#  Veriyi Bölme (Eğitim ve Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Ölçeklendirme (Standardization)
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# SMOTE ile Sınıf Dengesizliğini Çözme (SADECE EĞİTİM SETİNE UYGULANIR)
print(f"SMOTE Öncesi Sınıf Dağılımı:\n{y_train.value_counts()}\n")

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print(f"SMOTE Sonrası Sınıf Dağılımı:\n{y_train_smote.value_counts()}\n")

#  XGBoost Modeli Eğitimi (SMOTE uygulanmış verilerle)
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

param_grid = {
    'n_estimators': [100, 200, 300],        
    'max_depth': [3, 5, 7],                 
    'learning_rate': [0.01, 0.05, 0.1],     
    'subsample': [0.8, 1.0],                
    'colsample_bytree': [0.8, 1.0]          
}

grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=5, 
    n_jobs=-1
)

grid_search.fit(x_train_smote, y_train_smote)

# 7. Sonuçları Değerlendirme
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(x_test)

print("\n--- OPTİMİZASYON SONUÇLARI ---")
print(f"En İyi Parametreler: {grid_search.best_params_}")
print(f"Test Accuracy Skoru: {accuracy_score(y_test, y_pred):.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# =====================================================
# X VE Y DEĞİŞKENLERİNİ OLUŞTURMA (BEYAZ LİSTE)
# =====================================================

# 1. Sadece modelin öğrenmesini istediğimiz 9 kimyasal sütunu manuel seçiyoruz.
kimyasal_sutunlar = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]



# Sağlama yapıyoruz (Konsolda mutlaka 9 görmelisin)
print(f"--- GÜVENLİK KONTROLÜ ---")
print(f"X (Özellikler) Sütun Sayısı: {x.shape[1]} (Bu sayı 9 ise sızıntı tamamen bitmiştir!)")
print("-------------------------\n")


# print(f"X (Özellikler) Sütun Sayısı: {x.shape[1]}") # Burası tam olarak 9 olmalı!

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Ölçeklendirme 
sc = StandardScaler() 

# Train setinde öğren (fit) ve uygula (transform)
x_train = sc.fit_transform(x_train)

# Test setinde SADECE uygula (transform)
x_test = sc.transform(x_test) 

# Model Eğitimi
model = LogisticRegression()
model.fit(x_train, y_train)

tahmin=model.predict(x_test)

# x_test'in içinde birden fazla özellik olduğu için görselleştirme adına 
# X ekseninde göstermek üzere sadece 0. indeksteki ilk sütunu (özelliği) seç:
x_gorsel = x_test[:,2]

# plt.plot yerine plt.scatter kullanın
plt.scatter(x_gorsel, y_test, color='pink', label='Gerçek Veriler')
plt.scatter(x_gorsel, tahmin, color='blue', alpha=0.5, label='Model Tahminleri')

plt.title("Gerçek Değerler ve Tahminler")
plt.xlabel("Ölçeklendirilmiş Özellik")
plt.ylabel("Potability (İçilebilirlik)")
plt.legend()
plt.show()

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

print("=====================================================")
print("--- 1. LIGHTGBM MODELİ OPTİMİZASYONU ---")
print("=====================================================")

# LightGBM Pipeline'ı
lgbm_pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    # verbose=-1 ile LightGBM'in gereksiz uyarı mesajlarını susturuyoruz
    ('lgbm', LGBMClassifier(random_state=42, verbose=-1))
])

lgbm_param_grid = {
    'lgbm__n_estimators': [100, 200, 300],
    'lgbm__max_depth': [3, 5, 7],
    'lgbm__learning_rate': [0.01, 0.05, 0.1]
}

grid_lgbm = GridSearchCV(
    estimator=lgbm_pipeline, 
    param_grid=lgbm_param_grid, 
    scoring='accuracy', 
    cv=5, 
    n_jobs=-1
)

print("LightGBM eğitiliyor... Lütfen bekleyin.")
grid_lgbm.fit(x_train, y_train)

best_lgbm = grid_lgbm.best_estimator_
y_pred_lgbm = best_lgbm.predict(x_test)

print(f"\nLightGBM Test Accuracy Skoru: {accuracy_score(y_test, y_pred_lgbm):.4f}")
print("LightGBM Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_lgbm))


print("\n=====================================================")
print("--- 2. CATBOOST MODELİ OPTİMİZASYONU ---")
print("=====================================================")

# CatBoost Pipeline'ı
cat_pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    # verbose=False ile CatBoost'un her adımda ekrana yazı yazmasını engelliyoruz
    ('cat', CatBoostClassifier(random_state=42, verbose=False))
])

cat_param_grid = {
    'cat__iterations': [100, 200, 300], # CatBoost'ta n_estimators yerine iterations kullanılır
    'cat__depth': [3, 5, 7],            # max_depth yerine depth kullanılır
    'cat__learning_rate': [0.01, 0.05, 0.1]
}

grid_cat = GridSearchCV(
    estimator=cat_pipeline, 
    param_grid=cat_param_grid, 
    scoring='accuracy', 
    cv=5, 
    n_jobs=-1
)

print("CatBoost eğitiliyor... Lütfen bekleyin.")
grid_cat.fit(x_train, y_train)

best_cat = grid_cat.best_estimator_
y_pred_cat = best_cat.predict(x_test)

print(f"\nCatBoost Test Accuracy Skoru: {accuracy_score(y_test, y_pred_cat):.4f}")
print("CatBoost Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_cat))
# 1. Pipeline (Boru Hattı) Kurulumu
# SMOTE ve XGBoost'u birbirine bağlıyoruz.
pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
])

# 2. Parametre Izgarası
# Pipeline kullandığımız için parametre isimlerinin başına 'xgb__' eklemeliyiz
param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}

print("Pipeline ile SMOTE ve GridSearchCV çalışıyor (Sızıntısız)...")

# 3. GridSearchCV'yi Başlat
grid_search = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=5, 
    n_jobs=-1,
    verbose=1
)

# DİKKAT: Artık x_train_smote DEĞİL, orijinal x_train veriyoruz!
# Pipeline içeride SMOTE işlemini otomatik ve güvenli yapacak.
grid_search.fit(x_train, y_train)

# 4. Sonuçları Değerlendirme
print("\n--- OPTİMİZASYON SONUÇLARI ---")
print(f"En İyi Parametreler: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print(f"\nTest Accuracy Skoru: {accuracy_score(y_test, y_pred):.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

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
