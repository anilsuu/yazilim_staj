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


df = pd.read_csv("temizlenmis_su_kalitesi.csv")

# =====================================================
# Sızıntı engellemek için sütunları tekrar filtreleyerk devam ediyorum.
# =====================================================

# modelin öğrenmesi için deney_id ve potability sütunu hariç 9 sütunu seçtim.
kimyasal_sutunlar = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# Bu sütunları x'in içine aldım.
x = df[kimyasal_sutunlar]

# Hedef değişkeni büyük/küçük harf duyarlılığını otomatik aşarak aldım
y_sutun_adi = [col for col in df.columns if col.lower() == 'potability'][0]
y = df[y_sutun_adi]

# Sağlama yaptım 9 sütun mu geliyor diye
print(f"--- GÜVENLİK KONTROLÜ ---")
print(f"X (Özellikler) Sütun Sayısı: {x.shape[1]}")
print("-------------------------\n")


#  Veriyi Bölme (Eğitim ve Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


# Ölçeklendirme (Standardization)
sc = StandardScaler() 


# Train setinde öğren (fit) ve uygula (transform)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# SMOTE ile Sınıf Dengesizliğini Çözmek için (SADECE EĞİTİM SETİNE UYGULANIR)
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

# Sonuçları Değerlendirme
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(x_test)

print("\n--- OPTİMİZASYON SONUÇLARI ---")
print(f"En İyi Parametreler: {grid_search.best_params_}")
print(f"Test Accuracy Skoru: {accuracy_score(y_test, y_pred):.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


# # print(f"X (Özellikler) Sütun Sayısı: {x.shape[1]}") # Burası tam olarak 9 olmalı!

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
# x_gorsel = x_test[:,2]

# # plt.plot yerine plt.scatter kullanın
# plt.scatter(x_gorsel, y_test, color='pink', label='Gerçek Veriler')
# plt.scatter(x_gorsel, tahmin, color='blue', alpha=0.5, label='Model Tahminleri')

# plt.title("Gerçek Değerler ve Tahminler")
# plt.xlabel("Ölçeklendirilmiş Özellik")
# plt.ylabel("Potability (İçilebilirlik)")
# plt.legend()
# plt.show()


