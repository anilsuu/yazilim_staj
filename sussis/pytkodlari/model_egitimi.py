# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:53:06 2026

@author: nilsu
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

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