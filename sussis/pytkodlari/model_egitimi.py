# -*- coding: utf-8 -*-
"""

@author: nilsu
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Eksik modüller eklendi
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

df_son = pd.read_csv("temizlenmis_su_kalitesi.csv")

# modelin öğrenmesi için deney_id ve potability sütunu hariç 9 sütunu seçtim.
tum_kimyasal_sutunlar = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
    'ph_ideal', 'Turbidity_safe', 'Sulfate_safe', 'Chemistry_reactivity', 
    'solids_to_cond_ratio', 'hardness_sulfate_ratio', 'ph_dev_from_neutral', 
    'pollution_index', 'safety_violations'
]

# Grafikte en altta kalan özellikleri buraya yazdım.
silinecek_sutunlar = ['ph_ideal', 'safety_violations', 'Chemistry_reactivity', 'pollution_index'] 
secili_sutunlar = [col for col in tum_kimyasal_sutunlar if col not in silinecek_sutunlar]

# X ve y değişkenlerini tanımlama
X_sade = df_son[secili_sutunlar]
y_sutun_adi = [col for col in df_son.columns if col.lower() == 'potability'][0]
y_sade = df_son[y_sutun_adi]

print("\n--- GÜVENLİK KONTROLÜ ---")
print(f"Kullanılan Özellik Sayısı: {X_sade.shape[1]}")
print("-------------------------\n")


# Veriyi Bölme (Eğitim ve Test)
x_train, x_test, y_train, y_test = train_test_split(
    X_sade, y_sade, test_size=0.30, random_state=42, stratify=y_sade
)

# Ölçeklendirme (MaxAbsScaler olarak düzeltildi)
sc = MaxAbsScaler() 

# Train setinde öğren (fit) ve uygula (transform)
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

# Sınıf dengesizliğini algoritmaların kendi içine ağırlık verdim.
scale_weight = (y_train == 0).sum() / (y_train == 1).sum() 


print("\n=====================================================")
print("--- 1. VERİ SIZINTISI (TARGET LEAKAGE) KONTROLÜ ---")
print("=====================================================")

# x_train Pandas DataFrame formunda olduğu için doğrudan kopyalanabilir
aktif_x_train = x_train
df_check = aktif_x_train.copy()
print("Eğitim setindeki sütunlar:", df_check.columns.tolist())

# Hedef değişkeni güvenle tabloya ekle
df_check['HEDEF_POTABILITY'] = np.array(y_train).flatten()

# Korelasyon hesabı
korelasyonlar = df_check.corr()['HEDEF_POTABILITY'].drop('HEDEF_POTABILITY').sort_values(ascending=False)
print("\nÖzelliklerin Hedef Değişkenle (Potability) Korelasyonu:")
print(korelasyonlar)


print("\n=====================================================")
print("GRID SEARCH (SINIFLANDIRMA İÇİN GÜNCELLENDİ)")
print("=====================================================")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'ccp_alpha': [0.0, 0.01]
}

print("\nGridSearch çalışıyor...")
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

print("\n=====================================================")
print("1. CATBOOST MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")

cat_model = CatBoostClassifier(
    auto_class_weights='Balanced',  
    iterations=500,
    learning_rate=0.02,
    depth=4,
    verbose=False,   
    random_state=42
)

print("CatBoost eğitiliyor (SMOTE YOK)...")
cat_model.fit(x_train, y_train) 
y_pred_cat = cat_model.predict(x_test)

print(f"CatBoost Test Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_cat))

feature_importances = pd.DataFrame({
    'Feature': secili_sutunlar,
    'Importance': cat_model.get_feature_importance()
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances, hue='Feature', palette='viridis', legend=False)
plt.title('CatBoost - Özellik Önem Düzeyleri (Feature Importance)')
plt.xlabel('Önem Skoru')
plt.ylabel('Özellikler')
plt.tight_layout()
plt.show()


print("\n=====================================================")
print("2. LIGHTGBM MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")

lgbm_model = LGBMClassifier(
    class_weight='balanced', 
    n_estimators=100,
    learning_rate=0.02,
    max_depth=4,
    random_state=42,
    verbose=-1
)

print("LightGBM eğitiliyor (SMOTE YOK)...")
lgbm_model.fit(x_train_sc, y_train) 
y_pred_lgbm = lgbm_model.predict(x_test_sc)

feature_importances_lgbm = pd.DataFrame({
    'Feature': secili_sutunlar,
    'Importance': lgbm_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances_lgbm, hue='Feature', palette='viridis', legend=False)
plt.title('LightGBM - Özellik Önem Düzeyleri (Feature Importance)')
plt.xlabel('Önem Skoru')
plt.ylabel('Özellikler')
plt.tight_layout()
plt.show()

print(f"LightGBM Test Accuracy: {accuracy_score(y_test, y_pred_lgbm):.4f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_lgbm))


print("\n=====================================================")
print("3. XGBOOST MODELİ (Ağırlık Dengelemeli)")
print("=====================================================")
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_weight, 
    n_estimators=100,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,             
    min_child_weight=20,      
    gamma=1.5,               
    eval_metric='logloss'
)
xgb_model.fit(x_train_sc, y_train)
y_pred_xgb = xgb_model.predict(x_test_sc)

print(f"XGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_xgb))


print("\n--- MODEL SÜRECİ BAŞARIYLA TAMAMLANDI! ---")


print("\n=====================================================")
print("--- DÜZENLİLEŞTİRİLMİŞ OVERFITTING KONTROLÜ ---")
print("=====================================================")

xgb_saglam_kontrol = xgb.XGBClassifier(
    scale_pos_weight=scale_weight, 
    n_estimators=100,            
    learning_rate=0.02,          
    max_depth=4,                 
    min_child_weight=20,         
    gamma=2.0,                   
    subsample=0.7,               
    colsample_bytree=0.7,        
    reg_alpha=1.0,               
    reg_lambda=1.0,              
    random_state=42,
    eval_metric='logloss'
)

xgb_saglam_kontrol.fit(x_train_sc, y_train)

y_train_pred = xgb_saglam_kontrol.predict(x_train_sc)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = xgb_saglam_kontrol.predict(x_test_sc)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"EĞİTİM (Train) Doğruluğu : {train_accuracy:.4f}")
print(f"TEST Doğruluğu           : {test_accuracy:.4f}")
print("-" * 50)
print(f"Yeni Uçurum (Fark)       : {(train_accuracy - test_accuracy):.4f}")


print("\n=====================================================")
print("LEARNING CURVE (SINIFLANDIRMA SKORU İLE)")
print("=====================================================")
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



