# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt

# =====================================================
# Veriyi çekme
# =====================================================
df = pd.read_csv("buyuk_veri_su.csv")

tum_kimyasal_sutunlar = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# =====================================================
# Dinamik ve dengeli medyan hesaplamalı puanlama
# =====================================================

tummedyanlar = df[tum_kimyasal_sutunlar].median()

print(tummedyanlar)


    # pH: İdeal aralık 6.5 - 8.5
    
def dinamik_su_puani(row):
    puan = 0
    if 6.5 <= row['ph'] <= 8.5:
        puan += 1
    # Diğer parametreler verinin medyanından düşük/yakınsa temiz puanı alır
    if row['Hardness'] <= tummedyanlar['Hardness']:
        puan += 1
    if row['Solids'] <= tummedyanlar['Solids']:
        puan += 1
    if row['Chloramines'] <= tummedyanlar['Chloramines']:
        puan += 1
    if row['Sulfate'] <= tummedyanlar['Sulfate']:
        puan += 1
    if row['Conductivity'] <= tummedyanlar['Conductivity']:
        puan += 1
    if row['Organic_carbon'] <= tummedyanlar['Organic_carbon']:
        puan += 1
    if row['Trihalomethanes'] <= tummedyanlar['Trihalomethanes']:
        puan += 1
    if row['Turbidity'] <= tummedyanlar['Turbidity']:
        puan += 1
        
    # 9 kriterden en az 5 tanesini sağlayanlar 'İçilebilir (1)', diğerleri 'İçilemez (0)'
    return 1 if puan >= 5 else 0


df['Potability_Kural'] = df.apply(dinamik_su_puani, axis=1)

print("--- Yeni Dengeli Etiket Dağılımı ---")
print(df['Potability_Kural'].value_counts())
print("-" * 45)

print("-----------------------------------------------------")

# hedefleri (y) modele verdiğim için bu denetimli (supervised) bir öğrenmedir.
# Öğrenme için model eğitim seti
# =====================================================

X = df[tum_kimyasal_sutunlar]
y = df['Potability_Kural']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Model Eğitimi

model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10,
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train_sc, y_train)

print("-----------------------------------------------------")


# Modeli değerlendirme
# =====================================================


y_pred = model.predict(X_test_sc)

print(f"\nModel Doğruluk Oranı (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nKarmaşıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

print("-----------------------------------------------------")

# Web kısmı için dosya kaydetme
# =====================================================

joblib.dump(scaler, "scaler_kural.pkl")
joblib.dump(model, "rf_kural_su_model.pkl")
print("\nModel ve scaler başarıyla kaydedildi!")


# Modeli ve Scaler'ı Kaydet
joblib.dump(model, "rf_kural_su_model.joblib")
joblib.dump(scaler, "scaler_kural.joblib")
print("model hazır ve kaydedildi!")

# =====================================================
# Öğrenme eğrisini görselleştirme 

train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X_train_sc,
    y=y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring="accuracy",
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, "o-", color="blue", label="Eğitim Skoru (Train)")
plt.plot(
    train_sizes, test_mean, "s-", color="green", label="Doğrulama Skoru (Cross-Val)"
)

# Standart sapma alanlarını renklendirerek güven aralığı ekleme

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
    color="blue",
)
plt.fill_between(
    train_sizes,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.15,
    color="green",
)

plt.title("Random Forest Öğrenme Eğrisi (Learning Curve)", fontsize=13)
plt.xlabel("Eğitim Verisi Boyutu (Örnek Sayısı)", fontsize=11)
plt.ylabel("Doğruluk (Accuracy)", fontsize=11)
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test_sc)[:, 1]
print(f"ROC-AUC Skoru: {roc_auc_score(y_test, y_prob):.4f}")