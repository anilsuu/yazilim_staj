# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:17:06 2026

@author: nilsu
"""
#-*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer


# ─────────────────────────────────────────────────────
# 1 — VERİ YÜKLEME
# ─────────────────────────────────────────────────────

pd.read_csv("su_verisi.csv")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 1. Veriyi başlıksız olarak okuyoruz
df = pd.read_csv('su_verisi.csv', header=None)

# 2. Çiftlenmiş fazladan 9 sütunu atıp, sadece gerçek verinin olduğu ilk 9 sütunu alıyoruz
df = df.iloc[:, :9]

# 3. Belirttiğiniz başlıkları sırasıyla atıyoruz (ilk sütun ph)
headers = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
df.columns = headers


