# -*- coding: utf-8 -*-
"""
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

#Server ve database bilgisi
server = 'LAPTOP-MNJN06EU\\NILSS'
database = 'water_usability'

#Bağlantı metni -> güvenilirliği için TrustedConnections=yes olmalı 
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'


# Bağlantı metnini URL formatına çevirip motoru (engine) oluştur
params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Sql'den tablo Python'a 
sql_query = "SELECT * FROM Drinking_water1"
df = pd.read_sql(sql_query, engine)

print(df.head()) # İlk 5 satırı yazdır

print("Veri boyutu:", df.shape)


# DUPLICATE(TEKRAR EDEN VERİ) KONTROLÜ
print("Toplam duplicate kayıt:", df.duplicated().sum())

# Tüm sütunları baz alarak tamamen aynı olan satırları tespit edip,siler.
print("Duplicate gruplarındaki toplam satır:",
      df.duplicated(keep=False).sum())
df = df.drop_duplicates()



# print(df.describe())

# std=np.std()
# print(std)

# Tüm sayısal değişkenlerin standart sapmasını Series olarak döndürür
print("-----------------------------")


# print("Tüm sütunların standart sapması:")
# std_values = df.std(numeric_only=True)
# print(std_values)

print("Sütunların standart sapması:")
std_values = df.drop(columns=["deney_id"]).std(numeric_only=True)
print(std_values)

print("-----------------------------")


#Eksik Verilerin analiz edilmelerine göre doldurulmaları.
print("Eksik değerler:", df[["ph"]].isna().sum())

# ph sütunundaki null değerleri 'ph'sütununu medyanı ile doldurur.
df["ph"] = df["ph"].fillna(df["ph"].median())

print("ph Medyan değeri:", df["ph"].median())

# Eksik veri kalıp kalmadığının kontrolü
print("Doldurma sonrası 'ph' eksik değer sayısı:", df["ph"].isna().sum())

plt.hist(df["ph"], bins=20)

plt.xlabel("pH")
plt.ylabel("Frekans")
plt.title("pH Değerlerinin Dağılımı")

plt.show()
print("-----------------------------")


#Hardness Sütunu için eksik verilerin sayısı
print("Eksik değerler:" , df[["Hardness"]].isna().sum())

# Eksik veri olmadığı için sadece medyan değerinin gösterilmesi
print("Hardness Medyan değeri:", df["Hardness"].median())

plt.hist(df["Hardness"],bins=20)
plt.xlabel("Hardness")
plt.ylabel("frekans")
plt.title("Hardness değerlerinin Dağılımı")

plt.show()

print("-----------------------------")


# Solids sütunundaki eksik değerlerin sayısı
print("Eksik değerler:", df[["Solids"]].isna().sum())

# Solids sütununun medyan değerinin gösterilmesi
print("Solids Medyan değeri:", df["Solids"].median())

plt.hist(df["Solids"], bins=20)

plt.xlabel("Solids")
plt.ylabel("frekans")
plt.title("Solids Değerlerinin Dağılımı")

plt.show()

print("-----------------------------")


# Chloramines sütunundaki eksik değerlerin sayısı
print("Eksik değerler:", df[["Chloramines"]].isna().sum())

# Chloramines sütununun medyan değerinin gösterilmesi
print("Chloramines Medyan değeri:", df["Chloramines"].median())

plt.hist(df["Chloramines"], bins=20)

plt.xlabel("Chloramines")
plt.ylabel("frekans")
plt.title("Chloramines Değerlerinin Dağılımı")

plt.show()

print("-----------------------------")


,

