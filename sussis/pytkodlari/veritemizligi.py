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

#Histogram Grafiği
plt.hist(df["ph"], bins=20)
plt.xlabel("pH")
plt.ylabel("Frekans")
plt.title("pH Değerlerinin Dağılımı")
plt.show()

#Eksik Verilerin analiz edilmelerine göre doldurulmaları.
print("Eksik değerler:", df["ph"].isna().sum())

print("Medyan değeri:", df["ph"].median())

# ph sütunundaki null değerleri 'ph'sütununu medyanı ile doldurur.
df["ph"] = df["ph"].fillna(df["ph"].median())

# Eksik veri kalıp kalmadığının kontrolü
print("Doldurma sonrası 'ph' eksik değerler:", df["ph"].isna().sum())

#Histogram Grafiği
plt.hist(df["ph"], bins=20)
plt.xlabel("pH")
plt.ylabel("Frekans")
plt.title("pH Değerlerinin Dağılımı")
plt.show()
print("-----------------------------")

#Null değerlerin sayısı ve medyan değerinin hesaplanması
print("Eksik değerler:" , df["Hardness"].isna().sum())
print("Medyan değeri:", df["Hardness"].median())

#Histogram Grafiği
plt.hist(df["Hardness"],bins=20)
plt.xlabel("Hardness")
plt.ylabel("frekans")
plt.title("Hardness değerlerinin Dağılımı")
plt.show()

print("-----------------------------")

print("Eksik değerler:", df["Solids"].isna().sum())
print("Medyan değeri:", df["Solids"].median())

#Histogram Grafiği
plt.hist(df["Solids"], bins=20)
plt.xlabel("Solids")
plt.ylabel("frekans")
plt.title("Solids Değerlerinin Dağılımı")
plt.show()

print("-----------------------------")

print("Eksik değerler:", df["Chloramines"].isna().sum())
print("Medyan değeri:", df["Chloramines"].median())

#Histogram Grafiği
plt.hist(df["Chloramines"], bins=20)
plt.xlabel("Chloramines")
plt.ylabel("frekans")
plt.title("Chloramines Değerlerinin Dağılımı")
plt.show()

print("-----------------------------")


#Histogram Grafiği
plt.hist(df["Sulfate"],bins=20)
plt.xlabel("Sulfate")
plt.ylabel("frekans")
plt.title("Sulfate değerlerinin Dağılımı")
plt.show()

print("Eksik değerler:",df["Sulfate"].isna().sum())
print("Medyan Değeri:",df["Sulfate"].median())
df["Sulfate"]=df["Sulfate"].fillna(df["Sulfate"].median())
print("Doldurma sonrası Sulfate eksik değerler:",df["Sulfate"].isna().sum())

#Histogram Grafiği
plt.hist(df["Sulfate"],bins=20)
plt.xlabel("Sulfate")
plt.ylabel("frekans")
plt.title("Sulfate değerlerinin Dağılımı")
plt.show()

print("-----------------------------")

print("Eksik değerler:",df["Conductivity"].isna().sum())
print("Medyan Değeri:",df["Conductivity"].median())

#Histogram Grafiği
plt.hist(df["Conductivity"],bins=20)
plt.xlabel("Conductivity")
plt.ylabel("frekans")
plt.title("Conductivity değerlerinin Dağılımı")
plt.show()

print("-----------------------------")

print("Eksik değerler:",df["Organic_carbon"].isna().sum())
print("Medyan Değeri:",df["Organic_carbon"].median())

#Histogram Grafiği




#EKSİKLERİ DOLDURUP AYKIRI ANALİZİ YAP YARIN 
#BU SATIRLARI SİLLLLLL*****
#ENCODED PAZARTESİ VERİ SETİNİN YARISINI , SALI VERİ SETİNİN ÖBÜR YARISINI ENCODE


