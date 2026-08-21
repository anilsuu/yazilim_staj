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

def hesaplamainfo(column):
    print(f"Ortalama:{column}",df[column].mean())
    print(f"Medyan:{column}",df[column].median())
    print(f"Standart Sapma:{column}",df[column].std())
    print(f"Eksik değerler:{column}",df[column].isna().sum())
    
    
#Eksik Verilerin analiz edilmelerine göre doldurulmaları.
# Tüm sütunları baz alarak tamamen aynı olan satırları tespit edip,siler.
print("Duplicate gruplarındaki toplam satır:",
      df.duplicated(keep=False).sum())
df = df.drop_duplicates()

# print(df.describe())
# std=np.std()
# print(std)


print("ph eksik oranı:", df["ph"].isna().mean() * 100)
print("Sulfate eksik oranı:", df["Sulfate"].isna().mean() * 100)
print("Trihalomethanes eksik oranı:", df["Trihalomethanes"].isna().mean() * 100)

# Tüm sayısal değişkenlerin standart sapmasını Series olarak döndürür
print("-----------------------------------------------------")

# print("Tüm sütunların standart sapması:")
# std_values = df.std(numeric_only=True)
# print(std_values)

print("Sütunların standart sapması:")
std_values = df.drop(columns=["deney_id"]).std(numeric_only=True)
print(std_values)

print("-----------------------------------------------------")

#Histogram Grafiği
plt.hist(df["ph"].dropna(), bins=20)
plt.xlabel("pH")
plt.ylabel("Frekans")
plt.title("pH Değerlerinin Dağılımı")
plt.show()


print(hesaplamainfo("ph"))
# print("pH Ortalama:", df["ph"].mean())
# print("pH Medyan:", df["ph"].median())
# print("pH Standart Sapma:", df["ph"].std())
# #Eksik Verilerin analiz edilmelerine göre doldurulmaları.
# print("Eksik değerler:", df["ph"].isna().sum())

# ph sütunundaki null değerleri 'ph'sütununu medyanı ile doldurur.
df["ph"] = df["ph"].fillna(df["ph"].median())

# Eksik veri kalıp kalmadığının kontrolü
print("Doldurma sonrası pH Ortalama:", df["ph"].mean())
print("Doldurma sonrası pH Medyan:", df["ph"].median())
print("Doldurma sonrası pH Standart Sapma:", df["ph"].std())
print("Doldurma sonrası pH Eksik Değer:", df["ph"].isna().sum())


#Histogram Grafiği
plt.hist(df["ph"], bins=20)
plt.xlabel("pH")
plt.ylabel("Frekans")
plt.title("pH Değerlerinin Dağılımı")
plt.show()

print("-----------------------------------------------------")


#Null değerlerin sayısı ve medyan değerinin hesaplanması
print("Eksik değerler:" , df["Hardness"].isna().sum())
print("Medyan değeri:", df["Hardness"].median())

#Histogram Grafiği
plt.hist(df["Hardness"],bins=20)
plt.xlabel("Hardness")
plt.ylabel("frekans")
plt.title("Hardness değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")


print("Eksik değerler:", df["Solids"].isna().sum())
print("Medyan değeri:", df["Solids"].median())

#Histogram Grafiği
plt.hist(df["Solids"], bins=20)
plt.xlabel("Solids")
plt.ylabel("frekans")
plt.title("Solids Değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")


print("Eksik değerler:", df["Chloramines"].isna().sum())
print("Medyan değeri:", df["Chloramines"].median())

#Histogram Grafiği
plt.hist(df["Chloramines"], bins=20)
plt.xlabel("Chloramines")
plt.ylabel("frekans")
plt.title("Chloramines Değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")



#Histogram Grafiği
plt.hist(df["Sulfate"].dropna(),bins=20)
plt.xlabel("Sulfate")
plt.ylabel("frekans")
plt.title("Sulfate değerlerinin Dağılımı")
plt.show()

hesaplamainfo("Sulfate")

df["Sulfate"]=df["Sulfate"].fillna(df["Sulfate"].median())

# Eksik veri kalıp kalmadığının kontrolü
print("Doldurma sonrası Sulfate Ortalama:", df["Sulfate"].mean())
print("Doldurma sonrası Sulfate Medyan:", df["Sulfate"].median())
print("Doldurma sonrası Sulfate Standart Sapma:", df["Sulfate"].std())
print("Doldurma sonrası Sulfate Eksik Değer:", df["Sulfate"].isna().sum())

 
#Histogram Grafiği
plt.hist(df["Sulfate"],bins=20)
plt.xlabel("Sulfate")
plt.ylabel("frekans")
plt.title("Sulfate değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

print("Eksik değerler:",df["Conductivity"].isna().sum())
print("Medyan Değeri:",df["Conductivity"].median())

#Histogram Grafiği
plt.hist(df["Conductivity"],bins=20)
plt.xlabel("Conductivity")
plt.ylabel("frekans")
plt.title("Conductivity değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

print("Eksik değerler:",df["Organic_carbon"].isna().sum())
print("Medyan Değeri:",df["Organic_carbon"].median())

#Histogram Grafiği
plt.hist(df["Organic_carbon"],bins=20)
plt.xlabel("Organic_carbon")
plt.ylabel("frekans")
plt.title("Organic_carbon değerlerinin Dağılımı")
plt.show()


print("-----------------------------------------------------")

#Histogram Grafiği
plt.hist(df["Trihalomethanes"].dropna(), bins=20)
plt.xlabel("Trihalomethanes")
plt.ylabel("Frekans")
plt.title("Trihalomethanes Değerlerinin Dağılımı")
plt.show()

hesaplamainfo("Trihalomethanes")

df["Trihalomethanes"]=df["Trihalomethanes"].fillna(df["Trihalomethanes"].median())
print("Medyan değeri:", df["Trihalomethanes"].median())


print("Doldurma sonrası Trihalomethanes Ortalama:", df["Trihalomethanes"].mean())
print("Doldurma sonrası Trihalomethanes Medyan:", df["Trihalomethanes"].median())
print("Doldurma sonrası Trihalomethanes Standart Sapma:", df["Trihalomethanes"].std())
print("Doldurma sonrası Eksik değerler:", df["Trihalomethanes"].isna().sum())

#Histogram Grafiği
plt.hist(df["Trihalomethanes"], bins=20)
plt.xlabel("Trihalomethanes")
plt.ylabel("frekans")
plt.title("Trihalomethanes Değerlerinin Dağılımı")
plt.show()

print("-----------------------------------------------------")


print("Eksik değerler:", df["Turbidity"].isna().sum())
print("Medyan değeri:", df["Turbidity"].median())

#Histogram Grafiği
plt.hist(df["Turbidity"], bins=20)
plt.xlabel("Turbidity")
plt.ylabel("frekans")
plt.title("Turbidity Değerlerinin Dağılımı")
plt.show()



print("-----------------------------------------------------")

#
#EKSİKLERİ DOLDURUP AYKIRI ANALİZİ YAP YARIN 
#BU SATIRLARI SİLLLLLL*****
#ENCODED PAZARTESİ VERİ SETİNİN YARISINI , SALI VERİ SETİNİN ÖBÜR YARISINI ENCODE


