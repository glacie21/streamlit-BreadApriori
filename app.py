import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

#load dfset
df = pd.read_csv('bread basket.csv')
df['datetime'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

# df["month"] = df["date_time"].dt.month
# df["day"] = df["date_time"].dt.weekday
# df["hour"] = df["date_time"].dt.hour
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.weekday

df["month"].replace([i for i in range(1,13)], ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], inplace=True)
df["day"].replace([i for i in range(7)], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

st.title('Bread Basket Analysis menggunakan apriori') 

def get_data(period_day = '', weekday_weekend = '', month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["period_day"].str.contains(period_day)) & 
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month)) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] > 0 else "Data tidak ditemukan"

def user_input_features():
    item = st.selectbox('Pilih Item', df['Item'].unique())
    period_day = st.selectbox('Pilih Waktu', ['Morning', 'Afternoon', 'Evening'])
    weekday_weekend = st.selectbox('Pilih Hari', ['Weekday', 'Weekend'])
    month = st.select_slider('Pilih Bulan', df['month'].unique())
    day = st.select_slider('Pilih Hari', df['day'].unique())
    return item, period_day, weekday_weekend, month, day

item, period_day, weekday_weekend, month, day = user_input_features()
data = get_data(period_day.lower(), weekday_weekend.lower(), month, day)

def encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

if type(data) != type("No Result"):
    item_count = data.groupby(["Transaction","Item"]).size().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction',columns='Item',values='Count',aggfunc="sum").fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01
    frequent_itemsets = apriori(item_count_pivot, min_support=support, use_colnames=True)
    metric = 'lift'
    min_threshold = 1
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered_data = data.loc[data["antecedents"] == item_antecedents]
    
    if filtered_data.shape[0] > 0:  # Cek apakah data tersedia
        return list(filtered_data.iloc[0, :])
    else:
        return None  # Kembalikan None jika tidak ada data yang sesuai

if type(data) != type("No Result"):
    result = return_item_df(item)
    
    if result is not None:
        st.markdown("Hasil Rekomendasi : ")
        st.success(f"Jika konsumen membeli **{item}**, maka membeli **{result[1]}** secara bersamaan")
    else:
        st.error("Tidak ada rekomendasi yang ditemukan untuk item ini.")
else:
    st.error("Data tidak ditemukan")
