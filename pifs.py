#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import requests
import pandas as pd
from datetime import datetime

api_url = 'https://dh2.efir-net.ru/v2'
api_login = 'accentam-api-test1'
api_pass = '652Dsw'

def doPostRequest(url, body, token):
    if (token is None):
        headers = {'Content-Type': 'application/json'}
    else:
        headers = {'authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}

    response = requests.post(url, json=body, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def getToken(login, password):
    url = api_url+'/Account/Login'
    body = {
        'login': 'accentam-api-test1',
        'password': '652Dsw'
    }

    token = doPostRequest(url, body, None)
    if token is None:
        return None

    return token['token']

token = getToken(api_login, api_pass)


def fetch_all_trading_results(token, instruments, page_size=100):
    url = f"{api_url}/Moex/History"
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    all_data = []
    page = 0
    date_to = datetime.now().strftime("%Y-%m-%dT00:00:00Z")
    while True:
        body = {
            "engine": "stock",
            "market": "shares",
            "boardid": ["TQIF"],
            "instruments": instruments,
            "dateFrom": "2025-01-01T00:00:00Z",
            "dateTo": "2025-12-24T00:00:00Z",
            "tradingSessions": [],
            "pageNum": page,
            "pageSize": page_size
        }
        response = requests.post(url, json=body, headers=headers)
        if response.status_code != 200:
            print(f"Ошибка на странице {page}:", response.status_code, response.text)
            break
        page_data = response.json()
        if not page_data:
            break
        all_data.extend(page_data)
        page += 1
    return all_data

# делим список secid на части по 100 инструментов 
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


zpif_secids = [  
'RU000A105328',
'RU000A1099U0',
'RU000A1034U7',
'RU000A10A117',
'RU000A100WZ5'
]

token = getToken(api_login, api_pass)
if token:
    all_results = []

    # secid по 100 шт
    for chunk in chunk_list(zpif_secids, 100):
        chunk_data = fetch_all_trading_results(token, chunk)
        all_results.extend(chunk_data)

    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['shortname', 'isin', 'volume', 'close', 'tradedate']]
        print(df.to_string(index=False))
    else:
        print("Данных не найдено.")
else:
    print("Ошибка авторизации")

import streamlit as st
import pandas as pd
import plotly.express as px

df["tradedate"] = pd.to_datetime(df["tradedate"], errors="coerce", utc=True).dt.date

# === ВЫБОР ЗПИФОВ ===
available_funds = df["shortname"].dropna().unique()
selected_funds = st.multiselect("Выберите ЗПИФы", available_funds, default=list(available_funds[:5]))

# === ВЫБОР ДАТЫ ===
min_date = df["tradedate"].min()
max_date = df["tradedate"].max()

selected_date = st.slider(
    "Выберите дату",
    min_value=min_date,
    max_value=max_date,
    value=max_date,
)

# === ФИЛЬТРАЦИЯ ===
filtered_df = df[(df["shortname"].isin(selected_funds)) & (df["tradedate"] == selected_date)]

# === ГРАФИК ===
st.subheader("Объем торгов по выбранным ЗПИФам")
fig = px.bar(filtered_df, x="shortname", y="volume", hover_data=["isin", "close"], color="shortname")
st.plotly_chart(fig, use_container_width=True)
