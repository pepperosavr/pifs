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
    'RU000A0B6NQ3', 'RU000A0B7VG5', 'RU000A0ERGA7', 'RU000A0ETL65', 'RU000A0ETL73', 'RU000A0F6PB6', 'RU000A0HGA60', 'RU000A0HNNH0', 'RU000A0JC5V1', 'RU000A0JMGY7', 'RU000A0JNFT7', 'RU000A0JNFU5', 'RU000A0J2585', 'RU000A0JNHS5', 'RU000A0JNG89', 'RU000A0JNNP9', 'RU000A0JNPC2', 'RU000A0JNUW0', 'RU000A0JNUM1', 'RU000A0JNUN9', 'RU000A0JNWP0', 'RU000A0JP0T1', 'RU000A0JP450', 'RU000A0JP484', 'RU000A0JP5S2', 'RU000A0JP625', 'RU000A0JP633', 'RU000A0JP6S0', 'RU000A0JP6T8', 'RU000A0JP6U6', 'RU000A0JP6H3', 'RU000A0JNWT2', 'RU000A0JP8Z1', 'RU000A0JP6E0', 'RU000A0JP6Q4', 'RU000A0JP7Q2', 'RU000A0JPBS3', 'RU000A0JPBT1', 'RU000A0JPC32', 'RU000A0JPCP7', 'RU000A0JPCF8', 'RU000A0JPCX1', 'RU000A0JPDK6', 'RU000A0JPDN0', 'RU000A0JPCY9', 'RU000A0JPDU5', 'RU000A0JPE14', 'RU000A0JPEQ1', 'RU000A0JPEY5', 'RU000A0JPEZ2', 'RU000A0JPF54', 'RU000A0JPHY8', 'RU000A0JPJ27', 'RU000A0JPJ84', 'RU000A0JPJU2', 'RU000A0JPK16', 'RU000A0JPJR8', 'RU000A0JPKA2', 'RU000A0JPKE4', 'RU000A0JPKF1', 'RU000A0JPLG7', 'RU000A0JPC24', 'RU000A0JPLV6', 'RU000A0JPM30', 'RU000A0JPM71', 'RU000A0JPMS0', 'RU000A0JPMV4', 'RU000A0JPMW2', 'RU000A0JPMX0', 'RU000A0JPM22', 'RU000A0JPNW0', 'RU000A0JPNQ2', 'RU000A0JPME0', 'RU000A0JPNC2', 'RU000A0JPNE8', 'RU000A0JPQP7', 'RU000A0JPQ28', 'RU000A0JPQL6', 'RU000A0JPQM4', 'RU000A0JPRU5', 'RU000A0JPQN2', 'RU000A0JPTK2', 'RU000A0JPTL0', 'RU000A0JPTT3', 'RU000A0JPU14', 'RU000A0JPWL4', 'RU000A0JPWM2', 'RU000A0JPWN0', 'RU000A0JPWP5', 'RU000A0JPXA5', 'RU000A0JPY69', 'RU000A0JPYC9', 'RU000A0JPXJ6', 'RU000A0JPZA0', 'RU000A0JPYK2', 'RU000A0JPYW7', 'RU000A0JPZD4', 'RU000A0JPZG7', 'RU000A0JPZH5', 'RU000A0JPZY0', 'RU000A0JQ0M5', 'RU000A0JQ0P8', 'RU000A0JQ0S2', 'RU000A0JQ1D2', 'RU000A0JQ1R2', 'RU000A0JQ2Q2', 'RU000A0JQ4B0', 'RU000A0JQ4Q8', 'RU000A0JQ573', 'RU000A0JQ599', 'RU000A0JQ5P7', 'RU000A0JQ5S1', 'RU000A0JQ4S4', 'RU000A0JQ615', 'RU000A0JQ797', 'RU000A0JQ7K4', 'RU000A0JQA66', 'RU000A0JQC49', 'RU000A0JQC56', 'RU000A0JQAY1', 'RU000A0JQD06', 'RU000A0JQFW4', 'RU000A0JQFV6', 'RU000A0JQJZ9', 'RU000A0JQP85', 'RU000A0JQPB7', 'RU000A0JQPF8', 'RU000A0JQGH3', 'RU000A0JQQ35', 'RU000A0JQRN8', 'RU000A0JQSP1', 'RU000A0JQSW7', 'RU000A0JQTM6', 'RU000A0JQTC7', 'RU000A0JQT73', 'RU000A0JQUD3', 'RU000A0JQUF8', 'RU000A0JQUM4', 'RU000A0JQVQ3', 'RU000A0JQFY0', 'RU000A0JQY19', 'RU000A0JQYZ8', 'RU000A0JQZ00', 'RU000A0JQYE3', 'RU000A0JQZE0', 'RU000A0JR035', 'RU000A0JQZG5', 'RU000A0JQZU6', 'RU000A0JR092', 'RU000A0JR0C5', 'RU000A0JR0A9', 'RU000A0JR1C3', 'RU000A0JR266', 'RU000A0JR316', 'RU000A0JR3K2', 'RU000A0JR3F2', 'RU000A0JR5E0', 'RU000A0JR589', 'RU000A0JR746', 'RU000A0JR8H7', 'RU000A0JRA16', 'RU000A0JRCV1', 'RU000A0JRAJ0', 'RU000A0JRDZ0', 'RU000A0JRDW7', 'RU000A0JREU9', 'RU000A0JRE38', 'RU000A0JRE46', 'RU000A0JQCV3', 'RU000A0JRHK3', 'RU000A0JRHL1', 'RU000A0JRHR8', 'RU000A0JRHC0', 'RU000A0JRK06', 'RU000A0JRKJ9', 'RU000A0JRNZ9', 'RU000A0JRQT5', 'RU000A0JRQU3', 'RU000A0JRT98', 'RU000A0JRTR3', 'RU000A0JRU38', 'RU000A0JRUF6', 'RU000A0JRSY1', 'RU000A0JRST1', 'RU000A0JRVP3', 'RU000A0JRVR9', 'RU000A0JRSC7', 'RU000A0JS2F3', 'RU000A0JS363', 'RU000A0JS3C8', 'RU000A0JS504', 'RU000A0JS520', 'RU000A0JS538', 'RU000A0JS546', 'RU000A0JS595', 'RU000A0JS629', 'RU000A0JS652', 'RU000A0JS868', 'RU000A0JS8S3', 'RU000A0JS8W5', 'RU000A0JS991', 'RU000A0JS9C5', 'RU000A0JS9B7', 'RU000A0JS9A9', 'RU000A0JSAB5', 'RU000A0JSGF3', 'RU000A0JS7S5', 'RU000A0JSMB0', 'RU000A0JSME4', 'RU000A0JSSB7', 'RU000A0JSXE1', 'RU000A0JSWC7', 'RU000A0JSWF0', 'RU000A0JSWG8', 'RU000A0JSY41', 'RU000A0JT7S4', 'RU000A0JT8U8', 'RU000A0JT8W4', 'RU000A0JT916', 'RU000A0JT2H8', 'RU000A0JTCK0', 'RU000A0JT510', 'RU000A0JTEJ8', 'RU000A0JTHQ6', 'RU000A0JTJZ3', 'RU000A0JTJQ2', 'RU000A0JTK53', 'RU000A0JTJR0', 'RU000A0JTPA3', 'RU000A0JTBX5', 'RU000A0JTNQ4', 'RU000A0JTRK8', 'RU000A0JTSQ3', 'RU000A0JTWK8', 'RU000A0JTX17', 'RU000A0JTXM2', 'RU000A0JTY16', 'RU000A0JU1G9', 'RU000A0JU2E2', 'RU000A0JS2G1', 'RU000A0JU7Z6', 'RU000A0JU997', 'RU000A0JU0G1', 'RU000A0JQSM8', 'RU000A0JUE09', 'RU000A0JUCD3', 'RU000A0JUDS9', 'RU000A0JTSS9', 'RU000A0JQPA9', 'RU000A0JUAQ9', 'RU000A0JR3X5', 'RU000A0JUKC8', 'RU000A0JUHC4', 'RU000A0JUKN5', 'RU000A0JUKK1', 'RU000A0JUN16', 'RU000A0JUR20', 'RU000A0JUS03', 'RU000A0JUUC7', 'RU000A0JUTV9', 'RU000A0JUTW7', 'RU000A0JUGG7', 'RU000A0JUR38', 'RU000A0JUR61', 'RU000A0JUKR6', 'RU000A0JV7X0', 'RU000A0JV7U6', 'RU000A0JV7Y8', 'RU000A0JUYK2', 'RU000A0JVEC9', 'RU000A0JVGM3', 'RU000A0JVGN1', 'RU000A0JUZ61', 'RU000A0JVMR0', 'RU000A0JVUU7', 'RU000A0JX0W5', 'RU000A0JQP77', 'RU000A0JR7V0'
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


# In[ ]:




