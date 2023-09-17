# -- coding: utf-8 --
"""
 @time : 2023/8/7
 @file : kucun_generator.py
 @author : zhenghao
 @software: PyCharm

生成库存同比，水位，环比值的对冲因子

"""


import glob
import os
import re
import numpy as np
import datetime
import pandas as pd


nowtime = datetime.datetime.now().strftime("%Y-%m-%d")
factor_list = os.listdir("Y:\CTA\kucun")
tradingday = pd.read_csv("all_tradeday.csv",index_col=0,parse_dates=[0])
tradingday = tradingday.loc["2014-01-01":nowtime]


def shift_holiday(df):
    col_1 = df.columns[0]  # 交易日
    col_2 = df.columns[1]  # 需要对齐的数据

    def is_hoilday(x, name):
        if np.isnan(x[col_1]) and not np.isnan(x[name]):
            return 1
        else:
            return 0

    df_0 = df.copy()  # 保留交易日信息
    df = df.reset_index()
    index = df.columns[0]
    df['is_holiday'] = df.apply(lambda x: is_hoilday(x, col_2), axis=1)
    df[index] = df.apply(lambda x: np.nan if x['is_holiday'] == 1 else x[index], axis=1)
    df[index] = df[index].fillna(method='ffill')
    df.drop_duplicates(subset=[index], keep='last', inplace=True)
    df = df.set_index(index)
    res = pd.concat([df_0[col_1].to_frame(), df[col_2]], axis=1)
    res[col_2].fillna(inplace=True, method='ffill')
    res.dropna(inplace=True, subset=[col_1])
    return res


def generate_signal(df,sym,number,func='tongbi'):
    df = pd.concat([tradingday,df],axis=1)
    df = shift_holiday(df)
    df = df.iloc[:,1].to_frame()
    df.columns = [sym]

    if func in ['tongbi']:
        signal = df[sym].pct_change(244).to_frame()
        signal.columns = [sym]
    elif func in ['huanbi_week']:
        signal = df[sym].pct_change(5).to_frame()
        signal.columns = [sym]
    elif func in ['huanbi_month']:
        signal = df[sym].pct_change(20).to_frame()
        signal.columns = [sym]
    elif func in ['shuiwei']:
        signal = df[sym].rolling(244).rank(pct=True).to_frame()
        signal.columns = [sym]
    elif func=='huanbi_diff_week':
        df['环比'] = df[sym].pct_change(5)
        df['环比zscore'] = df['环比'].rolling(5).apply(lambda x:(x[-1]-x.mean())/x.std())
        signal = df[['环比zscore']]
        signal.columns = [sym]
    elif func=='huanbi_diff_month':
        df['环比'] = df[sym].pct_change(5)
        df['环比zscore'] = df['环比'].rolling(20).apply(lambda x:(x[-1]-x.mean())/x.std())
        signal = df[['环比zscore']]
        signal.columns = [sym]

    if sym in ['P', 'M', 'A', 'RM', 'OI', 'C', 'Y']:
        signal = signal.shift(1)
    elif sym in ['SC']:
        signal = signal.shift(3)
    elif sym in ['NI'] and number ==2:
        signal =signal.shift(1)
    elif sym in ['PP','L'] and number==1:
        signal = signal.shift(1)
    elif sym in ['NR']:
        signal = signal.shift(2)

    return signal


total = {
    "库存同比缓存":{},
    "库存环比缓存":{},
    "库存水位缓存":{},
}

for factor in factor_list:
    df_list = glob.glob(f"kucun/{factor}/*.csv")
    for df in df_list:
        name = os.path.split(df)[1]
        sym = re.sub(r"\d", "", name)[:-4]
        if sym in ['ZC', 'J', 'FU', 'HCBX', 'RBBX', 'CF']:
            continue
        else:
            data = pd.read_csv(df,index_col=0,parse_dates=[0])
            data = data[data.columns[1]]
            try:
                total[factor][sym].append(data)
            except:
                total[factor][sym] = []
                total[factor][sym].append(data)


tongbi_signal = []
huanbi_week_signal = []
huanbi_month_signal = []
shuiwei_signal = []
huanbi_diff_week_signal = []
huanbi_diff_month_signal = []

for factor_name,data_list in total.items():
    print(factor_name)
    if factor_name=='库存同比缓存':
        for sym, temp in data_list.items():
            temp = pd.concat(temp, axis=1)
            tongbi = pd.DataFrame()
            for k in range(len(list(temp.columns))):
                temp_signal = generate_signal(temp[[temp.columns[k]]], sym=sym, number=k,func='tongbi')
                tongbi = pd.concat([tongbi,temp_signal],axis=1)
            sym_signal = tongbi.apply(lambda x: x.dropna().mean(), axis=1).ffill().to_frame()
            sym_signal.columns = [sym]
            tongbi_signal.append(sym_signal)

    elif factor_name=='库存水位缓存':
        for sym, temp in data_list.items():
            temp = pd.concat(temp, axis=1)
            shuiwei = pd.DataFrame()
            for k in range(len(list(temp.columns))):
                temp_signal = generate_signal(temp[[temp.columns[k]]], sym=sym, number=k,func='shuiwei')
                shuiwei = pd.concat([shuiwei,temp_signal],axis=1)
            sym_signal = shuiwei.apply(lambda x: x.dropna().mean(), axis=1).ffill().to_frame()
            sym_signal.columns = [sym]
            shuiwei_signal.append(sym_signal)

    elif factor_name=='库存环比缓存':
        for sym, temp in data_list.items():
            temp = pd.concat(temp, axis=1)
            huanbi_month = pd.DataFrame()
            huanbi_week = pd.DataFrame()
            huanbi_diff_week= pd.DataFrame()
            huanbi_diff_month=pd.DataFrame()

            for k in range(len(list(temp.columns))):
                temp_signal = generate_signal(temp[[temp.columns[k]]], sym=sym, number=k,func='huanbi_week')
                huanbi_month = pd.concat([huanbi_month,temp_signal],axis=1)

                temp_signal = generate_signal(temp[[temp.columns[k]]], sym=sym, number=k, func='huanbi_week')
                huanbi_week= pd.concat([huanbi_week, temp_signal], axis=1)

                temp_signal = generate_signal(temp[[temp.columns[k]]], sym=sym, number=k,func='huanbi_diff_week')
                huanbi_diff_week = pd.concat([huanbi_diff_week,temp_signal],axis=1)

                temp_signal = generate_signal(temp[[temp.columns[k]]], sym=sym, number=k, func='huanbi_diff_month')
                huanbi_diff_month = pd.concat([huanbi_diff_month,temp_signal],axis=1)

            sym_signal = huanbi_week.apply(lambda x: x.dropna().mean(), axis=1).ffill().to_frame()
            sym_signal.columns = [sym]
            huanbi_week_signal.append(sym_signal)

            sym_signal = huanbi_month.apply(lambda x: x.dropna().mean(), axis=1).ffill().to_frame()
            sym_signal.columns = [sym]
            huanbi_month_signal.append(sym_signal)

            sym_signal = huanbi_diff_week.apply(lambda x: x.dropna().mean(), axis=1).ffill().to_frame()
            sym_signal.columns = [sym]
            huanbi_diff_week_signal.append(sym_signal)

            sym_signal = huanbi_diff_month.apply(lambda x: x.dropna().mean(), axis=1).ffill().to_frame()
            sym_signal.columns = [sym]
            huanbi_diff_month_signal.append(sym_signal)




pd.concat(tongbi_signal,axis=1).to_csv("C:/Project/截面因子V2/kucun_factor/kucun_tongbi.csv")
pd.concat(shuiwei_signal, axis=1).to_csv("C:/Project/截面因子v2/kucun_factor/kucun_shuiwei.csv")
pd.concat(huanbi_month_signal, axis=1).to_csv("C:/Project/截面因子V2/kucun_factor/kucun_huanbi_month.csv")
pd.concat(huanbi_week_signal, axis=1).to_csv("C:/Project/截面因子V2/kucun_factor/kucun_huanbi_week.csv")
pd.concat(huanbi_diff_week_signal, axis=1).to_csv("C:/Project/截面因子V2/kucun_factor/kucun_huanbi_diff_week.csv")
pd.concat(huanbi_diff_month_signal, axis=1).to_csv("C:/Project/截面因子V2/kucun_factor/kucun_huanbi_diff_month.csv")


# 给出月差因子

from CTA_factor import Jicha_Mom
heise = ['I', 'J', 'RB', 'HC']
youse = ['CU', 'AL', 'ZN', 'NI']
huagong = ['MA', 'PP', 'L', 'TA', 'BU', 'EG', 'RU', 'SP', 'UR', 'SA', 'FG', 'V', 'SC']
agriculture = ['SR', 'CF', 'C', 'CS', 'M', 'OI', 'A', 'RM', 'Y', 'P']
trading_symbols = heise + youse + huagong + agriculture

total_signal = pd.DataFrame()
for k in trading_symbols:
    temp = Jicha_Mom(k, mode=2, N=5)
    total_signal = pd.concat([total_signal, temp], axis=1)
total_signal.to_excel(r"C:\TBWork\仓单数据\yuecha_ratio_raw_mom.xlsx")
























