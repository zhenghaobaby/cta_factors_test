# -- coding: utf-8 --
"""
 @time : 2023/9/4
 @file : Minute_factor_calculator.py
 @author : zhenghao
 @software: PyCharm
"""
import pandas as pd
import json
from data_loader import DataFetcher
from auxiliary import multask
from indicators import *
import multiprocessing
import re
import os


def handle_one_bar(df, N):
    groups, bins = pd.qcut(df.rank(method='first', ascending=False), q=N, labels=False, retbins=True)
    return groups


def processing_ind(df,method=None):
    if method=='D':
        return df
    else:
        func,N = method.split("-")
        if func == 'MA':
            df = df.rolling(int(N)).mean().dropna(axis=0,how='all')
        elif func == "STD":
            df = df.rolling(int(N)).std().dropna(axis=0,how='all')
        elif func == 'CV':
            df = df.rolling(int(N)).std().dropna(axis=0, how='all')/df.rolling(int(N)).mean().dropna(axis=0,how='all')
        elif func == 'SUM':
            df = df.rolling(int(N)).sum().dropna(axis=0,how='all')
        elif func == 'PROD':
            def cum_prod(x):
                return x.prod()
            df = df.rolling(int(N)).apply(lambda x:cum_prod(x)).dropna(axis=0,how='all')
        elif func == 'NSTD':
            df = -df.rolling(int(N)).std().dropna(axis=0, how='all')
        else:
            print("未知方法！！！请扩充")
        return  df

if __name__ == '__main__':
    tradingday = pd.read_csv("all_tradeday.csv", index_col=0, parse_dates=[0])
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    datelist = list(tradingday.loc[start_date:end_date].index)
    ## load cache_data
    data_fetcher = DataFetcher()

    factor_formula = {
        'rv_umd': 'up_down_diff(ret(get_day_period(close(df)),1),0,std)',
        'rv_umd_u': 'up_down_diff_u(ret(get_day_period(close(df)),1),0,std)',
        'rv_umd_d': 'up_down_diff_d(ret(get_day_period(close(df)),1),0,std)',
        'rv_umd_ud': 'up_down_diff_ud(ret(get_day_period(close(df)),1),0,std)',
    }
    factor_rolling_method = {
        'rv_umd': ['MA-10','MA-5','MA-20','D'],
        'rv_umd_u': ['MA-10','MA-5','MA-20','D'],
        'rv_umd_d': ['MA-10','MA-5','MA-20','D'],
        'rv_umd_ud': ['MA-10','MA-5','MA-20','D'],
    }
    

    task_list = { key:[] for key in factor_formula.keys()}


    print("加载数据完成")
    for date in datelist:
        # 加载当日量价数据,统一加载
        dominant_code = data_fetcher.get_dominant_code(date)
        data = data_fetcher.get_price_ohlc_data(date = date)
        for factor,inputstr in factor_formula.items():
            if factor in ['cangdan','cangdan_huanbi']:
                data_input = data_fetcher.get_cangdan_data(date)
            elif factor in ['member_oi']:
                data_input = data_fetcher.get_member_oi_data(date)
            elif factor in ['jicha_mom']:
                data_input = data_fetcher.get_yuecha_ratio(date,method='raw_mom')
            elif factor in ['kucun_tongbi','kucun_shuiwei','kucun_huanbi_week','kucun_huanbi_month','kucun_huanbi_diff_week','kucun_huanbi_diff_month']:
                data_input = data_fetcher.get_kucun_data(date,method=factor[6:])
            else:
                data_input = data
            temp = {
                'date': date,
                'df': data_input,
                'inputstr': inputstr,
                'codes': dominant_code,
            }
            task_list[factor].append(temp)


    # for key,val in task_list.items():
    #     print(key)
    #     print(parse_func(**val[0]))



    # 计算因子，输出分组信息
    for key,tasks in task_list.items():
        print(key)
        res = multask(tasks=tasks, func=parse_func,process=12)
        res = pd.concat(res,axis=0)
        res = res.sort_index()

        for method in factor_rolling_method[key]:
            temp_res = processing_ind(res.copy(),method)
            group_res = temp_res.apply(lambda x:handle_one_bar(x,5),axis=1)
            file_dir = f'factor_res/{key}'
            if not os.path.exists(file_dir):
                os.makedirs(file_dir, exist_ok=True)

            if method is None:
                res.to_csv(file_dir + f"/{key}_val.csv")
                group_res.to_csv(file_dir + f"/{key}_group.csv")
            else:
                res.to_csv(file_dir + f"/{key}_val_{method}.csv")
                group_res.to_csv(file_dir + f"/{key}_group_{method}.csv")

















