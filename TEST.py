# -- coding: utf-8 --
"""
 @time : 2023/9/6
 @file : TEST.py
 @author : zhenghao
 @software: PyCharm
"""
import os

import pandas as pd

new_file_list = os.listdir("factor_res")

for file in new_file_list:
    new_df = pd.read_csv(f"factor_res/{file}/{file}_val.csv",index_col=0,parse_dates=[0])
    old_df = pd.read_csv(f"C:/Project/截面因子/factor_res/{file}/{file}_val.csv",index_col=0,parse_dates=[0])
    new_df = new_df.reindex(old_df.index)
    diff = new_df - old_df
    bias = diff.sum().sum()

    print(f"{file}:{bias}")


