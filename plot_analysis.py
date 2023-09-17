# -- coding: utf-8 --
"""
 @time : 2023/7/31
 @file : plot_analysis.py
 @author : zhenghao
 @software: PyCharm
"""

import glob
import os
import pandas as pd
import numpy as np
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import matplotlib.pyplot as plt


file_list = glob.glob(f"backtest_res/*/pnl.xlsx")
total = []
for file in file_list:
    path =os.path.split(os.path.split(file)[0])
    name = path[1]
    if name == 'ew_pnl_5':
        df = pd.read_excel(file,index_col=0)
    else:
        df = pd.read_excel(file,index_col=0)[['净值']]
        df.columns = [name]
    total.append(df)




total = pd.concat(total,axis=1)
total.to_csv("factor_jiankong.csv")
