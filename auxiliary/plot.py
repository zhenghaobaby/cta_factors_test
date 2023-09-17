# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
try:
    from utitilies import get_signal_interval
except:
    from .utitilies import get_signal_interval



def mesh_interval(interval_list, ax, color='gray', date_idx=True, alpha=1):
    """
    :param interval_list: [[start1, end1], [start2, end2],... ,[start_n, end_n]]
    :param ax:
    :param color:
    :param date_idx:
    :param alpha:
    :return:
    """
    if date_idx:
        for interval in interval_list:
            ax.axvspan(*mdates.datestr2num([str(j) for j in interval]), alpha=alpha, color=color, zorder=-10)
    else:
        for interval in interval_list:
            ax.axvspan(*interval, alpha=alpha, color=color, zorder=-10)
    return ax


def mesh_signal(signal: pd.Series, ax, color='gray', date_idx=True, alpha=1):
    """
    在某一图片上添加 signal == 1 的背景区间；
    :param signal:
    :param ax:
    :param color:
    :param date_idx: 如果 signal 和 ax 是 日期作为索引/x轴的，为True
    :param alpha:
    :return:
    """
    signal_list = get_signal_interval(signal, )
    ax = mesh_interval(signal_list, ax, color, date_idx, alpha)
    return ax



