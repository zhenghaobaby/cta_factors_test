import datetime

import pandas as pd
from pathlib import Path

TRD_CALENDER_SRC = Path(__file__).parent / 'all_tradeday.csv'
ALL_TRADEDAY = pd.read_csv(TRD_CALENDER_SRC, parse_dates=[0])['tradeday']
ALL_TRADEDAY.index = ALL_TRADEDAY.values
"""
日期函数
"""
def get_month_end(df):
    """
    df 是索引为日期的数据，返回索引中每个月的最后一天；
    :param df:
    :return:
    """
    df = pd.DataFrame(data=df.index, index=df.index, )
    df.columns = ['date']
    grouper = df.groupby(df.index.strftime('%Y-%m'))
    res = grouper.last().squeeze().values
    return res


def get_month_start(df):
    """
    df 是索引为日期的数据，返回索引中每个月的第一天；
    :param df:
    :return:
    """
    df = pd.DataFrame(data=df.index, index=df.index, )
    df.columns = ['date']
    grouper = df.groupby(df.index.strftime('%Y-%m'))
    res = grouper.first().squeeze().values
    return res


# 是否是交易日
def is_tradeday(date):
    date_ = pd.to_datetime(date)
    res = (date_ == ALL_TRADEDAY).any()
    return res


# 获取最近的交易日
def get_nearest_tradeday(date_arr, directions=1):
    # 当该日期不是交易日的时候，-1 往前最近一个交易日，1往后最近一个交易日
    ...


def get_tradeday_for_morning_task():
    now_time = datetime.datetime.now()
    all_tradeday = pd.read_csv(TRD_CALENDER_SRC, parse_dates=[0])
    all_tradeday['close'] = 0
    all_tradeday.set_index('tradeday', inplace=True)
    sub_tradeday = all_tradeday.loc['2010-01-04': now_time].copy()
    if (pd.to_datetime(now_time.date()) == sub_tradeday.index).any():
        if now_time.hour < 18:
            sub_tradeday = sub_tradeday.iloc[: -1]
    return sub_tradeday



if __name__ == '__main__':
    print(ALL_TRADEDAY.head())
    print(is_tradeday('2023-01-01'))
    res = get_month_end(ALL_TRADEDAY)
    print(any(pd.to_datetime('2026-11-30') == res))