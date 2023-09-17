# -- coding: utf-8 --
"""
 @time : 2023/8/22
 @file : tick_indicator.py
 @author : zhenghao
 @software: PyCharm
"""
import re
import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm

def parse_func(inputstr,df,date,codes):
    res = eval(inputstr)
    try:
        res = res[codes].to_frame().T
        res.index = [date]
        res.columns = [re.sub(r'\d+', '', k) for k in list(res.columns)]
    except:
        try:
            res = res.to_frame().T
            res.index = [date]
        except:
            return res
    return res




#region Fundamental fatcors

def cangdan(cangdan):
    cangdan = cangdan.fillna(0)
    cangdan_denominator = cangdan.iloc[-307:-181]
    cangdan_denominator = cangdan_denominator[cangdan_denominator != 0].mean(axis=0)
    cangdan_numerator = cangdan.iloc[-1]
    cangdan_tongcha = cangdan_numerator - cangdan_denominator

    # def non_zero_std(x):
    #     diff_x = x.diff()
    #     non_zero_x = x[(diff_x != 0) | (x != 0)]
    #
    #     if len(non_zero_x) / len(x) <= 0.5:
    #         return np.nan
    #     else:
    #         return x.std()
    #
    # # cangdan_std = cangdan.iloc[-244:].apply(lambda x: non_zero_std(x))
    cangdan_std = cangdan.iloc[-244:].apply(lambda x: x.std())
    alpha = cangdan_tongcha / cangdan_std
    alpha = -alpha
    return alpha


def cangdan_huanbi(cangdan):
    cangdan = cangdan.fillna(0)
    cangdan_huancha = cangdan.diff(20).iloc[-1]
    cangdan_std = cangdan.iloc[-244:].apply(lambda x: x.std())
    alpha = cangdan_huancha / cangdan_std
    alpha = -alpha
    alpha = alpha.clip(-3,3)
    return alpha


def member_oi(member_oi_rate):
    alpha = member_oi_rate
    return alpha

def jicha_mom(yuecha_ratio):
    alpha = yuecha_ratio.iloc[[-1]]
    return alpha

def kucun_tongbi(tongbi):
    alpha = -tongbi
    return alpha

def kucun_shuiwei(shuiwei):
    alpha = -shuiwei
    return alpha

def kucun_huanbi_week(huanbi):
    alpha = -huanbi
    return alpha
def kucun_huanbi_month(huanbi):
    alpha = -huanbi
    return alpha

def kucun_huanbi_diff_week(df):
    alpha = -df
    return alpha

def kucun_huanbi_diff_month(df):
    alpha = -df
    return alpha



#endregion







#region tick_fatcors

def VOI(df):
    # Order Imbalance Ratio
    df['bid_1_voldelta'] = df['bid_size1'].diff()
    df['ask_1_voldelta'] = df['ask_size1'].diff()
    df['bid_1_pdelta'] = df['bid_price1'].diff()
    df['ask_1_pdelta'] = df['ask_price1'].diff()

    df['delta_bid'] = np.where(df['bid_1_pdelta'] < 0, 0,
                               np.where(
                                   df['bid_1_pdelta'] == 0, df['bid_1_voldelta'],
                                   df['bid_size1']
                               ))
    df['delta_ask'] = np.where(df['ask_1_pdelta'] < 0, 0,
                               np.where(
                                   df['ask_1_pdelta'] == 0, df['ask_1_voldelta'],
                                   df['ask_size1']
                               ))

    df['voi'] = df['delta_bid'] - df['delta_ask']

    return df['voi']


def VOL_EWA(df):
    #Order Imbalance Ratio_ 五档加权
    sum=0
    df['voi_ewa'] = 0
    for i in range(1,6):
        weight = 1-(i-1)/5
        df[f'bid_{str(i)}_voldelta'] = df[f'bid_size{str(i)}'].diff()
        df[f'ask_{str(i)}_voldelta'] = df[f'ask_size{str(i)}'].diff()
        df[f'bid_{str(i)}_pdelta'] = df[f'bid_price{str(i)}'].diff()
        df[f'ask_{str(i)}_pdelta'] = df[f'ask_price{str(i)}'].diff()

        df[f'delta_bid_{str(i)}'] = np.where(df[f'bid_{str(i)}_pdelta'] < 0, 0,
                                   np.where(
                                       df[f'bid_{str(i)}_pdelta'] == 0, df[f'bid_{str(i)}_voldelta'],
                                       df[f'bid_size{str(i)}']
                                   ))
        df[f'delta_ask_{str(i)}'] = np.where(df[f'ask_{str(i)}_pdelta'] < 0, 0,
                                   np.where(
                                       df[f'ask_{str(i)}_pdelta'] == 0, df[f'ask_{str(i)}_voldelta'],
                                       df[f'ask_size{str(i)}']
                                   ))

        df[f'voi_{str(i)}'] =  df[f'delta_bid_{str(i)}'] - df[f'delta_ask_{str(i)}']
        df['vol_ewa'] += df[f'voi_{str(i)}']*weight
        sum+=weight

    df['voi_ewa'] = df['voi_ewa']/sum
    return df['voi_ewa']


def MOFI_EWA(df):
    """
    voi 当t bid< t-1 的bid的时候，取负V不是0
    """
    sum = 0
    df['mofi_ewa'] = 0
    for i in range(1, 6):
        weight = 1 - (i - 1) / 5
        df[f'bid_{str(i)}_voldelta'] = df[f'bid_size{str(i)}'].diff()
        df[f'ask_{str(i)}_voldelta'] = df[f'ask_size{str(i)}'].diff()
        df[f'bid_{str(i)}_pdelta'] = df[f'bid_price{str(i)}'].diff()
        df[f'ask_{str(i)}_pdelta'] = df[f'ask_price{str(i)}'].diff()
        df[f'pre_bid_size{str(i)}'] = df[f'bid_size{str(i)}'].shift()
        df[f'pre_ask_size{str(i)}'] = df[f'ask_size{str(i)}'].shift()

        df[f'delta_bid_{str(i)}'] = np.where(df[f'bid_{str(i)}_pdelta'] < 0,  df[f'pre_bid_size{str(i)}'] ,
                                             np.where(
                                                 df[f'bid_{str(i)}_pdelta'] == 0, df[f'bid_{str(i)}_voldelta'],
                                                 df[f'bid_size{str(i)}']
                                             ))
        df[f'delta_ask_{str(i)}'] = np.where(df[f'ask_{str(i)}_pdelta'] < 0, df[f'pre_ask_size{str(i)}'] ,
                                             np.where(
                                                 df[f'ask_{str(i)}_pdelta'] == 0, df[f'ask_{str(i)}_voldelta'],
                                                 df[f'ask_size{str(i)}']
                                             ))

        df[f'voi_{str(i)}'] = df[f'delta_bid_{str(i)}'] - df[f'delta_ask_{str(i)}']
        df['vol_ewa'] += df[f'voi_{str(i)}'] * weight
        sum += weight

    df['voi_ewa'] = df['voi_ewa'] / sum
    return df['voi_ewa']


def OIR(df):
    """
    OIR:  （加权委买量 - 加权委卖量）/(加权委买量+加权委卖量)
    OIR 为买卖委托量差与其和的比值，衡量了不均衡程度在其总买卖委托量中的占比
    """
    sum = 0
    df['w_bid_vol'] = 0
    df['w_ask_vol'] = 0
    for i in range(1, 6):
        weight = 1 - (i - 1) / 5
        sum+=weight
        df['w_bid_vol'] += df[f'bid_size{str(i)}']*weight
        df['w_ask_vol'] += df[f'ask_size{str(i)}']*weight

    df['w_bid_vol']/=sum
    df['w_ask_vol']/=sum

    df['OIR'] = (df['w_bid_vol']-df['w_ask_vol'])/(df['w_bid_vol']+df['w_ask_vol'])

    return df['OIR']


def SOIR(df):
    """
    OIR 是通过先将盘口委托数量衰减加权平均的方法，将不同档位的数据利用起来；而 SOIR 则是先计算每档 的委托数量失衡比例，然后再加权的方法，将不同档位的利用起来:
    """
    sum = 0
    df['soir'] = 0
    for i in range(1, 6):
        weight = 1 - (i - 1) / 5
        sum += weight
        df['soir_w'] += (df[f'bid_size{str(i)}'] - df[f'ask_size{str(i)}']) / (df[f'bid_size{str(i)}'] + df[f'ask_size{str(i)}'])
        df['soir']+= df['soir_w']*weight

    df['soir']/=df['soir']/sum

    return df['soir']


def PIR(df):
    """
    PIR 和 OIR 的计算方法非常接近，只是将委托量的数据变为委托的价格进行计算。根据每个股票的分钟委托 数据，可以计算买卖委托之间的价格比例 PIR（Price Imbalance Ratio）:
    """
    sum = 0
    df['w_bid_p'] = 0
    df['w_ask_p'] = 0
    for i in range(1, 6):
        weight = 1 - (i - 1) / 5
        sum += weight
        df['w_bid_p'] += df[f'bid_price{str(i)}'] * weight
        df['w_ask_p'] += df[f'ask_price{str(i)}'] * weight

    df['w_bid_p'] /= sum
    df['w_ask_p'] /= sum

    df['PIR'] = (df['w_bid_p'] - df['w_ask_p']) / (df['w_bid_p'] + df['w_ask_p'])

    return df['PIR']


def MPB(df):
    """
    我们此处定义一个时间段(t-1,t]的平均交易价格为TPt，即这一时间段的市场价格。中间价格表示为，它是时间t时买入和卖出价格的算术平均值，即这一时间段的平均委托挂单价格。
    当一时间段内，交易均价高于平均中间价格，交易均价更接近卖一价，卖方发起的交易，此时卖压大，未来的价格趋向下行的可能性大，
    且差值MPBt越大，未来价格走低的可能性就越高，因此交易均价将像市场平均中间价格回归。反之亦然，平均交易价格在平均中间价附近上下波动
    """

    df['vol'] = df['vol']

#endregion


#region OHLC col
def close(df):
    return df['close']

def open(df):
    return df['open']

def high(df):
    return df['high']

def low(df):
    return df['low']

def volume(df):
    return df['volume']

def turnover(df):
    return df['total_turnover']

def open_int(df):
    return df['open_interest']

#endregion



#region 切分函数

def get_day_minus_period(df,minus):
    date = df.index[-1]
    date = datetime.datetime(date.year, date.month, date.day)
    return df.loc[date + datetime.timedelta(hours=9,minutes=minus):date + datetime.timedelta(hours=14,minutes=60-minus)]


def get_day_period(df):
    date = df.index[-1]
    date = datetime.datetime(date.year,date.month,date.day)
    return df.loc[date + datetime.timedelta(hours=9):date+datetime.timedelta(hours=15)]

# func --上行-下行/总体
def up_down_diff(df,k,func):
    return (func(df[df>k]) - func(df[df<k]))/(func(df))
def up_down_diff_u(df,k,func):
    return (func(df[df>=k]) - func(df[df<k]))/(func(df))
def up_down_diff_d(df,k,func):
    return (func(df[df>k]) - func(df[df<=k]))/(func(df))
def up_down_diff_ud(df,k,func):
    return (func(df[df>=k]) - func(df[df<=k]))/(func(df))



# y以x排序前N个值,以func函数处理
def y_rank_by_x(y:pd.DataFrame,x:pd.DataFrame,func,rank=3):
    S_rank = x.rank(axis=0, method='first', ascending=False)
    S_area = S_rank <= rank
    alpha = func(y[S_area])
    return alpha


"""minute_distribution"""

def skew(df,window=None):
    if window is None:
        return df.dropna(how='all',axis=0).skew()
    else:
        return df.dropna(how='all',axis=0).rolling().skew()

def kurt(df,window=None):
    if window is None:
        return df.dropna(how='all',axis=0).kurt()
    else:
        return df.dropna(how='all',axis=0).rolling().kurt()


def std(df,window=None):
    if window is None:
        return df.std()
    else:
        return df.rolling(window=window).std()

def mean(df,window=None):
    if window is None:
        return df.mean()
    else:
        return df.rolling(window=window).mean()

def ret(df,window=1):
    return df.pct_change(window)

def diff(df,window=1):
    return df.diff(window)

def minus_median_abs(df):
    return abs(df-df.median())


def ranknorm(df: pd.Series, trunc: float = 0.00135) -> pd.Series:
    nd_ranknorm = norm.ppf(df.rank(method = 'average', pct=True).clip(lower = trunc, upper = 1-trunc)) / norm.ppf(1 - trunc)
    df_ranknorm = pd.Series(nd_ranknorm, index = df.index)
    return df_ranknorm











