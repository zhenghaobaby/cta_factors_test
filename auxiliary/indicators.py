# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

try:
    from utitilies import zero_series, empty_series
except:
    from .utitilies import zero_series, empty_series

""" 技术分析指标 """
def rolling_drawdown(arr, n):
    rolling_max = arr.rolling(n).max()
    draw_down = 100 * (arr - rolling_max) / rolling_max
    return draw_down


def atr(df, n=20, high='High', low='Low', close='Close'):
    sub = pd.DataFrame(dtype=float)
    sub['prv_close'] = df[close].shift()
    sub['high'] = df[high]
    sub['low'] = df[low]
    sub['h-l'] = sub['high'] - sub['low']
    sub['h-c'] = (sub['high'] - sub['prv_close']).abs()
    sub['l-c'] = (sub['low'] - sub['prv_close']).abs()
    true_range = sub[['h-l', 'h-c', 'l-c']].max(axis=1)
    return true_range.rolling(n).mean()


def atr_ratio(df: pd.DataFrame, n=20, high='High', low='Low', close='Close'):
    close_arr = df[close]  # .loc[df[close].first_valid_index():]
    sub = pd.DataFrame(dtype=float)
    sub['prv_close'] = df[close].shift()
    sub['high'] = df[high]
    sub['low'] = df[low]
    sub['h-l'] = sub['high'] - sub['low']
    sub['h-c'] = (sub['high'] - sub['prv_close']).abs()
    sub['l-c'] = (sub['low'] - sub['prv_close']).abs()
    true_range = sub[['h-l', 'h-c', 'l-c']].max(axis=1)
    result = 1000 * true_range / close_arr
    return result.rolling(n).mean()


def boll_flag(df, close='closeprice', high='highprice', n=60, std=3):
    mid = df[close].rolling(n).mean()
    band = df[close].rolling(n).std()
    upper = mid + std * band
    boll_flag_signal = zero_series(df[close])
    avg_price = (df[close] + df[high]) / 2
    prv_upper = upper.shift()
    prv_mid = mid.shift()
    boll_flag_signal[avg_price >= prv_upper] = 1
    boll_flag_signal[(avg_price < prv_upper) & (df[close] < prv_mid)] = 0
    return boll_flag_signal


def smma(series: pd.Series, window_size: int) -> pd.Series:
    return series.ewm(alpha=1 / window_size, min_periods=window_size).mean()


def rsi(series: pd.Series, window_size: int) -> pd.Series:
    """

    RSI = 100-100/(1+RS)
    RS= SMMA(U,T)/SMMA(D,T)
    SMMA是α=1/T 的EMA
    U : abs(收盘-前收)， D=0; D:前收-收盘，U=0
    """
    # ups = pd.Series(np.zeros(series.shape[0]), index=series.index)
    # downs = pd.Series(np.zeros(series.shape[0]), index=series.index)
    price_change = series.diff()
    ups = price_change.clip(lower=0)
    downs = price_change.clip(upper=0).abs()
    rs = smma(ups, window_size) / smma(downs, window_size)
    result = 100 - 100 / (1 + rs)
    return result


def zigzag(df: pd.DataFrame,
           n=60, atr_multi=10, min_retrace_pct=1,
           high='high', low='low', close='close', open='open'):
    atr_arr = atr(df, n=n, high=high, low=low, close=close, )
    retrace_point = pd.concat([df[close] * min_retrace_pct / 100, atr_arr * atr_multi], axis=1).max(axis=1,
                                                                                                    skipna=False)
    ZZHigh = None
    ZZLow = None
    ZZDirection = None
    ZZFlag = empty_series(df)
    ZZDirection_arr = empty_series(df)
    pivot_arr = empty_series(df)

    for idx, val in retrace_point.items():
        cur_close = df[close].loc[idx]
        if (ZZHigh is None) and (ZZLow is None):
            ZZHigh = df[open].loc[idx]
            ZZLow = df[low].loc[idx]
            pivot_idx = idx
        else:
            if ZZDirection is None:
                if cur_close >= ZZLow + val:
                    ZZHigh = cur_close
                    ZZDirection = 1
                    ZZFlag.loc[idx] = ZZHigh
                    # print(ZZHigh)
                    ZZDirection_arr.loc[idx] = ZZDirection
                    pivot_arr.loc[pivot_idx] = ZZLow
                    pivot_idx = idx
                    pivot_val = ZZHigh
                elif cur_close <= ZZHigh - val:
                    ZZLow = cur_close
                    ZZDirection = -1
                    ZZFlag.loc[idx] = ZZLow
                    # print(ZZLow)
                    ZZDirection_arr.loc[idx] = ZZDirection
                    pivot_arr.loc[pivot_idx] = ZZHigh
                    pivot_idx = idx
                    pivot_val = ZZLow
            else:
                if ZZDirection == 1:
                    if cur_close >= ZZHigh:
                        ZZHigh = cur_close
                        pivot_idx = idx
                        pivot_val = ZZHigh
                    elif cur_close <= ZZHigh - val:
                        ZZLow = cur_close
                        ZZDirection = -1
                        ZZFlag.loc[idx] = ZZHigh
                        ZZDirection_arr.loc[idx] = ZZDirection
                        pivot_arr.loc[pivot_idx] = pivot_val
                        pivot_idx = idx
                        pivot_val = cur_close
                else:
                    if cur_close <= ZZLow:
                        ZZLow = cur_close
                        pivot_idx = idx
                        pivot_val = ZZLow
                    elif cur_close >= ZZLow + val:
                        ZZHigh = cur_close
                        ZZDirection = 1
                        ZZFlag.loc[idx] = ZZLow
                        # print(ZZHigh)
                        ZZDirection_arr.loc[idx] = ZZDirection
                        pivot_arr.loc[pivot_idx] = pivot_val
                        pivot_idx = idx
                        pivot_val = cur_close
    pivot_arr.loc[idx] = cur_close
    ZZFlag = ZZFlag.ffill()
    ZZDirection_arr = ZZDirection_arr.ffill()
    pivot_arr_signal = pivot_arr.dropna().diff().apply(np.sign).shift(-1).reindex(pivot_arr.index).ffill()
    items = {'arr': df[close],
             'ZZDirection': ZZDirection_arr,
             'ZZFlag': ZZFlag,
             'Pivot': pivot_arr,
             'PivotSignal': pivot_arr_signal}
    return pd.DataFrame(items)


def zigzag_2(arr: pd.Series, theta=0.03):
    ZZHigh_arr = empty_series(arr)
    ZZHigh_arr.iloc[0] = arr.iloc[0]
    ZZLow_arr = empty_series(arr)
    ZZLow_arr.iloc[0] = arr.iloc[0]
    ZZDirection_arr = empty_series(arr)
    ZZDirection_arr.iloc[0] = 0
    MyLow_arr = empty_series(arr)
    MyLow_arr.iloc[0] = 0
    first_row = True
    prv_idx = None
    # 有未来数据的 信号点
    pivot_arr = empty_series(arr)
    for idx, val in arr.items():
        if first_row:
            first_row = False
            prv_idx = idx
            continue
        else:
            ZZDirection = ZZDirection_arr.loc[prv_idx]
            ZZHigh = ZZHigh_arr.loc[prv_idx]
            ZZLow = ZZLow_arr.loc[prv_idx]
            # 更新 高点
            if ZZDirection == 1:
                if val > ZZHigh:
                    ZZHigh = val
            else:
                if ZZDirection == -1:
                    if val > ZZLow + theta:
                        ZZHigh = val
                    else:
                        ...
                else:
                    if val > ZZLow + theta:
                        ZZHigh = val
                    else:
                        ...
            ZZHigh_arr.loc[idx] = ZZHigh

            # 更新低点
            if ZZDirection == -1:
                if val < ZZLow:
                    ZZLow = val
            else:
                if ZZDirection == 1:
                    if val < ZZHigh - theta:
                        ZZLow = val
                    else:
                        ...
                else:
                    if val < ZZHigh - theta:
                        ZZLow = val
                    else:
                        ...
            ZZLow_arr.loc[idx] = ZZLow

            # 更新 ZZDirection
            if ZZDirection == 0:
                if val > ZZLow + theta:
                    ZZDirection = 1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZLow].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZLow
                if val < ZZHigh - theta:
                    ZZDirection = -1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZHigh].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZHigh

            elif ZZDirection == 1:
                if val < ZZHigh - theta:
                    ZZDirection = -1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZHigh].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZHigh

            else:
                if val > ZZLow + theta:
                    ZZDirection = 1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZLow].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZLow
            ZZDirection_arr.loc[idx] = ZZDirection
            prv_idx = idx
    pivot_arr.loc[idx] = val
    pivot_arr_signal = pivot_arr.dropna().diff().apply(np.sign).shift(-1).reindex(pivot_arr.index).ffill()
    items = {'arr': arr,
             'ZZHigh': ZZHigh_arr,
             'ZZLow': ZZLow_arr,
             'ZZDirection': ZZDirection_arr,
             'Pivot': pivot_arr,
             'PivotSignal': pivot_arr_signal
             }
    res = pd.DataFrame(items)
    return res


def zigzag_3(arr: pd.Series, theta=0.03):
    ZZHigh_arr = empty_series(arr)
    ZZHigh_arr.iloc[0] = arr.iloc[0]
    ZZLow_arr = empty_series(arr)
    ZZLow_arr.iloc[0] = arr.iloc[0]
    ZZDirection_arr = empty_series(arr)
    ZZDirection_arr.iloc[0] = 0
    MyLow_arr = empty_series(arr)
    MyLow_arr.iloc[0] = 0
    first_row = True
    prv_idx = None
    # 有未来数据的 信号点
    pivot_arr = empty_series(arr)
    for idx, val in arr.items():
        if first_row:
            first_row = False
            prv_idx = idx
            continue
        else:
            ZZDirection = ZZDirection_arr.loc[prv_idx]
            ZZHigh = ZZHigh_arr.loc[prv_idx]
            ZZLow = ZZLow_arr.loc[prv_idx]

            # 更新 高点
            if ZZDirection == 1:
                if val > ZZHigh:
                    ZZHigh = val
            else:
                if ZZDirection == -1:
                    if val > ZZLow * ( 1 + theta):
                        ZZHigh = val
                    else:
                        ...
                else:
                    if val > ZZLow * (1 + theta):
                        ZZHigh = val
                    else:
                        ...
            ZZHigh_arr.loc[idx] = ZZHigh

            # 更新低点
            if ZZDirection == -1:
                if val < ZZLow:
                    ZZLow = val
            else:
                if ZZDirection == 1:
                    if val < ZZHigh * (1  - theta):
                        ZZLow = val
                    else:
                        ...
                else:
                    if val < ZZHigh * ( 1 - theta):
                        ZZLow = val
                    else:
                        ...
            ZZLow_arr.loc[idx] = ZZLow

            # 更新 ZZDirection
            if ZZDirection == 0:
                if val > ZZLow * ( 1 + theta):
                    ZZDirection = 1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZLow].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZLow
                if val < ZZHigh * ( 1 - theta):
                    ZZDirection = -1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZHigh].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZHigh

            elif ZZDirection == 1:
                if val < ZZHigh * ( 1- theta):
                    ZZDirection = -1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZHigh].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZHigh

            else:
                if val > ZZLow * (1  + theta):
                    ZZDirection = 1
                    pivot_idx = arr.loc[:idx][arr.loc[:idx] == ZZLow].index[-1]
                    pivot_arr.loc[pivot_idx] = ZZLow

            ZZDirection_arr.loc[idx] = ZZDirection
            prv_idx = idx

    pivot_arr.loc[idx] = val
    pivot_arr_signal = pivot_arr.dropna().diff().apply(np.sign).shift(-1).reindex(pivot_arr.index).ffill()

    items = {'arr': arr,
             'ZZHigh': ZZHigh_arr,
             'ZZLow': ZZLow_arr,
             'ZZDirection': ZZDirection_arr,
             'Pivot': pivot_arr,
             'PivotSignal': pivot_arr_signal
             }
    res = pd.DataFrame(items)
    return res



def get_breakout_signal(arr, n, background=None):
    if background is None:
        background = empty_series(arr).fillna(True)
    rolling_high = arr.rolling(n).max().shift()
    rolling_low = arr.rolling(n).min().shift()
    res = empty_series(arr)
    res[background & (arr > rolling_high)] = 1
    res[background & (arr < rolling_low)] = -1
    res[~background] = 'U'
    res = res.ffill().replace('U', 0)
    return res

def get_breakout_easy_short(arr, fast, slow, background=None):
    assert fast <= slow
    if background is None:
        background = empty_series(arr).fillna(1)
    fast_low = arr.rolling(fast).min().shift()
    slow_high = arr.rolling(slow).max().shift()
    arr = arr.reindex(background.index)
    res = empty_series(arr)
    res[(background == 1) & (arr > slow_high)] = 1
    res[(background == 1)& (arr < fast_low)] = -1
    res[~(background == 1)] = 'U'
    res = res.ffill()
    res = res.replace('U', 0)
    return res


def get_breakout_easy_long(arr, fast, slow, background=None):
    assert fast <= slow
    if background is None:
        background = empty_series(arr).fillna(1)
    fast_high = arr.rolling(fast).max().shift()
    slow_low = arr.rolling(slow).min().shift()
    res = empty_series(arr)
    res[(background == 1) & (arr > fast_high)] = 1
    res[(background == 1) & (arr < slow_low)] = -1
    res[~(background == 1)] = 'U'
    res = res.ffill()
    res = res.replace('U', 0)
    return res


