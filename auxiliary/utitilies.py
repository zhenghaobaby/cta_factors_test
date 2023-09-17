import configparser
from itertools import groupby
import pandas as pd
import numpy as np


"""找出前N大回撤"""
def get_top_n_drawdown(drawdown_arr, n=6):
    dd = drawdown_arr.abs()
    in_drawdown = zero_series(drawdown_arr)
    in_drawdown[dd != 0] = 1
    in_drawdown_list = get_signal_interval(in_drawdown)

    drawdown_record = []
    for start, end in in_drawdown_list:
        cur_drawdown = dd.loc[start: end].max()
        idx_max = dd.loc[start: end].idxmax()
        drawdown_record.append([start, idx_max, end, cur_drawdown])

    tb = pd.DataFrame(drawdown_record, columns=['start', 'max_date', 'end', 'drawdown']).sort_values(['drawdown'], ascending=False).reset_index(drop=True)
    return tb.iloc[:n]

def get_signal_interval(arr, ):
    """
    arr 是一个只有 0, 1 的序列，本函数返回一个列表：每一段连续的 1 的开始和结束
    :param arr:
    :return:
    """
    arr_cumsum = cumsum_with_reset(arr, reset=0).replace(0, np.nan)
    start_idx = list(arr_cumsum[arr_cumsum == 1].index)
    end_idx = list(arr_cumsum[arr_cumsum.notna() & arr_cumsum.shift(-1).isna()].index)
    try:
        assert len(start_idx) == len(end_idx)
        return list(zip(start_idx, end_idx))
    except AssertionError:
        end_idx.append(arr.index[-1])
        assert len(start_idx) == len(end_idx)
        return list(zip(start_idx, end_idx))



def cumsum_with_reset(arr: pd.Series, reset=0):
    """
    当遇到 == reset 的元素时，cumsum 归零；

    Args:
        arr:
        reset:

    Returns: pd.Series
    """
    if reset is None:
        v = arr.copy()
    else:
        v = arr.replace(reset, np.nan)
    cumsum = v.cumsum().fillna(method='pad')
    reset = -cumsum[v.isnull()].diff().fillna(cumsum)
    result = v.where(v.notnull(), reset).cumsum()
    return result


def empty_series(arr):
    try:
        return pd.Series(index=arr.index, dtype=float)
    except TypeError:
        return pd.Series(index=range(len(arr)), dtype=float)


def zero_series(arr):
    return empty_series(arr).fillna(0)


"""list_to_arr_form"""
def list_to_arr_form(arr, interval_list):
    res = zero_series(arr)
    for start, end in interval_list:
        res.loc[start: end] = 1
    return res



"""滚动窗口"""
def rolling_df(df, n,  step=1, min_window_size=None):
    if min_window_size is None:
        idx_start = n
        for i in range(idx_start, df.shape[0] + 1, step):
            try:
                yield df.iloc[i-n: i]
            except:
                yield df[i - n: i]
    else:
        idx_start = min_window_size
        for i in range(idx_start, df.shape[0] + 1, step):
            try:
                if i >= n:
                    yield df.iloc[i-n: i]
                else:
                    yield df.iloc[: i]
            except:
                if i >= n:
                    yield df[i-n: i]
                else:
                    yield df[: i]


"""滚动日期窗口，Panel data"""
def rolling_dates(df, n=244, date_col=None,interval=1, min_window_size=None):
    if date_col is None:
        date_arr = df.index.drop_duplicates()
        assert date_arr.is_monotonic
        df_group_indices = df.groupby(df.index).indices
    else:
        date_arr = df[date_col].drop_duplicates()
        assert date_arr.is_monotonic
        df_group_indices = df.groupby(date_col).indices

    for date_list in rolling_df(date_arr, n=n, step=interval, min_window_size=min_window_size):
        sub_list = []
        for date_idx in date_list:
            idx = df_group_indices[date_idx]
            sub_list.append(idx)
        yield df.iloc[np.concatenate(sub_list)]


def export_to_ini(array, dst, section=None):
    if section is None:
        section = 'Value'
    cf = configparser.ConfigParser()
    cf[section] = {}
    for idx, r in array.iteritems():
        dt = idx.strftime('%Y%m%d')
        cf[section][dt] = str(r)
    with open(dst, 'w') as file:
        cf.write(file)
        file.close()
    print('Done!')


def export_df_to_ini(df, dst,):
    df = df.copy().astype(str)
    df.index = df.index.strftime('%Y%m%d')
    df_dict = df.to_dict(orient='index')
    cf = configparser.ConfigParser()
    cf.optionxform = str
    cf.read_dict(df_dict)
    with open(dst, 'w') as configfile:
        cf.write(configfile)
        configfile.close()
    print('Done!')


# check elements all identical
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def quick_backtest(price: pd.Series, signal: pd.Series, shift=True, mode=None):
    assert price.shape == signal.shape
    if mode is None:
        mode = 'diff'
    else:
        if mode not in ['diff', 'simple', 'cumprod']:
            raise ValueError('未知的mode')

    # 当有信号的时候， 做空，无信号的时候，不做
    # 信号都是次日可交易
    if shift:
        pos = signal.shift().fillna(0)
    else:
        pos = signal
    pos = 0 * (pos == 0) + 1 * (pos == 1) + (-1) * (pos == -1)

    if mode == 'diff':
        pct_change = price.diff()
    else:
        pct_change = (price - price.shift()) / price.shift().abs()

    if mode == 'cumprod':
        pnl = (1 + (pct_change * pos)).cumprod()
    else:
        pnl = (pct_change * pos).cumsum()
    return pnl


def cal_metrics(pnl: pd.Series)->dict:
    pnl = pnl.copy()
    pnl = pnl.dropna()
    res = dict()
    res['收益%'] = (pnl.iloc[-1] - pnl.iloc[0]) * 100
    res['年化收益%'] = pnl.diff().mean() * 244 * 100
    res['年化波动%'] = pnl.diff().std() * np.sqrt(244) * 100
    res['sharpe'] = res['年化收益%'] / res['年化波动%']
    res['最大回撤%'] = abs((pnl - pnl.expanding().max()).min()) * 100
    res['最大回撤开始'] = pnl.loc[:(pnl - pnl.expanding().max()).idxmin()].idxmax()
    res['最大回撤结束'] = (pnl - pnl.expanding().max()).idxmin()
    res['收益回撤比'] = res['年化收益%'] / res['最大回撤%']
    return res
