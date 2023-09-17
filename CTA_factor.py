
"""
本文件根据输入因子名称与对应品种列表生成对应的因子数据与信号
"""
import sys
sys.path.append(r"C:\Wind\Wind.NET.Client\WindNET\x64")
sys.path.append(r"C:\Project\Fundamental_Store\utils")
sys.path.append(r"C:\Project\Fundamental_Store\MySQL")

from fetch_data import fetch_data
from common import *


SymbolMultiplier = {'RB': 10, 'HC': 10, 'I': 100, 'J': 100, 'JM': 60, 'ZC': 100, 'FG': 20, 'MA': 10, 'TA': 5,
                    'L': 5, 'PP': 5, 'BU': 10, 'RU': 10, 'CU': 5, 'AL': 5, 'ZN': 5, 'PB': 5, 'NI': 1, 'SN': 1,
                    'M': 10, 'Y': 10, 'RM': 10, 'OI': 10, 'P': 10, 'A': 10, 'C': 10, 'CS': 10, 'CF': 5, 'SR': 10,
                    'JD': 10, 'V': 5, 'EG': 10, 'AG': 15, 'AU': 1000, 'SC': 1000, 'B': 10, 'IF': 300, 'IH': 300,
                    'IC': 200, 'FU': 10, 'SA': 20, 'SS': 5, 'EB': 5, 'SP': 10, 'UR': 20, }
def upper_bound(nums, target):
    low, high = 0, len(nums) - 1
    pos = len(nums)
    while low < high:
        mid = int((low + high) / 2)
        if nums[mid] <= target:
            low = mid + 1
        else:  # >
            high = mid
            pos = high
    if nums[low] > target:
        pos = low
    return pos - 1


def adjust_tongbi_ratio(x,date_list,value,inforce_date):
    if np.isnan(x['仓单']) or x['仓单']==0:
        return 1e8
    else:
        if x['日期'].month == 2 and x['日期'].day == 29:
            pre_time = datetime.datetime(x['日期'].year - 1, x['日期'].month, x['日期'].day - 1)
        else:
            pre_time = datetime.datetime(x['日期'].year - 1, x['日期'].month, x['日期'].day)
        pre_time = pd.to_datetime(pre_time)

        if upper_bound(inforce_date,pre_time)<0:
            return np.nan

        else:
            pre_cut_date = inforce_date[upper_bound(inforce_date,pre_time)]
            pre_cut_pos = upper_bound(date_list,pre_cut_date)

            now_cut_date = inforce_date[upper_bound(inforce_date, x['日期'])]
            now_cut_pos = upper_bound(date_list, now_cut_date)
            now_pos = upper_bound(date_list, x['日期'])
            distance = now_pos - now_cut_pos

            pre_pos = pre_cut_pos+distance
            pre_val = value[pre_pos][0]


            if pre_val==0 or np.isnan(pre_val):
                return 1e8
            else:
                now_val = x['仓单']
                return now_val/pre_val



def adjust_month_tongbi_ratio(x,date_list,value,inforce_date):
    #为了一定能够找到去年相同月份，我们设置月中时间，再用二分法进行锚定搜索
    pre_time = datetime.datetime(x['日期'].year - 1, x['日期'].month, 15)
    pre_time = pd.to_datetime(pre_time)

    if upper_bound(inforce_date,pre_time)<0:
        return np.nan

    else:
        pre_cut_date = inforce_date[upper_bound(inforce_date,pre_time)]
        pre_cut_pos = upper_bound(date_list,pre_cut_date)

        now_cut_date = inforce_date[upper_bound(inforce_date, x['日期'])]
        now_cut_pos = upper_bound(date_list, now_cut_date)
        now_pos = upper_bound(date_list, x['日期'])
        distance = now_pos - now_cut_pos

        pre_pos = pre_cut_pos+distance
        # 检查是否超过当前月份
        while date_list[pre_pos].month!=x['日期'].month:
            pre_pos-=1
        pre_val = value[pre_pos][0]
        now_val = x['仓单']
        return now_val-pre_val



def processing_data(type_name, table_name):
    store_data = fetch_data(db_name=type_name, table_name=table_name)
    store_data.index = pd.to_datetime(store_data['日期'])
    del store_data['日期']
    store_data = store_data.reset_index().drop_duplicates(subset='日期', keep='last').set_index('日期')
    return store_data

def xrank(x,ratio):
    choose_number = int(len(x.dropna())*ratio)
    x_rank = x.rank(method='min', ascending=False)
    x_rank_reverse = x.rank(method='min')
    long_pos = x_rank[x_rank <= choose_number]
    short_pos = x_rank_reverse[x_rank_reverse <= choose_number]

    x_long = []
    x_short = []

    for k in list(long_pos.index):
        x_long.append(k)
    for j in list(short_pos.index):
        x_short.append(j)
    for m in list(x.index):
        if m in x_long:
            x[m] = 1
        elif m in x_short:
            x[m] = -1
        else:
            x[m] = 0
    return x


def month_diff_cal(x,y):
    try:
        dominant_month = int(str(x)[-2:])
    except:
        dominant_month = int(str(x)[-1:])

    try:
        sec_dominant_month = int(str(y)[-2:])
    except:
        sec_dominant_month = int(str(y)[-1:])


    if x==y:
        month_diff = 0
    else:
        if sec_dominant_month<=dominant_month:
            month_diff = sec_dominant_month+12-dominant_month
        else:
            month_diff = sec_dominant_month-dominant_month
    return month_diff


def year_month_rate(x, type_name):
    res = np.log(x[type_name + "主力收盘价"]/x[type_name + "次主力收盘价"])*(12 / x['月份差']) * 100

    return res


def XTSMOM(type_name,N):
    """
    截面涨幅  公式： alpha = x[-1]/x[0]-1
    :param type_list: 选取品种列表
    :param N: 涨幅回望期
    :return: 因子原始值
    """

    price = processing_data(type_name,'指数连续')
    price[type_name] = price[type_name].rolling(N).apply(lambda x:x[-1]/x[0]-1)

    return price[type_name].to_frame()

def TSMOM(type_name,N):
    """
      做多上涨，做空下跌  公式： alpha = x[-1]/x[0]-1
      :param type_list: 选取品种列表
      :param N: 涨幅回望期
      :return: 因子信号
      """

    price = processing_data(type_name, '指数连续')
    price[type_name] = price[type_name].rolling(N).apply(lambda x: x[-1] / x[0] - 1)
    res = price[[type_name]]
    res[res>0]=1
    res[res<0]=-1
    res[res==0]=0
    res.fillna(0,inplace=True)
    return res

def Mkt_effience(type_name,N):
    price = processing_data(type_name,'指数连续')
    price['diff'] = price[type_name].diff()
    price['road'] = price['diff'].rolling(N-1).apply(lambda x:abs(x).sum())
    price['shift'] = price[type_name].rolling(N).apply(lambda x:x.max()-x.min())
    price['trend'] = price[type_name].rolling(N).apply(lambda x: np.sign(x[-1]-x[0]))
    price['sr'] = price['trend']*price['shift']/price['road']
    res = price[['sr']]
    res.columns = [type_name]
    return res

def RSM(type_name,N,threshold):
    """
    计算过去N个交易日上涨个数占比与下跌个数占比，大于阈值做多，小于阈值做空
    :param type_list: 选取品种
    :param N: 回望期
    :param threshold: 阈值
    :return: 因子多空信号
    """
    price = processing_data(type_name, '指数连续')
    price['updown'] = price[type_name].rolling(2).apply(lambda x:np.sign(x[-1]-x[0]))
    price[type_name] = price['updown'].rolling(N).apply(lambda x: (x[x>0].sum())/len(x))
    res = price[type_name]
    res[res>threshold]=1
    res[res<threshold]=-1
    res[res==threshold]=0
    res.fillna(0,inplace=True)
    return res

def Complex_MoM(type_name,N,threshold):
    """
    做多累计涨幅大于0且上涨天数大于阈值的品种，反之做空
    :param type_list: 选取品种
    :param N: 回望天数
    :param threshold: 阈值
    :return: 因子信号
    """
    price = processing_data(type_name, '指数连续')
    price['updown'] = price[type_name].rolling(2).apply(lambda x: np.sign(x[-1] - x[0]))
    price['upratio'] = price['updown'].rolling(N).apply(lambda x: (x[x > 0].sum()) / len(x))
    price['ret'] = price[type_name].rolling(N).apply(lambda x: x[-1]/x[0]-1)
    price[type_name] = price.apply(lambda x: 1 if x['ret']>0 and x['upratio']>threshold else -1 if
                              x['ret']<0 and x['upratio']<threshold else 0,axis=1)
    res = price[[type_name]]
    return res


def term_structure(type_name,mode):
    """
    计算期限结构给出的月差收益率因子
    mode 1: 主力-次主力   ln（主力/次主力)*（12/月份差）
         2： 近月-主力    ln（近月/主力）*（12/月份差）
    :param type_list: 选取品种列表
    :param mode: 1：主力-次主力模式  2. 近月-主力模式
    :return: 因子原始值，排序值，排序信号值
    """
    if mode==1:
        df = processing_data(type_name,'月差')
        df.dropna(inplace=True)
        df['月份差'] = df.apply(lambda x: month_diff_cal(x['主力合约'],x['次主力合约']), axis=1)
        df[type_name] = df.apply(lambda x: year_month_rate(x, type_name), axis=1)
        res = df[[type_name]]
        return res

    elif mode==2:

        df = processing_data(type_name,'期限结构')
        df_1 = processing_data(type_name,'月差')
        df['主力合约index'] = df.apply(lambda x: eval(x['可交易合约集合']).index(x['主力合约code']),axis=1)
        df['选取合约成交量'] = df.apply(lambda x: eval(x['volume'])[0:x['主力合约index']],axis=1)
        df['选取合约持仓量'] = df.apply(lambda x: eval(x['open_interest'])[0:x['主力合约index']],axis=1)

        def f(x):
            if x['主力合约index']==0:
                return int(x['主力合约index'])
            else:
                max_vol = max(x['选取合约成交量'])
                max_oi = max(x['选取合约持仓量'])
                if max_vol==0:
                    if max_oi==0:
                        return int(x['主力合约index']-1)
                    else:
                        return int(x['选取合约持仓量'].index(max_oi))
                else:
                    return int(x['选取合约成交量'].index(max_vol))


        df['近月index'] = df.apply(lambda x: f(x),axis=1)
        df['近月价格'] = df.apply(lambda x: float(eval(x['close'])[x['近月index']]),axis=1)
        df['近月合约'] = df.apply(lambda x: eval(x['可交易合约集合'])[x['近月index']],axis=1)
        df['月份差'] = df.apply(lambda x:month_diff_cal(x['近月合约'],x['主力合约code']),axis=1)
        df['主力价格'] = df_1[type_name+'主力收盘价']
        df[type_name] = df.apply(lambda x: 0 if x['月份差']==0 else np.log(x['近月价格']/x['主力价格'])*12/(x['月份差'])*100,axis=1)
        res = df[[type_name]]
        return res

    else:
        pass

def Jicha_Mom(type_name,mode,N):
    """
    jicha_mom: ret(近月，N) - ret(远月，N)
    :param type_list:选取品种
    :param mode: 1：主力-次主力  2：近月-主力
    :return: 因子原始值，排序值，排序信号值
    QA: 近月连续ret-主力连续ret是否合理？是否复权处理合约价格？
    """
    if mode==1:
        df = processing_data(type_name,'月差')
        df.dropna(inplace=True)
        df['主力ret'] = df[type_name+'主力收盘价'].rolling(N).apply(lambda x:x[-1]/x[0]-1)
        df['次主力ret'] =df[type_name+'次主力收盘价'].rolling(N).apply(lambda x:x[-1]/x[0]-1)
        df[type_name] = df['主力ret'] - df['次主力ret']
        res = df[[type_name]]
        return res

    elif mode==2:
        df = processing_data(type_name, '期限结构')
        df_1 = processing_data(type_name, '月差')
        df['主力合约index'] = df.apply(lambda x: eval(x['可交易合约集合']).index(x['主力合约code']), axis=1)
        df['选取合约成交量'] = df.apply(lambda x: eval(x['volume'])[0:x['主力合约index']], axis=1)
        df['选取合约持仓量'] = df.apply(lambda x: eval(x['open_interest'])[0:x['主力合约index']], axis=1)

        def f(x):
            if x['主力合约index'] == 0:
                return int(x['主力合约index'])
            else:
                max_vol = max(x['选取合约成交量'])
                max_oi = max(x['选取合约持仓量'])
                if max_vol == 0:
                    if max_oi == 0:
                        return int(x['主力合约index'] - 1)
                    else:
                        return int(x['选取合约持仓量'].index(max_oi))
                else:
                    return int(x['选取合约成交量'].index(max_vol))

        df['近月index'] = df.apply(lambda x: f(x), axis=1)
        df['近月价格'] = df.apply(lambda x: float(eval(x['close'])[x['近月index']]), axis=1)
        df['近月合约'] = df.apply(lambda x: eval(x['可交易合约集合'])[x['近月index']], axis=1)
        df['月份差'] = df.apply(lambda x: month_diff_cal(x['近月合约'], x['主力合约code']), axis=1)
        df['主力价格'] = df_1[type_name + '主力收盘价']


        df['近月ret'] = df['近月价格'].pct_change(N)
        df['主力ret'] = df['主力价格'].pct_change(N)
        df[type_name] = df['近月ret'] - df['主力ret']
        res = df[[type_name]]
        return res

    elif mode==3:
        df = processing_data(type_name, '月差')
        df.dropna(inplace=True)
        df['月份差'] = df.apply(lambda x: month_diff_cal(x['主力合约'], x['次主力合约']), axis=1)
        df['月差率'] = df.apply(lambda x: year_month_rate(x, type_name), axis=1)
        df['极差'] = df['月差率'].rolling(244).apply(lambda x:x.max()-x.min())
        df['月差变化'] = df['月差率'].rolling(N).apply(lambda x:x[-1]-x[0])
        df[type_name] = df['月差变化']/df['极差']
        res = df[[type_name]]
        return res
    else:
        pass

def value(type_name,N):
    """
    商品价值因子，过去N年的现货价格（以期货价格替代）除现在的现货价格
    过去价格取 （N-0.5,N+0.5）的价格均值
    :param type_list:选取品种列表
    :param N: 回望年数
    :return: 因子值
    """
    left = int((N+0.5)*244)
    right = int((N-0.5)*244)
    price = processing_data(type_name, '指数连续')
    price[type_name] = price[type_name].rolling(left).apply(lambda x: np.log(x[:244].mean()/x[-1]))
    res =price[[type_name]]
    return res

def skewness(type_name,N):
    """
    过去收益率分布的偏度作为因子值
    :param type_list:
    :param N: 回望期
    :return: 因子值
    """

    price = processing_data(type_name, '指数连续')
    price['ret'] = price[type_name].rolling(2).apply(lambda x:x[-1]/x[0]-1)
    price[type_name] = price['ret'].rolling(N).skew()
    res = -price[[type_name]]
    return res

def Kurtosis(type_name,N):
    """
    过去收益率分布的峰度作为因子值
    :param type_list:
    :param N: 回望期
    :return: 因子值
    """

    price = processing_data(type_name, '指数连续')
    price['ret'] = price[type_name].rolling(2).apply(lambda x: x[-1] / x[0] - 1)
    price[type_name] = price['ret'].rolling(N).kurt()
    res = -price[[type_name]]
    return res


def OI_ratio(type_name,N):
    """
    做多持仓量变化率最大的品种，做空持仓变化最低的品种
    :param type_list:选取品种
    :param N:回望天数
    :return:因子值
    QA: 需要尝试切换到主力如何
    """

    df = processing_data(type_name, '指数连续v2')
    df[type_name] = df['open_interest'].rolling(N).apply(lambda x:np.log(x[-1]/x[0]))
    res = df[[type_name]]
    return res


def liquidity(type_name,N):
    """
    （收益率绝对值/交易额）的N日均值
    :param type_list: 选取品种
    :param N: 回望期
    :return: 因子值
    QA：切换到主力合约？
    """
    df = processing_data(type_name, '指数连续v2')
    df['ret'] = df['close'].rolling(2).apply(lambda x:x[-1]/x[0]-1)
    df['ILLIQ'] = df.apply(lambda x: np.nan if x['total_turnover']==0 else abs(x['ret'])/x['total_turnover'],axis=1)
    df[type_name] = df['ILLIQ'].rolling(N).mean()
    res = -df[[type_name]]
    return res


def cangdan_factor_tongbi(type_name):
    """
    仓单研报的同比计算得到因子
    :param type_list:
    :return:
    """
    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()
    cangdan['仓单分母'] = cangdan['仓单'].rolling(307).apply(lambda x: x[:126][x != 0].mean())
    cangdan['仓单分子'] = cangdan['仓单']
    cangdan['仓单同差'] = cangdan['仓单分子'] - cangdan['仓单分母']

    def non_zero_std(x):
        diff_x = x.diff()
        non_zero_x = x[(diff_x!=0)|(x!=0)]

        if len(non_zero_x)/len(x)<=0.5:
            return np.nan
        else:
            return x.std()


    cangdan['基准值'] = cangdan['仓单'].rolling(244).apply(lambda x:non_zero_std(x))
    # cangdan['基准值'] = cangdan['仓单'].rolling(244).std()
    cangdan['仓单同比'] = cangdan['仓单同差'] / cangdan['基准值']
    res = -cangdan[['仓单同比']]
    res.columns = [type_name]
    return res




def cangdan_factor_huanbi(type_name,N):
    """
    计算注册仓单的环比变化率，如果初期仓单数据为0，则得到nan值不给予排序
    :param type_list: 选取品种
    :param N: 回望期
    :return: 因子值
    """
    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()

    cangdan['环比'] = cangdan['仓单'].rolling(N).apply(lambda x:x[-1]-x[0])
    cangdan['基准值'] = cangdan['仓单'].rolling(244).std()
    cangdan['环比比例'] = cangdan['环比']/cangdan['基准值']
    res = -cangdan[['环比比例']]
    res.columns = [type_name]
    return res


def cangdan_factor_huanbi_ts(type_name,N,thresholdin,thresholdout):
    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()

    def non_zero_std(x):
        diff_x = x.diff()
        non_zero_x = x[(diff_x != 0) | (x != 0)]

        if len(non_zero_x) / len(x) <= 0.5:
            return np.nan
        else:
            return x.std()

    cangdan['环比'] = cangdan['仓单'].diff(N)
    cangdan['基准值'] = cangdan['仓单'].rolling(244).std()
    cangdan['环比比例'] = cangdan['环比']/cangdan['基准值']
    cangdan['环比信号'] = cangdan['环比比例'].rolling(2).apply(lambda x:func_internia(x,thresholdin,-thresholdin,thresholdout,-thresholdout))
    cangdan['环比信号'] = cangdan['环比信号'].ffill()
    res = cangdan[['环比信号']]
    res.columns = [type_name]
    return res






def cangdan_factor_shuiwei(type_name,N):
    """
    计算注册仓单的环比变化率，如果初期仓单数据为0，则得到nan值不给予排序
    :param type_list: 选取品种
    :param N: 回望期
    :return: 因子值
    """
    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()

    cangdan['水位'] = cangdan['仓单'].rolling(N).apply(lambda x:water(x))
    res = cangdan[['水位']]
    res.columns = [type_name]
    return res


def cangdan_adjust_tongbi(type_name):

    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    Tradingday = Tradingday[Tradingday.index>=pd.to_datetime("2010-01-01")]
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = cangdan.iloc[:, 1].to_frame()
    cangdan.fillna(0,inplace=True)

    inforce_date = pd.read_excel("仓单有效期完整.xlsx", index_col=0)
    inforce_date = inforce_date[type_name].dropna().to_list()
    date_list = list(cangdan.index)
    value = cangdan.values
    cangdan = cangdan.reset_index()
    cangdan['同比'] = cangdan.apply(lambda x: adjust_tongbi_ratio(x,date_list,value,inforce_date), axis=1)
    cangdan = cangdan.set_index("日期")

    res = -cangdan[['同比']]
    res.columns = [type_name]

    return res


def cangdan_month_adjust_tongbi(type_name):
    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    Tradingday = Tradingday[Tradingday.index>=pd.to_datetime("2010-01-01")]
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()

    inforce_date = pd.read_excel("月度初始交易日.xlsx", index_col=0)
    inforce_date = inforce_date[type_name].dropna().to_list()
    date_list = list(cangdan.index)
    value = cangdan.values
    cangdan = cangdan.reset_index()
    cangdan['仓单同差'] = cangdan.apply(lambda x: adjust_month_tongbi_ratio(x,date_list,value,inforce_date), axis=1)
    cangdan = cangdan.set_index("日期")
    cangdan['基准值'] = cangdan['仓单'].rolling(244).std()
    cangdan['仓单同比'] = cangdan['仓单同差'] / cangdan['基准值']
    res = -cangdan[['仓单同比']]
    res.columns = [type_name]

    # return res
    return cangdan[['仓单']],cangdan[['仓单同差']],cangdan[['仓单同比']]



def member_OI(type_name,N,rank,factor):
    """
    factor_1: 净多头持仓量变化率  netlong_t/netlong_(t-N)-1
    factor_2: 净多头/总持仓的变化率 δnetlong_t/total_OI
    factor_3: 净多头持仓量占会员持仓比例  (long-short)/(long+short)
    factor_4: 多头持仓变化率 long_t/long_(t-N)-1
    factor_5: 空头持仓变化率 short_t/short_(t-N)-1
    factor_6: 多头持仓量相对总持仓占比变化  δlong/total_OI
    factor_7: 空头持仓量相对总持仓占比变化相反数 -δshort/total_OI
    factor_8: 多头与空头持仓变化方向 sign(long_t/long_(t-N)-1)+sign(short_(t-N)/short_t-1)
    factor_9: 多头持仓变化方向 sign(long_t/long_(t-N)-1)
    factor_10: 空头持仓变化方向 sign(short_(t-N)/short_t-1)
    factor_11: 知情度： (long+short)/total_oi
    factor_12: 净多持仓变化率 (long-short)/(long+short)的变化
    factor_13: 多头持仓率
    factor_14: 空头持仓率

    :param type_list:选取品种
    :param N: 回望期
    :param rank: top数
    :param factor: 选取因子代号
    :return: 因子值
    """

    df_long = fetch_data(type_name,'会员多头持仓')
    df_short = fetch_data(type_name,'会员空头持仓')

    long_start_date = df_long['日期'][0]
    short_start_date = df_short['日期'][0]
    start_date = max(long_start_date,short_start_date)
    df_long = df_long[df_long['日期']>=start_date]
    df_short = df_short[df_short['日期']>=start_date]

    total_OI = fetch_data(type_name,'指数连续v2')
    df_long = df_long[df_long['排名']<=rank]
    df_short = df_short[df_short['排名']<=rank]
    long_oi = []
    short_oi = []
    mkt_oi = []


    for tradingday,temp in df_long.groupby("日期"):
        long_oi.append(temp['持仓量'].sum())


    for tradingday,temp in df_short.groupby("日期"):
        short_oi.append(temp['持仓量'].sum())

        if len(total_OI.loc[total_OI['日期']==tradingday,'open_interest']) == 0:
            mkt_oi.append(np.nan)
        else:
            mkt_oi.append(total_OI.loc[total_OI['日期']==tradingday,'open_interest'].values[0])


    df = {}

    df['多头持仓'] = long_oi
    df['空头持仓'] = short_oi
    df['市场持仓'] = mkt_oi
    df = pd.DataFrame(df,index=df_long.copy().drop_duplicates('日期')['日期'])
    df['净多头持仓'] = df['多头持仓']-df['空头持仓']


    ## 计算不同因子
    if factor==1:
        df[type_name] = df['净多头持仓'].rolling(N).apply(lambda x:x[-1]/x[0]-1)
    elif factor==2:
        df['净多头占比'] = df['净多头持仓']/df['市场持仓']
        df[type_name] = df['净多头占比'].rolling(N).apply(lambda x:x[-1]-x[0])
    elif factor==3:
        df[type_name] = df.apply(lambda x:(x['多头持仓']-x['空头持仓'])/(x['多头持仓']+x['空头持仓']),axis=1)
    elif factor==4:
        df[type_name] = df['多头持仓'].rolling(N).apply(lambda x:x[-1]/x[0]-1)
    elif factor==5:
        df[type_name]  = df['空头持仓'].rolling(N).apply(lambda x:-(x[-1]/x[0]-1))
    elif factor==6:
        df['多头持仓比例'] = df['多头持仓']/df['市场持仓']
        df[type_name]  = df['多头持仓比例'].rolling(N).apply(lambda x:x[-1]-x[0])
    elif factor==7:
        df['空头持仓比例'] = df['空头持仓'] / df['市场持仓']
        df[type_name] = df['空头持仓比例'].rolling(N).apply(lambda x: -(x[-1]-x[0]))
    elif factor==8:
        df['多头持仓变化'] = df['多头持仓'].rolling(N).apply(lambda x: x[-1] / x[0] - 1)
        df['空头持仓变化'] = df['空头持仓'].rolling(N).apply(lambda x: x[0] / x[-1] - 1)
        df[type_name] = df.apply(lambda x: np.sign(x['多头持仓变化'])+np.sign(x['空头持仓变化']),axis=1)
    elif factor==9:
        df['多头持仓变化'] = df['多头持仓'].rolling(N).apply(lambda x: x[-1] / x[0] - 1)
        df[type_name]  = df.apply(lambda x: np.sign(x['多头持仓变化']))
    elif factor==10:
        df['空头持仓变化'] = df['空头持仓'].rolling(N).apply(lambda x: x[0] / x[-1] - 1)
        df[type_name]  = df.apply(lambda x: np.sign(x['空头持仓变化']))
    elif factor==11:
        df[type_name] = (df['多头持仓']+df['空头持仓'])/df['市场持仓']
    elif factor==12:
        df['多头比例'] = df.apply(lambda x:(x['多头持仓']-x['空头持仓'])/(x['多头持仓']+x['空头持仓']),axis=1)
        df[type_name] = df['多头比例'].rolling(N).apply(lambda x:x[-1]-x.mean())
    elif factor==13:
        df[type_name] = df.apply(lambda x: x['多头持仓'] / (x['多头持仓'] + x['空头持仓']), axis=1)
    elif factor==14:
        df[type_name] = df.apply(lambda x: x['空头持仓'] / (x['多头持仓'] + x['空头持仓']), axis=1)

    res = df[[type_name]]
    return res





def kaigong_rate(type_name,N):
    df = pd.read_excel("整体开工率.xlsx",index_col=0)
    df['开工同比'] = df[type_name].rolling(244).apply(lambda x:x[-1]/x[0])
    df[type_name] = df['开工同比'].rolling(N).apply(lambda x:-x[-1]/x[0])
    return df[[type_name]]

def lirun(type_name,N):
    df = pd.read_excel("整体利润.xlsx",index_col=0)
    df['水位'] = df[type_name].rolling(N).apply(lambda x:water(x))
    res = -df[['水位']]
    res.columns = [type_name]
    return res

def get_month_OI(type_name):
    df = processing_data(type_name, '期限结构')
    df = df.reset_index()
    df['month'] = df['日期'].apply(lambda x: x.month)
    df['year'] = df['日期'].apply(lambda x:x.year)

    if type_name=='SC':
        df['now_code'] = df.apply(lambda x:
                                  type_name + str(x['year']+1)[2:] +  str(0)+ str(x['month'] + 1-12) if (x['month']+1) > 12 else
                                  type_name + str(x['year'])[2:] + str(x['month']+1) if (x['month']+1) >= 10 else
        type_name + str(x['year'])[2:] + str(0) + str(x['month']+1), axis=1)

    else:
        df['now_code'] = df.apply(lambda x: type_name + str(x['year'])[2:] + str(x['month']) if x['month'] >= 10 else
                                type_name + str(x['year'])[2:] + str(0) + str(x['month']), axis=1)


    df['now_code_index'] = df.apply(lambda x: eval(x['可交易合约集合']).index(x['now_code']) if x['now_code'] in eval(x['可交易合约集合'])
                                    else np.nan,axis=1)
    df['now_code_OI'] = df.apply(lambda x: 0 if np.isnan(x['now_code_index'])
            else eval(x['open_interest'])[int(x['now_code_index'])],axis=1)
    df = df.set_index('日期')

    return df['now_code_OI']




def get_member_OI(type_name):
    df_long = fetch_data(type_name,'会员多头持仓')
    df_short = fetch_data(type_name,'会员空头持仓')

    long_start_date = df_long['日期'][0]
    short_start_date = df_short['日期'][0]
    start_date = max(long_start_date,short_start_date)
    df_long = df_long[df_long['日期']>=start_date]
    df_short = df_short[df_short['日期']>=start_date]

    df_long = df_long[df_long['排名']<=20]
    df_short = df_short[df_short['排名']<=20]
    long_oi = []
    short_oi = []


    for tradingday,temp in df_long.groupby("日期"):
        long_oi.append(temp['持仓量'].sum())

    for tradingday,temp in df_short.groupby("日期"):
        short_oi.append(temp['持仓量'].sum())


    df = {}
    df['多头持仓'] = long_oi
    df['空头持仓'] = short_oi
    df = pd.DataFrame(df,index=df_long.copy().drop_duplicates('日期')['日期'])
    df['净多头持仓'] = df['多头持仓']-df['空头持仓']
    ## 计算不同因子
    df['净多持仓率'] = df.apply(lambda x:(x['多头持仓']-x['空头持仓'])/(x['多头持仓']+x['空头持仓']),axis=1)
    df['多头持仓率'] = df.apply(lambda x: x['多头持仓'] / (x['多头持仓'] + x['空头持仓']), axis=1)
    df['空头持仓率'] = df.apply(lambda x: x['空头持仓'] / (x['多头持仓'] + x['空头持仓']), axis=1)
    # return df[['净多持仓率']],df[['多头持仓率']],df[['空头持仓率']]
    return df[['多头持仓']],df[['空头持仓']],df[['净多头持仓']],df[['净多持仓率']]


def get_oi(type_name):
    daily_data_folder = "C:/TBWork/DailyData/"
    tradingday = processing_data('P', '指数连续')
    tradingday.columns = ['close']
    file = pd.read_csv(daily_data_folder + type_name + ".csv", index_col=0)
    file.index = pd.to_datetime(file.index)
    file = pd.concat([tradingday, file], axis=1)
    file.fillna(inplace=True, method='ffill')
    file.dropna(inplace=True)
    file = file.iloc[:, 1:]
    res = file[['OpenInt']]
    res.columns = [type_name]
    return res


def catch_moment(df,type_name):

    df = df.reset_index()
    df['year'] = df['日期'].apply(lambda x:x.year)
    df['month'] = df['日期'].apply(lambda x:x.month)
    df = df.set_index('日期')

    adjust_df=pd.DataFrame()
    for id,temp_df in df.groupby(["year",'month']):
        def f(x):
            try:
                if x[0]!=0 and x[1]==0:
                    return 1
                else:
                    return 0
            except:
                m=2
        if type_name == 'SC':
            temp_df.loc[temp_df.index[-1],'catch_flag']=1
        else:
            temp_df['catch_flag'] = temp_df['now_code_OI'].rolling(2).apply(lambda x: f(x))
            temp_df['catch_flag'] = temp_df['catch_flag'].shift(-1)
        temp_df['new_OI'] = temp_df.apply(lambda x: x['now_code_OI'] if x['catch_flag']==1 else np.nan,axis=1)
        temp_df['new_OI'] = temp_df['new_OI'].fillna(method='bfill')
        adjust_df = pd.concat([adjust_df,temp_df],axis=0)

    return adjust_df[['new_OI']]

def catch_true_moment(df,type_name):

    df = df.reset_index()
    df['year'] = df['日期'].apply(lambda x:x.year)
    df['month'] = df['日期'].apply(lambda x:x.month)
    df = df.set_index('日期')

    adjust_df=pd.DataFrame()
    for id,temp_df in df.groupby(["year",'month']):
        def f(x):
            try:
                if x[0]!=0 and x[1]==0:
                    return 1
                else:
                    return 0
            except:
                m=2
        if type_name == 'SC':
            temp_df.loc[temp_df.index[-1],'catch_flag']=1
        else:
            temp_df['catch_flag'] = temp_df['now_code_OI'].rolling(2).apply(lambda x: f(x))
            temp_df['catch_flag'] = temp_df['catch_flag'].shift(-1)
        temp_df['new_OI'] = temp_df.apply(lambda x: x['now_code_OI'] if x['catch_flag']==1 else np.nan,axis=1)
        temp_df['new_OI'] = temp_df['new_OI'].fillna(method='bfill')
        adjust_df = pd.concat([adjust_df,temp_df],axis=0)

    return adjust_df[['new_OI']]




def cangdan_to_share(df,type_name):

    if type_name in ['CF']:
        return df*8
    elif type_name in ['CU','AL','NI','ZN','RB','HC','SC','BU','RU','SP']:
        return df/SymbolMultiplier[type_name]
    else:
        return df


def xushibi(type_name,mode):
    """
    :param type_name: 指定品种
    :param mode: mode 1 ： 仓单值/主力持仓量的一半  2. 仓单值/会员多头持仓
    :return:
    """
    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()
    cangdan_share = cangdan_to_share(cangdan,type_name)

    if mode==1:
        information = processing_data(type_name, '期限结构')
        information['主力pos'] = information.apply(lambda x: eval(x['可交易合约集合']).index(x['主力合约code']),axis=1)
        information['持仓量'] = information.apply(lambda x: eval(x['open_interest'])[x['主力pos']],axis=1)
        OI_info = information[['持仓量']]
        temp = pd.concat([cangdan_share,OI_info],axis=1)
        temp['实虚比'] = temp['仓单']/(temp['持仓量']*0.5)
        res = temp[['实虚比']]
        res.columns = [type_name]
    elif mode==2:
        df_long = fetch_data(type_name, '会员多头持仓')
        df_long = df_long[df_long['排名'] <= 20]
        long_oi = []
        for tradingday, temp in df_long.groupby("日期"):
            long_oi.append(temp['持仓量'].sum())
        df = {}
        df['多头持仓'] = long_oi
        df = pd.DataFrame(df, index=df_long.copy().drop_duplicates('日期')['日期'])
        temp = pd.concat([cangdan_share, df['多头持仓']], axis=1)
        temp['实虚比'] = temp['仓单']/temp['多头持仓']
        res = temp[['实虚比']]
        res.columns = [type_name]

    return res.fillna(method='ffill')

def xushibi_tongbi(type_name,mode):
    
    xsb = xushibi(type_name,mode=mode)
    xsb.columns = ['实虚比']
    xsb['分母'] = xsb['实虚比'].rolling(307).apply(lambda x: x[:126][x != 0].mean())
    xsb['分子'] = xsb['实虚比']
    xsb['同差'] = xsb['分子'] - xsb['分母']

    def non_zero_std(x):
        diff_x = x.diff()
        non_zero_x = x[(diff_x != 0) | (x != 0)]

        if len(non_zero_x) / len(x) <= 0.5:
            return np.nan
        else:
            return x.std()

    xsb['基准值'] = xsb['实虚比'].rolling(244).apply(lambda x: non_zero_std(x))
    # xsb['基准值'] = xsb['实虚比'].rolling(244).std()
    xsb['同比'] = xsb['同差'] / xsb['基准值']
    res = -xsb[['同比']]
    res.columns = [type_name]
    return res
    



def cangdan_seasonal_sig(type_name):

    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date = end_date.strftime("%Y-%m-%d")
    date_list = pd.date_range("2010-01-01", end_date, freq='D')
    date_list = pd.DataFrame([0] * len(date_list), index=date_list)
    date_list.index = pd.to_datetime(date_list.index)

    cangdan = processing_data(type_name,'仓单')
    data = pd.concat([date_list, cangdan], axis=1)
    data.fillna(inplace=True, method='ffill')
    data = data.iloc[:,1].to_frame().dropna()
    data = data.reset_index()

    if type_name=='SA':
        data_season_sig = cangdan.applymap(lambda x: 0)
    else:
        data['year'] = data['index'].apply(lambda x:x.year)
        data['month-day'] = data['index'].apply(lambda x: x.strftime("%m/%d"))
        data_season = data.pivot(values = '仓单', columns = 'month-day', index='year')

        # data_season_rank = data_season.expanding(4).rank()
        # data_season_count = data_season.expanding(4).count()
        # data_season_ratio = data_season_rank/data_season_count
        # data_season_sig = data_season_ratio.applymap(lambda x: 1 if x<=0.25 else -1 if x>=0.75 else 0)
        # data_season_sig = data_season_sig.stack().reset_index()


        data_season_lower = data_season.expanding(4).quantile(0.25)
        data_season_upper = data_season.expanding(4).quantile(0.75)
        data_season_sig_0 = data_season - data_season_lower
        data_season_sig_1 = data_season - data_season_upper
        data_season_sig_0 = data_season_sig_0.applymap(lambda x: 1 if x<0 else 0)
        data_season_sig_1 = data_season_sig_1.applymap(lambda x: -1 if x>0 else 0)
        data_season_sig = data_season_sig_0+data_season_sig_1

        # data_season_sig = data_season-data_season.expanding(4).apply(lambda x: 1 if x[-1]<=x.quantile(0.25) else -1 if x[-1]>=x.quantile(0.75) else 0 )
        data_season_sig = data_season_sig.stack().reset_index()


        data_season_sig['date'] = data_season_sig.apply(lambda x: pd.to_datetime(str(x['year'])+"/"+x['month-day'],errors='coerce'),axis=1)
        data_season_sig = data_season_sig.iloc[:,2:4]
        data_season_sig = data_season_sig.set_index('date')

    data_season_sig.columns = [type_name]
    data_season_sig = data_season_sig.loc[data_season_sig.index.notnull()]

    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    cangdan = pd.concat([Tradingday, data_season_sig], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()

    return cangdan



def OI_water_signal(type_name):
    rate = member_OI(type_name,20,20,3)
    rate['level'] = rate[type_name].rolling(244).apply(lambda x:water(x))
    rate['signal'] = rate['level'].apply(lambda x: 1 if x>=0.9 else -1 if x<=0.1 else 0)

    res = rate['signal'].to_frame()
    res.columns = [type_name]

    return res



def candan_no_change_sig(type_name,N=10,threshold=0.1):
    Tradingday = processing_data("P", '指数连续')
    Tradingday.columns = ['close']
    cangdan = processing_data(type_name, '仓单')
    cangdan = pd.concat([Tradingday, cangdan], axis=1)
    cangdan = shift_holiday(cangdan)
    cangdan = cangdan.iloc[:, 1].to_frame().dropna()

    # def non_zero_std(x):
    #     diff_x = x.diff()
    #     non_zero_x = x[(diff_x != 0) | (x != 0)]
    #
    #     if len(non_zero_x) / len(x) <= 0.5:
    #         return np.nan
    #     else:
    #         return non_zero_x.std()


    cangdan['变动'] = cangdan['仓单'].diff(1)
    cangdan['变动累加'] = abs(cangdan['变动']).rolling(N-1).sum()
    cangdan['振幅比例'] = cangdan['变动累加']/cangdan['仓单'].rolling(244).std()
    cangdan['无变动信号'] = cangdan['振幅比例'].apply(lambda x: 1 if x<=threshold else 0)



    res = cangdan['无变动信号'].to_frame()
    res.columns = [type_name]

    return res














if __name__ == '__main__':

    # candan_no_change_sig('SC',N=20,threshold=0.1)

    heise = ['I', 'J', 'RB', 'HC']
    youse = ['CU', 'AL', 'ZN', 'NI']
    huagong = ['MA', 'PP', 'L', 'TA', 'BU', 'EG', 'RU', 'SP', 'UR', 'SA', 'FG', 'V','SC']
    agriculture = ['SR', 'CF', 'C', 'CS', 'M', 'OI', 'A', 'RM', 'Y', 'P']
    trading_symbols = heise + youse + huagong + agriculture



    total_signal = pd.DataFrame()
    for k in trading_symbols:
        temp = Jicha_Mom(k,mode=2,N=5)
        total_signal = pd.concat([total_signal,temp],axis=1)
    total_signal.to_excel(r"C:\TBWork\仓单数据\yuecha_ratio_raw_mom.xlsx")