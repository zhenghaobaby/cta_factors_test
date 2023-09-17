# -- coding: utf-8 --
"""
 @time : 2023/7/10
 @file : data_loader.py
 @author : zhenghao
 @software: PyCharm

不同进程之间的缓存容器

"""

from cachetools import cached, TTLCache
import datetime
import pickle
import pandas as pd
import rqdatac
from cachetools import TTLCache
from pathlib import Path

root = "C:/Project/截面因子测试/"

class FIFOTTLCache(TTLCache):
    def __init__(self, maxsize, ttl):
        super().__init__(maxsize, ttl)
        self.queue = []

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.queue.append(key)
        self._prune()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.queue.remove(key)
        self.queue.append(key)
        return value

    def __delitem__(self, key):
        super().__delitem__(key)
        self.queue.remove(key)

    def _prune(self):
        if len(self.queue) > self.maxsize:
            key = self.queue.pop(0)
            super().__delitem__(key)


class DataFetcher:
    script_path = Path(__file__)
    tradingday = pd.read_csv(script_path.parent / "all_tradeday.csv", index_col=0, parse_dates=[0])

    def __init__(self):
        # self.price_data = {
        #     2023: pickle.load(open("data_cache/data_2023.pickle", "rb")),
        #     2022: pickle.load(open("data_cache/data_2022.pickle", "rb")),
        #     2021: pickle.load(open("data_cache/data_2021.pickle", "rb")),
        #     2020: pickle.load(open("data_cache/data_2020.pickle", "rb")),
        #     2019: pickle.load(open("data_cache/data_2019.pickle", "rb")),
        #     2018: pickle.load(open("data_cache/data_2018.pickle", "rb")),
        #     2017: pickle.load(open("data_cache/data_2017.pickle", "rb")),
        #     2016: pickle.load(open("data_cache/data_2016.pickle", "rb")),
        #     2015: pickle.load(open("data_cache/data_2015.pickle", "rb")),
        # }

        self.price_ohlc_data = {
            2023: pickle.load(open("data_ohlc_cache/data_2023.pickle", "rb")),
            2022: pickle.load(open("data_ohlc_cache/data_2022.pickle", "rb")),
            2021: pickle.load(open("data_ohlc_cache/data_2021.pickle", "rb")),
            2020: pickle.load(open("data_ohlc_cache/data_2020.pickle", "rb")),
            2019: pickle.load(open("data_ohlc_cache/data_2019.pickle", "rb")),
            2018: pickle.load(open("data_ohlc_cache/data_2018.pickle", "rb")),
            2017: pickle.load(open("data_ohlc_cache/data_2017.pickle", "rb")),
            2016: pickle.load(open("data_ohlc_cache/data_2016.pickle", "rb")),
            2015: pickle.load(open("data_ohlc_cache/data_2015.pickle", "rb")),
        }

        self.dominant_code = pickle.load(open("data_cache/dominant_code.pickle","rb"))
        self.starting_trade_date = {'TA': "2010-01-04", 'PP': "2015-02-28", 'L': "2007-07-31",
                               'CU': "2010-01-04", 'ZN': "2010-01-04", 'NI': "2015-03-27",
                               'AL': "2010-01-04", 'FG': "2012-12-03", 'RU': "2010-01-04",
                               'V': "2010-01-04", 'BU': "2013-10-09", 'MA': "2011-10-28",
                               'HC': "2014-03-21", 'RB': "2010-01-04", 'I': "2013-10-18",
                               'Y': "2010-01-04", 'P': "2010-01-04", 'EG': "2018-12-10",
                               'SC': "2018-03-26", 'SA': "2020-12-06", 'EB': "2020-09-26",
                               'OI': "2010-01-04", 'J': "2011-05-03", 'JM': "2014-03-28",
                               'M': "2010-01-04", 'UR': "2019-08-09", 'AP': "2017-12-28",
                               'CF': "2010-01-04", 'SP': "2018-11-27", 'SR': "2010-01-01",
                               'FU': "2018-07-16", 'C': '2010-01-01', 'CS': "2015-12-22",
                               'A': "2010-01-01", 'RM': "2012-12-28", 'NR': "2019-08-12",}
        self.cangdan_data = pd.read_excel(r"C:\TBWork\仓单数据\cangdan.xlsx",index_col=0,parse_dates=[0])
        self.member_oi_data = pd.read_excel(r"C:\TBWork\仓单数据\净多持仓.xlsx",index_col=0,parse_dates=[0])
        self.yuecha_ratio = pd.read_excel(r"C:\TBWork\仓单数据\yuecha_ratio.xlsx",index_col=0,parse_dates=[0])
        self.yuecha_ratio_mom = pd.read_excel(r"C:\TBWork\仓单数据\yuecha_ratio_mom.xlsx",index_col=0,parse_dates=[0])

        self.yuecha_ratio_raw = pd.read_excel(r"C:\TBWork\仓单数据\yuecha_ratio_raw.xlsx",index_col=0,parse_dates=[0])
        self.yuecha_ratio_raw_mom = pd.read_excel(r"C:\TBWork\仓单数据\yuecha_ratio_raw_mom.xlsx",index_col=0,parse_dates=[0])

        self.kucun_tongbi = pd.read_csv(root+"kucun_factor/kucun_tongbi.csv",index_col=0,parse_dates=[0])
        self.kucun_huanbi_month = pd.read_csv(root+"kucun_factor/kucun_huanbi_month.csv",index_col=0,parse_dates=[0])
        self.kucun_huanbi_week = pd.read_csv(root+"kucun_factor/kucun_huanbi_week.csv",index_col=0,parse_dates=[0])
        self.kucun_shuiwei = pd.read_csv(root+"kucun_factor/kucun_shuiwei.csv",index_col=0,parse_dates=[0])
        self.kucun_huanbi_diff_week = pd.read_csv(root+"kucun_factor/kucun_huanbi_diff_week.csv",index_col=0,parse_dates=[0])
        self.kucun_huanbi_diff_month = pd.read_csv(root+"kucun_factor/kucun_huanbi_diff_month.csv", index_col=0, parse_dates=[0])





    def get_trading_days(self,date,N):
        """
        获取回溯天数的交易日列表
        """
        now_idx = list(self.tradingday.index).index(pd.to_datetime(date))
        pre_idx = now_idx - N + 1
        d = list(self.tradingday.index)[pre_idx:now_idx+1] #包括当天
        return d


    def get_daily_data(self,date,N):
        df = self.Daily_data.copy()
        df = df.reset_index()

        start_date = self.get_trading_days(date,N)[0]
        return df[ (df['date']>=start_date.strftime("%Y-%m-%d"))&(df['date']<=date.strftime("%Y-%m-%d"))]


    # @cached(cache=FIFOTTLCache(maxsize=512, ttl=3600))
    def get_price_data(self,info):
        """
        返回指定date的数据
        """
        info =eval(info)
        date = info['date']
        symbols = info['symbols']

        # 数据库获取，现在比较慢
        # df = fetch_data_on_day(db_name='basic_data', table_name='price', datetime=date)
        # 本地获取
        data = self.price_data[pd.to_datetime(date).year]
        df = data[pd.to_datetime(date)]

        if symbols == None: #采用当时主力合约日内合成数据时，不用考虑下面数据没有的情况
            return df
        else:
            ##检查如何没有足够的数据（出现新主力不是前面的次主力的时候），将新数据添加进数据
            deletion_codes = list(set(symbols) - set(df['order_book_id'].tolist()))
            if len(deletion_codes) == 0:
                pass
            else:
                print(str(date) + ":" + str(deletion_codes) + "没有缓存！！！\n")
                rqdatac.init()
                date = pd.to_datetime(date)
                append_df = rqdatac.get_price(deletion_codes,
                                              start_date=datetime.datetime(year=date.year, month=date.month, day=date.day,
                                                                           hour=8, minute=0, second=0),
                                              end_date=datetime.datetime(year=date.year, month=date.month, day=date.day,
                                                                         hour=15, minute=15, second=0),
                                              frequency='1m', expect_df=True, )
                append_df = append_df.reset_index()
                append_df = append_df[['order_book_id','datetime','trading_date','open','close','high','low','total_turnover','volume','open_interest']]
                # insert(append_df,db_name='basic_data',table_name='price')
                df = pd.concat([df, append_df], axis=0)

                #文件保存
                update_df = {}
                update_df[pd.to_datetime(date)] = df
                self.price_data[pd.to_datetime(date).year].update(update_df)


        return df


    def get_price_ohlc_data(self,date):
        data = self.price_ohlc_data[pd.to_datetime(date).year]
        df = data[pd.to_datetime(date)]

        return df



    def get_automl_data(self,date):
        return self.automl_pred.loc[date:date]

    # @cached(cache=FIFOTTLCache(maxsize=512, ttl=3600))
    def get_cangdan_data(self,date):
        return self.cangdan_data.loc[:date]


    def get_member_oi_data(self,date):
        return self.member_oi_data.loc[date:date]


    def get_kucun_data(self,date,method='tongbi'):
        if method=='tongbi':
            return self.kucun_tongbi.loc[date:date]
        elif method=='huanbi_week':
            return self.kucun_huanbi_week.loc[date:date]
        elif method=='huanbi_month':
            return self.kucun_huanbi_month.loc[date:date]
        elif method=='shuiwei':
            return self.kucun_shuiwei.loc[date:date]
        elif method=='huanbi_diff_week':
            return self.kucun_huanbi_diff_week.loc[date:date]
        elif method=='huanbi_diff_month':
            return self.kucun_huanbi_diff_month.loc[date:date]


    # @cached(cache=FIFOTTLCache(maxsize=512, ttl=3600))
    def get_yuecha_ratio(self,date,method='raw'):
        if method=='raw':
            return self.yuecha_ratio_raw.loc[:date]
        elif method == 'raw_mom':
            return self.yuecha_ratio_raw_mom.loc[:date]

        elif method=='yuanqi':
            return self.yuecha_ratio.loc[:date]
        elif method == 'yuanqi_mom':
            return self.yuecha_ratio_mom.loc[:]


    def dumps_price_data(self):
        for key,val in self.price_data.items():
            with open(f"data_cache/data_{key}.pickle", "wb") as f:
                pickle.dump(val, f)

    # @cached(cache=FIFOTTLCache(maxsize=512,ttl=3600))
    def get_dominant_code(self,date):
        #数据库获取
        # df = fetch_data_on_time(db_name='basic_data', table_name='code', datetime=date)
        # df = df['order_book_id'].tolist()

        #本地获取
        df = self.dominant_code[pd.to_datetime(date)]
        code_list = []
        for k in df:
            if pd.to_datetime(date)<pd.to_datetime(self.starting_trade_date[k[:-4]])+datetime.timedelta(days=120):
                continue
            else:
                code_list.append(k)

        return code_list



if __name__ == '__main__':
    a = DataFetcher()
    a.get_daily_data("2023-08-01",N=20)
