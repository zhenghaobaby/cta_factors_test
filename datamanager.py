# -- coding: utf-8 --
"""
 @time : 2023/6/29
 @file : datamanager.py
 @author : zhenghao
 @software: PyCharm
"""
import glob

from rqdatac import *
import rqdatac
import pandas as pd
import os
import datetime
import pickle

rqdatac.init()


nowtime = datetime.datetime.now().strftime("%Y-%m-%d")
tradingday = pd.read_csv("all_tradeday.csv",index_col=0,parse_dates=[0])


def adjust_tradingday(date, shift=1):
    """
    传入的date需要时pd.datetime后的格式，Timestamp
    """
    pos = list(tradingday.index).index(date)
    aj_pos = pos + shift
    return list(tradingday.index)[aj_pos]

def get_dominant_data_1m(code_info,idx):
    try:
        df = get_price(code_info,start_date=datetime.datetime(year=idx.year,month=idx.month,day=idx.day,hour=8,minute=0,second=0),
                       end_date=datetime.datetime(year=idx.year, month=idx.month, day=idx.day, hour=15, minute=15,second=0),
                       frequency='1m',expect_df=True,
                       # fields=['open','close','high','low','total_turnover','volume','open_interest','prev_close']
                     )
        df = df.reset_index()
        df = df[['order_book_id','datetime','trading_date','open','close','high','low','total_turnover','volume','open_interest']]
    except:
        with open("data_cache/error.txt",'a+') as file:
            file.write(f"{str(idx)}_{str(code_info)} has problem!!!\n")
        df = None

    return df


def get_data_D(symbols,start="2014-01-01",end='2023-08-29'):

    code_list = []
    for sym in symbols:
        code_list.append(sym+"889")

    df = get_price(code_list,start_date=start,end_date=end,frequency='1d',expect_df=True)
    df.to_csv("data_cache/data_day.csv")



def get_dominant_code(symbols,start ="2023-01-03",end="2023-08-29"):
    dominant_trading_code = {}
    if os.path.exists("data_cache/dominant_code.pickle"):
        with open("data_cache/dominant_code.pickle", 'rb') as f:
            dominant_trading_code = pickle.load(f)
            # start = list(dominant_trading_code.keys())[-1]
        f.close()

    pre_trading_code = {}
    if os.path.exists("data_cache/trading_code.pickle"):
        with open("data_cache/trading_code.pickle", 'rb') as f:
            pre_trading_code = pickle.load(f)
        f.close()


    trading_code_dict = {}
    dominant_code_dict = {}
    #当起点是换月的一天，在持续更新时会出现问题，因此需要往前挪一天，让其可以比
    start = pd.to_datetime(start)
    start = adjust_tradingday(start,shift=-1)
    Date_list = tradingday.loc[start:pd.to_datetime(end)]

    pre_dominant_code_list = []
    for year in [2023]:
        data_dict = {}
        if os.path.exists(f"data_cache/data_{year}.pickle"):
            with open(f"data_cache/data_{year}.pickle", 'rb') as f:
                pre_data_dict = pickle.load(f)
                f.close()
        else:
            pre_data_dict = {}

        for date in list(Date_list.index):
            if date.year!=year:
                continue
            trading_code_list = []
            dominant_code_list = []
            for sym in symbols:
                try:
                    dominant_code = futures.get_dominant(underlying_symbol=sym,start_date=date,end_date=date,rank=1)
                    sec_dominant_code = futures.get_dominant(underlying_symbol=sym,start_date=date,end_date=date,rank=2)
                    code_info = pd.concat([dominant_code, sec_dominant_code], axis=1)
                    code_info.columns = ['主力', '次主力']
                    trading_code_list+=(list(code_info.values[0]))
                    dominant_code_list+=(list(dominant_code.values))
                except:
                    with open("data_cache/error.txt", 'a+') as file:
                        file.write(f"{sym} has problem!!!\n")

            if len(pre_dominant_code_list) == 0:
                pre_dominant_code_list = dominant_code_list
            else:
                pre_codes = list(set(pre_dominant_code_list)-set(dominant_code_list))
                if len(pre_codes)==0:
                    pre_dominant_code_list = dominant_code_list
                else:
                    print(f"{pre_codes}换月！！！")
                    trading_code_list+=pre_codes
                    pre_dominant_code_list = dominant_code_list

            data_dict[date] = get_dominant_data_1m(code_info=trading_code_list, idx=date)
            trading_code_dict[date] = trading_code_list
            dominant_code_dict[date] = dominant_code_list
            print(date, dominant_code_list)


        # 更新主力合约量价信息
        pre_data_dict.update(data_dict)
        with open(f"data_cache/data_{year}.pickle", 'wb') as f:
            pickle.dump(pre_data_dict, f)


    # 更新主力合约信息数据
    dominant_trading_code.update(dominant_code_dict)
    pre_trading_code.update(trading_code_dict)
    with open("data_cache/dominant_code.pickle","wb") as f:
        pickle.dump(dominant_trading_code,f)
    with open("data_cache/trading_code.pickle","wb") as f:
        pickle.dump(pre_trading_code,f)



def convert_to_ohlc(year):

    file_list = glob.glob(f"data_cache/data_{year}*.pickle")
    for file in file_list:
        year = os.path.split(file)[1]
        data = {}
        with open(file,"rb") as f:
            df = pickle.load(f)
            for key,val in df.items():
                print(key)
                try:
                    temp = {
                        'open': val.pivot(values='open',index='datetime',columns ='order_book_id'),
                        'high': val.pivot(values='high',index='datetime',columns ='order_book_id'),
                        'low': val.pivot(values='low',index='datetime',columns ='order_book_id'),
                        'close': val.pivot(values='close',index='datetime',columns ='order_book_id'),
                        'total_turnover':val.pivot(values='total_turnover',index='datetime',columns ='order_book_id'),
                        'volume':val.pivot(values='volume',index='datetime',columns ='order_book_id'),
                        'open_interest':val.pivot(values='open_interest',index='datetime',columns ='order_book_id'),
                    }
                    data[key] = temp
                except:
                    print(f"{key} has problem!!!")
                    continue

        with open('data_ohlc_cache/'+year,'wb') as f:
            pickle.dump(data,f)













if __name__ == '__main__':



    # heise = ['I', 'J', 'RB', 'HC']
    # youse = ['CU', 'AL', 'ZN', 'NI']
    # huagong = ['MA', 'PP', 'L', 'TA', 'BU', 'EG', 'RU', 'SP', 'UR', 'SA', 'FG', 'V', 'SC', 'NR']
    # agriculture = ['SR', 'CF', 'C', 'CS', 'M', 'OI', 'A', 'RM', 'Y', 'P']
    # total = heise+youse+huagong+agriculture
    # get_dominant_code(symbols= total,start="2023-08-01",end='2023-09-05')
    convert_to_ohlc(year=2023)

