"""
该程序用于生成不同库存的单品种回测结果或者单一品种的多策略结果，也可以作为每天自动生成信号和回测结果的蓝本
"""


import os
import matplotlib.pyplot as plt
from FBT import LBY_BackTester_QH
import datetime
import numpy as np
import re

import pandas as pd
import configparser


class myconf(configparser.ConfigParser):
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)

        # 这里重写了optionxform方法，直接返回选项名

    def optionxform(self, optionstr):
        return optionstr


class FBT_type_balance(LBY_BackTester_QH):
    def set_para(self, HS300, pos, type_name, adjust_days):
        self.HS300 = HS300
        self.signal = pos
        self.type_name = type_name
        self.Days = adjust_days


    def LBY_PrepareData(self):
        self.pre_sig = np.nan

        num_vars = self.Days
        for i in range(num_vars):
            setattr(self, f"var_score_{i}", {})
            for symbol in self.parameter_symbol_type_list:
                getattr(self, f"var_score_{i}")[symbol] = 0


    def LBY_AfterTradingDayEnd(self, current_dt):
        my_time = datetime.datetime(current_dt.year, current_dt.month, current_dt.day, 15, 0, 0)
        signal_bar = self.signal[self.signal.index == pd.to_datetime(str(current_dt)[:10])]

        # signal_bar.dropna(inplace=True, axis=1)

        if len(signal_bar) == 0:
            for symbol in self.parameter_symbol_type_list:
                self.var_score[symbol] = 0
        else:

            dominant_bar = pd.DataFrame(self.DominantContractCodeDic, index=['dominant_code'])
            total = pd.concat([signal_bar, dominant_bar], axis=0)
            total.dropna(inplace=True, axis=1)
            signal_bar = total.iloc[[0]]



            long_number = abs(signal_bar[signal_bar>0].dropna(axis=1).sum(axis=1)[0])
            short_numer = abs(signal_bar[signal_bar<0].dropna(axis=1).sum(axis=1)[0])


            for i in range(self.Days):
                if self.TradingDaysCounter % self.Days == i:
                    for symbol in signal_bar.columns:
                        if signal_bar[symbol][0] == 0:
                            getattr(self, f"var_score_{i}")[symbol] = 0

                        elif signal_bar[symbol][0]>0:
                            code = self.DominantContractCodeDic[symbol]
                            price = self.intraday_1m_bars.loc[my_time, code]
                            if getattr(self, f"var_score_{i}")[symbol] > 0 and self.pre_sig[symbol][0] == signal_bar[symbol][0]:
                                continue
                            else:
                                getattr(self, f"var_score_{i}")[symbol] = int(signal_bar[symbol][0] * self.BackTestStartCash / (
                                                2*long_number * price * self.SymbolMultiplier[code[:-4]]))
                        else:
                            code = self.DominantContractCodeDic[symbol]
                            price = self.intraday_1m_bars.loc[my_time, code]
                            if getattr(self, f"var_score_{i}")[symbol] < 0 and self.pre_sig[symbol][0] == signal_bar[symbol][0]:
                                continue
                            else:
                                getattr(self, f"var_score_{i}")[symbol] = int(
                                    signal_bar[symbol][0] * self.BackTestStartCash / (
                                            2*short_numer * price * self.SymbolMultiplier[code[:-4]]))


                else:
                    continue

            self.pre_sig = self.signal[self.signal.index == pd.to_datetime(str(current_dt)[:10])]
            for symbol in signal_bar.columns:
                sum_var_score = sum(getattr(self, f"var_score_{n}")[symbol] for n in range(self.Days))
                self.var_score[symbol] = int(sum_var_score / self.Days)




        self.LBY_GenerateOrders(current_dt)


    def Trading_Analysis(self):
        # file_path = f"backtest_res/{self.model}/{self.group_info}/"
        #
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)

        # # 交易记录
        # order_log = self.AllOrderRecordList_specific
        # order_log = pd.DataFrame(order_log)
        # order_log.to_excel(file_path + "/order_log.xlsx")
        start_time = pd.to_datetime(self.BackTestBeginDate)

        # 净值计算
        PNL = self.DailyTotalPortfolioCapital
        PNL = pd.DataFrame(PNL, index=['净值'])
        PNL = PNL.T
        PNL = PNL[PNL.index >= start_time]

        # 手续费计算
        # Trade_fee = self.DailyTotalTradingFees
        # Trade_fee = pd.DataFrame(Trade_fee, index=['手续费'])
        # Trade_fee = Trade_fee.T
        # Trade_fee = Trade_fee[Trade_fee.index >= start_time]
        #
        # PNL = pd.concat([PNL, Trade_fee], axis=1)
        PNL['净值'] = PNL['净值'].apply(lambda x: x / self.BackTestStartCash)
        PNL['回撤'] = PNL['净值'] - PNL['净值'].cummax()
        # fig = plt.figure(figsize=(32, 20))
        # ax1 = plt.subplot(111)
        # PNL['净值'].plot(color='b')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.legend(loc='upper left', fontsize=20)
        # ax2 = plt.twinx()
        # PNL['回撤'].plot(color='y', alpha=0.5)
        # plt.yticks(fontsize=20)
        # plt.legend(fontsize=20)
        # plt.savefig(file_path + "/PNL.png")

        # 交易结果的指标计算
        # during_date = (pd.to_datetime(self.BackTestEndDate) - start_time).days
        # during_year = during_date / 365.0
        PNL['盈亏'] = PNL['净值'].diff(1)
        res = {}
        res['收益%'] = (PNL['净值'][-1] - 1) * 100
        res['年化收益%'] = PNL['盈亏'].mean() * 244 * 100
        res['年化波动%'] = PNL['盈亏'].std() * np.sqrt(244) * 100
        res['sharpe'] = res['年化收益%'] / res['年化波动%']
        res['最大回撤%'] = abs(PNL['回撤'].min()) * 100



        if res['最大回撤%'] == 0:
            res['收益回撤比'] = 0
        else:
            res['收益回撤比'] = res['年化收益%'] / res['最大回撤%']
        res = pd.DataFrame(res, index=['回测统计'])


        return res,PNL
        # res.to_excel(file_path + "/result_analysis.xlsx")
        # PNL.to_excel(file_path + "/PNL.xlsx")




def simple_backtest(pos,tradingday =None,adjust_days=None,trading_fee=False):


    StartCash = 1E8 # 初始账户资金

    ##分批下单生成列表
    time_list_0 = list(pd.date_range("9:01", "09:30", freq="5min"))
    time_list_1 = list(pd.date_range("10:31", "11:26", freq="5min"))

    for i in range(len(time_list_0)):
        time_list_0[i] = str(time_list_0[i])[11:16]
    for i in range(len(time_list_1)):
        time_list_1[i] = str(time_list_1[i])[11:16]
    time_list = time_list_0+time_list_1

    RebalanceTimeList = time_list  # 时间戳为列表里时间的K线完成时触发调仓逻辑，每个时间格式hh:mm,比如'09:15'
    BeginDate = list(pos.index)[0]
    EndDate = list(pos.index)[-1]

    type_list = list(pos.columns)
    backtester = FBT_type_balance(StartCash, RebalanceTimeList, BeginDate, EndDate, type_list)
    backtester.set_para(tradingday,pos,type_list,adjust_days)


    if not trading_fee :
        backtester.BackTestFeeRatio = 0
        backtester.BackTestSlipTicks = 0
    stat_res,pnl = backtester.LBY_DoBackTest()


    return stat_res,pnl





if __name__ == '__main__':
    pass



















