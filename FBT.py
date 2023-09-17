# 导入函数库
import pickle
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import os
import warnings
warnings.filterwarnings('ignore')


class LBY_PositionInfo:
    contract_code = ''
    symbol_type = ''
    avg_entry_price = 0
    current_contracts = 0  # 由于持仓被分成多空两类，那么这里的头寸就不带正负号了，而是绝对值
    last_trading_day_close_price = 0

    def __init__(self, ContractCode):
        self.contract_code = ContractCode
        self.symbol_type = ContractCode[:-4]
        self.avg_entry_price = 0
        self.current_contracts = 0  # 由于持仓被分成多空两类，那么这里的头寸就不带正负号了，而是绝对值
        self.last_trading_day_close_price = 0

    def PrintInfo(self):
        print(f"PositionDetail: {self.contract_code}, {self.current_contracts}, {self.avg_entry_price}")


class LBY_BackTester_QH:
    ###### 本回测框架只适用于下午收盘后判断调仓逻辑，下一交易日上午和下午指定时间点调仓 ######
    ###### 注意：聚宽K线的时间戳是K线结束的时间，而非K线生成时刻 ######

    ''' 策略-参数区 '''
    # parameter_symbol_type_list = ['RB', 'HC', 'I', 'J', 'JM', 'ZC', 'FG', 'MA', 'TA', 'L', 'PP', 'BU', 'RU', 'CU', 'AL', 'ZN', 'PB',
    #                 'NI', 'SN', 'M', 'Y', 'RM', 'OI', 'P', 'A', 'C', 'CS', 'CF', 'SR', 'JD','EG','V']
    parameter_length = {'RB': 3, 'CU': 4}
    ''' 策略-变量区 '''
    basis_price_df = {}
    delivery_date_df = {}
    # var_score = {}

    def __init__(self, start_cash=10000000, rebalance_times=['9:01'], backtest_begin_date='2015-01-01',
                 backtest_end_date='2015-12-31',parameter_list = []):
        # print("本回测框架只适用于下午收盘后判断调仓逻辑，下一交易日上午和下午指定时间点调仓\n")
        # print("注意：聚宽K线的时间戳是K线结束的时间，而非K线生成时刻\n")
        self.BackTestStartCash = start_cash
        self.BackTestRebalanceTimeList = rebalance_times
        self.BackTestBeginDate = backtest_begin_date
        self.BackTestEndDate = backtest_end_date
        self.TotalPortfolioCapital = start_cash
        self.parameter_symbol_type_list = parameter_list
        self.AllOrderRecordList_specific = {
            'time': [],
            'buy/sell': [],
            'contract_code': [],
            'order_lots': [],
            'order_price': [],
            'trading_fee': [],
        }
        self.var_score={}
        for type in self.parameter_symbol_type_list:
            self.var_score[type] = 0

        # 实例变量
        self.DailyTotalPortfolioCapital = {}  # 存储每日收盘后的账户动态权益
        self.TotalTradingFees = 0  # 累计交易成本
        self.DailyTotalTradingFees = {}  # 存储每日收盘后的累计交易成本
        self.AllOrderRecordList = []  # 存储所有交易记录

        self.CurrentPosition_L = {}  # 存储实时的期货多头持仓，持仓信息是LBY_PositionInfo实例
        self.DailyPosition_L = {}  # 存储每日收盘后的期货多头持仓信息，持仓信息是LBY_PositionInfo实例
        self.CurrentPosition_S = {}  # 存储实时的期货空头持仓，持仓信息是LBY_PositionInfo实例
        self.DailyPosition_S = {}  # 存储每日收盘后的期货空头持仓信息，持仓信息是LBY_PositionInfo实例

        self.TradingDaysCounter = 0  # 从回测的那天起，第几个交易日。
        self.AllDominantContractCodeDic = {}  # 存储所有日期主力合约信息
        self.DominantContractCodeDic = {}  # 存储当日主力合约信息
        self.CurrentYear_1mKbarInfoDic = {}  # 存储当前年份的所有日内K线信息

        self.order_info_columns = ('code', 'from_lots', 'to_lots')
        self.order_info_long = pd.DataFrame(columns=self.order_info_columns)
        self.order_info_short = pd.DataFrame(columns=self.order_info_columns)
        self.intraday_1m_bars = {}  # 格式为pannel，即使单合约我也增加了'CU8888.XSGE'变成两个合约，这样返回结果一定是pannel格式
        self.traded_counter = 0  # 当天已交易次数


    # 统一类变量
    ''' 回测框架-参数区 '''
    BackTestStartCash = 10000  # 初始账户资金
    BackTestRebalanceTimeList = ['09:01']  # 时间戳为列表里时间的K线完成时触发调仓逻辑，每个时间格式hh:mm,比如'09:15'
    BackTestBeginDate = "2010-01-01"
    BackTestEndDate = "2020-01-10"
    BackTestFeeRatio = 1.0 / 10000
    BackTestSlipTicks = 1



    ''' 回测框架-变量区 '''
    # starting_trade_date = {'PB': datetime.date(2014, 7, 25),
    #                        'SN': datetime.date(2015, 12, 21),
    #                        'HC': datetime.date(2015, 12, 21),
    #                        'FU': datetime.date(2018, 7, 17),
    #                        'BU': datetime.date(2015, 3, 2),
    #                        'CS': datetime.date(2015, 9, 29),
    #                        'V': datetime.date(2016, 9, 12),
    #                        'J': datetime.date(2012, 6, 20),
    #                        'ZC': datetime.date(2015, 11, 19),
    #                        'MA': datetime.date(2014, 11, 19),
    #                        'OI': datetime.date(2013, 6, 3)}
    starting_trade_date={'BU':datetime.date(2014,9,16),
                         'B':datetime.date(2018,3,1)}
    SymbolMultiplier = {'RB': 10, 'HC': 10, 'I': 100, 'J': 100, 'JM': 60, 'ZC': 100, 'FG': 20, 'MA': 10, 'TA': 5,
                        'L': 5, 'PP': 5, 'BU': 10, 'RU': 10, 'CU': 5, 'AL': 5, 'ZN': 5, 'PB': 5, 'NI': 1, 'SN': 1,
                        'M': 10, 'Y': 10, 'RM': 10, 'OI': 10, 'P': 10, 'A': 10, 'C': 10, 'CS': 10, 'CF': 5, 'SR': 10,
                        'JD': 10, 'V': 5, 'EG': 10, 'AG': 15, 'AU': 1000, 'SC': 1000, 'B': 10, 'IF': 300, 'IH': 300,
                        'IC': 200, 'FU': 10, 'SA': 20, 'SS': 5, 'EB': 5, 'SP': 10, 'UR': 20, 'AP': 10, 'SM': 5, 'SF': 5,
                        'PG': 20, 'RI': 20, 'WH': 20, 'CJ': 5, 'LH': 16, 'NR': 10, 'PK': 5, 'PF': 5, }
    SymbolMinMovePoint = {'RB': 1, 'HC': 1, 'I': 0.5, 'J': 0.5, 'JM': 0.5, 'ZC': 0.2, 'FG': 1, 'MA': 1, 'TA': 2, 'L': 5,
                          'PP': 1, 'BU': 2, 'RU': 5, 'CU': 10, 'AL': 5, 'ZN': 5, 'PB': 5, 'NI': 10, 'SN': 10, 'M': 1,
                          'Y': 2, 'RM': 1, 'OI': 1, 'P': 2, 'A': 1, 'C': 1, 'CS': 1, 'CF': 5, 'SR': 1, 'JD': 1, 'V': 5,
                          'EG': 1, 'AG': 1, 'AU': 0.02,
                          'SC': 0.01, 'B': 1, 'IF': 0.2, 'IH': 0.2, 'IC': 0.2, 'FU': 1, 'SA': 1, 'SS': 5, 'EB': 1,
                          'SP': 2, 'UR': 1, 'AP': 1, 'SM': 2, 'SF': 2, 'PG': 1,'RI':1,'WH': 1, 'CJ': 5, 'LH': 5, 'NR': 5, 'PK': 2, 'PF': 2,}
    ''' 函数区 '''

    def reset_order_info(self):
        if len(self.order_info_long) > 0:
            # self.order_info_long = pd.DataFrame(columns=self.order_info_columns)
            self.order_info_long.drop(self.order_info_long.index, inplace=True)
        if len(self.order_info_short) > 0:
            # self.order_info_short = pd.DataFrame(columns=self.order_info_columns)
            self.order_info_short.drop(self.order_info_short.index, inplace=True)

    def compare_starting_trade_date(self, symbol, current_dt):
        if symbol not in self.starting_trade_date.keys():
            return True
        else:
            trade_day = self.starting_trade_date[symbol]
            delta = datetime.date(current_dt.year, current_dt.month, current_dt.day) - trade_day
            if delta.days > 0:
                return True
            else:
                return False

    def LBY_PrepareData(self):
        ## 读取交易信号
        signal = pd.read_csv("signal/水位rank_01_极值.csv", index_col=0)
        self.signal = signal


    def LBY_BeforeTradingDayBegin(self, current_dt):
        self.TradingDaysCounter = self.TradingDaysCounter + 1


    def During_Tradingday(self,current_dt):
        self.LBY_GenerateOrders(current_dt)

    def LBY_AfterTradingDayEnd(self, current_dt):
        my_time = datetime.datetime(current_dt.year, current_dt.month, current_dt.day, 15, 0, 0)
        signal_bar = self.signal[self.signal.index == str(current_dt)[:10]]
        if len(signal_bar) == 0:
            for symbol in self.parameter_symbol_type_list:
                self.var_score[symbol] = np.nan
        else:
            dominant_bar = pd.DataFrame(self.DominantContractCodeDic, index=['dominant_code'])
            total = pd.concat([signal_bar, dominant_bar], axis=0)
            total.dropna(inplace=True, axis=1)

            if len(total.columns) < 1:
                for symbol in self.parameter_symbol_type_list:
                    self.var_score[symbol] = 0
            else:
                total = total.iloc[[0]]
                long_number = total[total > 0].sum(axis=1)[0]
                short_number = total[total < 0].sum(axis=1)[0]
                total_number = long_number + abs(short_number)

                for symbol in self.parameter_symbol_type_list:
                    if (not self.compare_starting_trade_date(symbol, current_dt)) or (
                            symbol not in self.DominantContractCodeDic.keys()):
                        continue
                    else:
                        if signal_bar[symbol][0] == 1:
                            code = self.DominantContractCodeDic[symbol]
                            price = self.intraday_1m_bars.loc[my_time, code]
                            if self.var_score.get(symbol) > 0:
                                continue
                            else:
                                self.var_score[symbol] = int(
                                    self.BackTestStartCash / (total_number * price * self.SymbolMultiplier[code[:-4]]))

                        elif signal_bar[symbol][0] == -1:
                            code = self.DominantContractCodeDic[symbol]
                            price = self.intraday_1m_bars.loc[my_time, code]

                            if self.var_score.get(symbol) < 0:
                                continue
                            else:
                                self.var_score[symbol] = -int(self.BackTestStartCash / (
                                        total_number * price * self.SymbolMultiplier[code[:-4]]))

                        else:
                            self.var_score[symbol] = signal_bar[symbol][0]
        self.LBY_GenerateOrders(current_dt)



        # 策略核心逻辑开始
        # self.LBY_GenerateOrders(current_dt)


        #
        # for symbol in self.parameter_symbol_type_list:
        #     # mylength = self.parameter_length[symbol]
        #     # mylength = 5
        #     # remainder = self.TradingDaysCounter % (mylength * 2)
        #
        #     if len(temp)==0 or number<5:
        #         self.var_score[symbol] = np.nan
        #     else:
        #         if temp[symbol][0]==1:
        #             long_number+=1
        #         elif temp[symbol][0]==-1:
        #             short_number+=1
        #         self.var_score[symbol] = temp[symbol][0]
            # self.var_score[symbol] = remainder - mylength
        # print(self.var_score)
        # 策略核心逻辑结束


    def LBY_GenerateOrders(self, current_dt):
        self.reset_order_info()
        my_time = datetime.datetime(current_dt.year, current_dt.month, current_dt.day, 9, 1, 0)

        # 将持仓的合约类型与代码关联
        symbol_contract_map_L = {}
        symbol_contract_map_S = {}
        current_position_code_list = list(self.CurrentPosition_L.keys())
        for code in current_position_code_list:
            symbol_contract_map_L[code[:-4]] = code
        current_position_code_list = list(self.CurrentPosition_S.keys())
        for code in current_position_code_list:
            symbol_contract_map_S[code[:-4]] = code

        for symbol in self.parameter_symbol_type_list:
            if (not self.compare_starting_trade_date(symbol, current_dt)) or (
                    symbol not in self.DominantContractCodeDic.keys()):
                continue
            myscore = self.var_score[symbol]
            main_contract_code = self.DominantContractCodeDic[symbol]
            # main_contract_price = self.intraday_1m_bars['close'].loc[my_time, main_contract_code]

            # main_contract_price = self.intraday_1m_bars.loc[my_time, main_contract_code]

            target_lots = myscore
            # 获取当前持仓信息
            current_position_code_L = ''
            current_position_lots_L = 0
            current_position_code_S = ''
            current_position_lots_S = 0
            current_position_flag = 0
            if symbol in symbol_contract_map_L.keys():
                current_position_code_L = symbol_contract_map_L[symbol]
                current_position_lots_L = self.CurrentPosition_L[current_position_code_L].current_contracts
            if symbol in symbol_contract_map_S.keys():
                current_position_code_S = symbol_contract_map_S[symbol]
                current_position_lots_S = self.CurrentPosition_S[current_position_code_S].current_contracts
            if current_position_lots_L > 0:
                current_position_flag = 1
            elif current_position_lots_S > 0:
                current_position_flag = -1

            # 计算调仓信息
            if current_position_flag == 0:
                # 当前无持仓
                if target_lots > 0:
                    # 新开多
                    self.order_info_long = self.order_info_long.append([{self.order_info_columns[0]: main_contract_code,
                                                                         self.order_info_columns[1]: 0,
                                                                         self.order_info_columns[2]: target_lots}],
                                                                       ignore_index=True)
                elif target_lots < 0:
                    # 新开空
                    self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                               0]: main_contract_code,
                                                                           self.order_info_columns[1]: 0,
                                                                           self.order_info_columns[2]: abs(
                                                                               target_lots)}], ignore_index=True)
            elif current_position_flag == 1:
                # 当前持有多头头寸
                if current_position_lots_L == target_lots:
                    # 新老仓位一样的话，只需要判断是否需要移仓换月
                    if main_contract_code != current_position_code_L:
                        self.order_info_long = self.order_info_long.append([{self.order_info_columns[0]: current_position_code_L,
                                                                             self.order_info_columns[1]: current_position_lots_L,
                                                                             self.order_info_columns[2]: 0}],
                                                                           ignore_index=True)
                        self.order_info_long = self.order_info_long.append([{self.order_info_columns[0]: main_contract_code,
                                                                             self.order_info_columns[1]: 0,
                                                                             self.order_info_columns[2]: current_position_lots_L}],
                                                                           ignore_index=True)
                else:  # 新老仓位不一样
                    if main_contract_code != current_position_code_L:
                        # 需要移仓，先把老仓位平掉
                        self.order_info_long = self.order_info_long.append([{self.order_info_columns[
                                                                                 0]: current_position_code_L,
                                                                             self.order_info_columns[
                                                                                 1]: current_position_lots_L,
                                                                             self.order_info_columns[2]: 0}],
                                                                           ignore_index=True)
                        if target_lots > 0:
                            self.order_info_long = self.order_info_long.append([{self.order_info_columns[
                                                                                     0]: main_contract_code,
                                                                                 self.order_info_columns[1]: 0,
                                                                                 self.order_info_columns[
                                                                                     2]: target_lots}],
                                                                               ignore_index=True)
                        elif target_lots < 0:
                            self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                       0]: main_contract_code,
                                                                                   self.order_info_columns[1]: 0,
                                                                                   self.order_info_columns[2]: abs(
                                                                                       target_lots)}],
                                                                                 ignore_index=True)
                    else:
                        # 不需要移仓，只需要同合约调仓
                        if target_lots >= 0:
                            self.order_info_long = self.order_info_long.append([{self.order_info_columns[
                                                                                     0]: main_contract_code,
                                                                                 self.order_info_columns[
                                                                                     1]: current_position_lots_L,
                                                                                 self.order_info_columns[
                                                                                     2]: target_lots}],
                                                                               ignore_index=True)
                        else:
                            # 先把多头持仓全部平掉，再开空
                            self.order_info_long = self.order_info_long.append([{self.order_info_columns[
                                                                                     0]: main_contract_code,
                                                                                 self.order_info_columns[
                                                                                     1]: current_position_lots_L,
                                                                                 self.order_info_columns[2]: 0}],
                                                                               ignore_index=True)
                            self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                       0]: main_contract_code,
                                                                                   self.order_info_columns[1]: 0,
                                                                                   self.order_info_columns[2]: abs(
                                                                                       target_lots)}],
                                                                                 ignore_index=True)
            elif current_position_flag == -1:
                # 当前持有空头头寸
                if -current_position_lots_S == target_lots:
                    # 新老仓位一样的话，只需要判断是否需要移仓换月
                    if main_contract_code != current_position_code_S:
                        self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                   0]: current_position_code_S,
                                                                               self.order_info_columns[
                                                                                   1]: current_position_lots_S,
                                                                               self.order_info_columns[2]: 0}],
                                                                             ignore_index=True)
                        self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                   0]: main_contract_code,
                                                                               self.order_info_columns[1]: 0,
                                                                               self.order_info_columns[
                                                                                   2]: current_position_lots_S}],
                                                                             ignore_index=True)
                else:  # 新老仓位不一样
                    if main_contract_code != current_position_code_S:
                        # 需要移仓，先把老仓位平掉
                        self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                   0]: current_position_code_S,
                                                                               self.order_info_columns[
                                                                                   1]: current_position_lots_S,
                                                                               self.order_info_columns[2]: 0}],
                                                                             ignore_index=True)
                        if target_lots < 0:
                            self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                       0]: main_contract_code,
                                                                                   self.order_info_columns[1]: 0,
                                                                                   self.order_info_columns[2]: abs(
                                                                                       target_lots)}],
                                                                                 ignore_index=True)
                        elif target_lots > 0:
                            self.order_info_long = self.order_info_long.append([{self.order_info_columns[
                                                                                     0]: main_contract_code,
                                                                                 self.order_info_columns[1]: 0,
                                                                                 self.order_info_columns[2]: abs(
                                                                                     target_lots)}], ignore_index=True)
                    else:
                        # 不需要移仓，只需要同合约调仓
                        if target_lots <= 0:
                            self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                       0]: main_contract_code,
                                                                                   self.order_info_columns[
                                                                                       1]: current_position_lots_S,
                                                                                   self.order_info_columns[2]: abs(
                                                                                       target_lots)}],
                                                                                 ignore_index=True)
                        else:
                            # 先把空头持仓全部平掉，再开多
                            self.order_info_short = self.order_info_short.append([{self.order_info_columns[
                                                                                       0]: main_contract_code,
                                                                                   self.order_info_columns[
                                                                                       1]: current_position_lots_S,
                                                                                   self.order_info_columns[2]: 0}],
                                                                                 ignore_index=True)
                            self.order_info_long = self.order_info_long.append([{self.order_info_columns[
                                                                                     0]: main_contract_code,
                                                                                 self.order_info_columns[1]: 0,
                                                                                 self.order_info_columns[2]: abs(
                                                                                     target_lots)}], ignore_index=True)

    def LBY_ClearingAfterTradingDayEnd(self, current_dt):
        # print(self.TotalPortfolioCapital)
        # 公共模块开始，以下代码不要改动
        self.traded_counter = 0  # 当天已交易次数重置为0
        # 根据收盘价，结算各头寸当日持有盈亏，并更新各变量
        my_time = datetime.datetime(current_dt.year, current_dt.month, current_dt.day, 15, 0, 0)
        # 多头持仓
        # print('long position:')
        position_key_list = list(self.CurrentPosition_L.keys())
        for position_key in position_key_list:
            position_info = self.CurrentPosition_L[position_key]
            # position_info.PrintInfo()
            if position_info.current_contracts == 0:
                continue
            # today_close_price = self.intraday_1m_bars['close'].loc[my_time, position_key]

            today_close_price = self.intraday_1m_bars.loc[my_time, position_key]

            # 记录距离昨收价的平仓盈亏
            daily_profit = position_info.current_contracts * (
                        today_close_price - position_info.last_trading_day_close_price) * self.SymbolMultiplier[
                               position_info.symbol_type]
            # 更新最新权益
            self.TotalPortfolioCapital = self.TotalPortfolioCapital + daily_profit
            position_info.last_trading_day_close_price = today_close_price
            self.CurrentPosition_L[position_key] = position_info
        self.DailyPosition_L[current_dt] = self.CurrentPosition_L.copy()
        # print(self.TotalPortfolioCapital)
        # 空头持仓
        # print('short position:')
        position_key_list = list(self.CurrentPosition_S.keys())
        for position_key in position_key_list:
            position_info = self.CurrentPosition_S[position_key]
            # position_info.PrintInfo()
            if position_info.current_contracts == 0:
                continue
            # today_close_price = self.intraday_1m_bars['close'].loc[my_time, position_key]


            today_close_price = self.intraday_1m_bars.loc[my_time, position_key]





            # 记录距离昨收价的平仓盈亏
            daily_profit = (0 - position_info.current_contracts) * (
                        today_close_price - position_info.last_trading_day_close_price) * self.SymbolMultiplier[
                               position_info.symbol_type]
            # 更新最新权益
            self.TotalPortfolioCapital = self.TotalPortfolioCapital + daily_profit
            position_info.last_trading_day_close_price = today_close_price
            self.CurrentPosition_S[position_key] = position_info
        self.DailyPosition_S[current_dt] = self.CurrentPosition_S.copy()
        # 将收盘后的动态权益和累计费用记入历史
        self.DailyTotalPortfolioCapital[current_dt] = self.TotalPortfolioCapital
        self.DailyTotalTradingFees[current_dt] = self.TotalTradingFees
        # 公共模块结束，以上代码不要改动
        # print(self.TotalPortfolioCapital)

    def LBY_Buy(self, OrderTime, ContractCode, OrderLots, OrderPrice, MarketType='QH'):
        if OrderLots <= 0:
            print('error: can not Buy')
            return
        self.AllOrderRecordList.append(f"{OrderTime}, Buy, {ContractCode}, {OrderLots}, {OrderPrice}")

        # print(self.AllOrderRecordList[-1])
        position_info = self.CurrentPosition_L.get(ContractCode)
        if position_info.contract_code == '':
            position_info.contract_code = ContractCode
            position_info.symbol_type = ContractCode[:-4]
        # 记录本次交易的费用
        trading_fee = OrderLots * (OrderPrice * self.SymbolMultiplier[position_info.symbol_type] * self.BackTestFeeRatio +
                                   self.BackTestSlipTicks * self.SymbolMinMovePoint[position_info.symbol_type] * self.SymbolMultiplier[position_info.symbol_type])

        # 更新最新权益
        self.TotalPortfolioCapital = self.TotalPortfolioCapital - trading_fee
        # 更新最新总费用
        self.TotalTradingFees = self.TotalTradingFees + trading_fee
        # 更新持仓的其他属性信息
        position_info.avg_entry_price = (position_info.avg_entry_price * position_info.current_contracts + OrderPrice * OrderLots) / (
                                                    position_info.current_contracts + OrderLots)
        position_info.last_trading_day_close_price = (
                                                                 position_info.last_trading_day_close_price * position_info.current_contracts + OrderPrice * OrderLots) / (
                                                                 position_info.current_contracts + OrderLots)
        position_info.current_contracts = position_info.current_contracts + OrderLots
        self.CurrentPosition_L[ContractCode] = position_info

        #记录详细的order信息
        self.AllOrderRecordList_specific['time'].append(OrderTime)
        self.AllOrderRecordList_specific['buy/sell'].append('买')
        self.AllOrderRecordList_specific['contract_code'].append(ContractCode)
        self.AllOrderRecordList_specific['order_lots'].append(OrderLots)
        self.AllOrderRecordList_specific['order_price'].append(OrderPrice)
        self.AllOrderRecordList_specific['trading_fee'].append(trading_fee)


        # position_info.PrintInfo()
        # print(self.TotalPortfolioCapital)

    def LBY_BuyToCover(self, OrderTime, ContractCode, OrderLots, OrderPrice, MarketType='QH'):
        if OrderLots <= 0:
            print('error: can not BuyToCover')
            return
        self.AllOrderRecordList.append(f"{OrderTime}, BuyToCover, {ContractCode}, {OrderLots}, {OrderPrice}")
        # print(self.AllOrderRecordList[-1])
        position_info = self.CurrentPosition_S.get(ContractCode)
        if position_info is None or position_info.contract_code == '' or position_info.current_contracts == 0:
            print(f"{ContractCode}无空头持仓，无法平仓\n")
        else:
            # 买入平仓，不仅要计算交易费用，还要计算距离昨收价的平仓盈亏
            # 记录本次交易的费用
            trading_fee = OrderLots * (OrderPrice * self.SymbolMultiplier[
                position_info.symbol_type] * self.BackTestFeeRatio + self.BackTestSlipTicks * self.SymbolMinMovePoint[
                                           position_info.symbol_type] * self.SymbolMultiplier[
                                           position_info.symbol_type])
            # 记录距离昨收价的平仓盈亏
            daily_profit = (0 - OrderLots) * (OrderPrice - position_info.last_trading_day_close_price) * \
                           self.SymbolMultiplier[position_info.symbol_type]
            # 更新最新权益
            self.TotalPortfolioCapital = self.TotalPortfolioCapital - trading_fee + daily_profit
            # 更新最新总费用
            self.TotalTradingFees = self.TotalTradingFees + trading_fee
            # 平仓不用改变平均持仓价及昨收价，只需要更新持仓头寸就行
            position_info.current_contracts = position_info.current_contracts - OrderLots
            # self.CurrentPosition_S[ContractCode] = position_info
            if position_info.current_contracts == 0:
                self.CurrentPosition_S.pop(ContractCode)
            else:
                self.CurrentPosition_S[ContractCode] = position_info
            # print(self.TotalPortfolioCapital)
            self.AllOrderRecordList_specific['time'].append(OrderTime)
            self.AllOrderRecordList_specific['buy/sell'].append('买平')
            self.AllOrderRecordList_specific['contract_code'].append(ContractCode)
            self.AllOrderRecordList_specific['order_lots'].append(OrderLots)
            self.AllOrderRecordList_specific['order_price'].append(OrderPrice)
            self.AllOrderRecordList_specific['trading_fee'].append(trading_fee)

    def LBY_SellShort(self, OrderTime, ContractCode, OrderLots, OrderPrice, MarketType='QH'):
        if OrderLots <= 0:
            print('error: can not SellShort')
            return
        self.AllOrderRecordList.append(f"{OrderTime}, SellShort, {ContractCode}, {OrderLots}, {OrderPrice}")
        # print(self.AllOrderRecordList[-1])
        position_info = self.CurrentPosition_S.get(ContractCode)
        if position_info.contract_code == '':
            position_info.contract_code = ContractCode
            position_info.symbol_type = ContractCode[:-4]
        # 记录本次交易的费用
        trading_fee = OrderLots * (OrderPrice * self.SymbolMultiplier[
            position_info.symbol_type] * self.BackTestFeeRatio + self.BackTestSlipTicks * self.SymbolMinMovePoint[
                                       position_info.symbol_type] * self.SymbolMultiplier[position_info.symbol_type])
        # 更新最新权益
        self.TotalPortfolioCapital = self.TotalPortfolioCapital - trading_fee
        # 更新最新总费用
        self.TotalTradingFees = self.TotalTradingFees + trading_fee
        # 更新持仓的其他属性信息
        position_info.avg_entry_price = (
                                                    position_info.avg_entry_price * position_info.current_contracts + OrderPrice * OrderLots) / (
                                                    position_info.current_contracts + OrderLots)
        position_info.last_trading_day_close_price = (
                                                                 position_info.last_trading_day_close_price * position_info.current_contracts + OrderPrice * OrderLots) / (
                                                                 position_info.current_contracts + OrderLots)
        position_info.current_contracts = position_info.current_contracts + OrderLots
        self.CurrentPosition_S[ContractCode] = position_info
        # position_info.PrintInfo()
        # print(self.TotalPortfolioCapital)
        self.AllOrderRecordList_specific['time'].append(OrderTime)
        self.AllOrderRecordList_specific['buy/sell'].append('卖')
        self.AllOrderRecordList_specific['contract_code'].append(ContractCode)
        self.AllOrderRecordList_specific['order_lots'].append(OrderLots)
        self.AllOrderRecordList_specific['order_price'].append(OrderPrice)
        self.AllOrderRecordList_specific['trading_fee'].append(trading_fee)

    def LBY_Sell(self, OrderTime, ContractCode, OrderLots, OrderPrice, MarketType='QH'):
        if OrderLots <= 0:
            print('error: can not Sell')
            return
        self.AllOrderRecordList.append(f"{OrderTime}, Sell, {ContractCode}, {OrderLots}, {OrderPrice}")
        # print(self.AllOrderRecordList[-1])
        position_info = self.CurrentPosition_L.get(ContractCode)
        if position_info is None or position_info.contract_code == '' or position_info.current_contracts == 0:
            print(f"{ContractCode}无多头持仓，无法平仓\n")
        else:
            # 卖出平仓，不仅要计算交易费用，还要计算距离昨收价的平仓盈亏
            # 记录本次交易的费用
            trading_fee = OrderLots * (OrderPrice * self.SymbolMultiplier[
                position_info.symbol_type] * self.BackTestFeeRatio + self.BackTestSlipTicks * self.SymbolMinMovePoint[
                                           position_info.symbol_type] * self.SymbolMultiplier[
                                           position_info.symbol_type])
            # 记录距离昨收价的平仓盈亏
            daily_profit = OrderLots * (OrderPrice - position_info.last_trading_day_close_price) * \
                           self.SymbolMultiplier[position_info.symbol_type]
            # 更新最新权益
            self.TotalPortfolioCapital = self.TotalPortfolioCapital - trading_fee + daily_profit
            # 更新最新总费用
            self.TotalTradingFees = self.TotalTradingFees + trading_fee
            # 平仓不用改变平均持仓价及昨收价，只需要更新持仓头寸就行
            position_info.current_contracts = position_info.current_contracts - OrderLots
            # self.CurrentPosition_L[ContractCode] = position_info
            if position_info.current_contracts == 0:
                self.CurrentPosition_L.pop(ContractCode)
            else:
                self.CurrentPosition_L[ContractCode] = position_info
            # print(self.TotalPortfolioCapital)
            self.AllOrderRecordList_specific['time'].append(OrderTime)
            self.AllOrderRecordList_specific['buy/sell'].append('卖平')
            self.AllOrderRecordList_specific['contract_code'].append(ContractCode)
            self.AllOrderRecordList_specific['order_lots'].append(OrderLots)
            self.AllOrderRecordList_specific['order_price'].append(OrderPrice)
            self.AllOrderRecordList_specific['trading_fee'].append(trading_fee)

    # ToLots: 目标仓位，绝对值. Side: 持仓方向，long/short
    def LBY_OrderToTarget(self, OrderTime, ContractCode, ToLots, OrderPrice, Side='long', MarketType='QH'):
        # print(f"LBY_OrderToTarget {OrderTime}, {ContractCode}, {ToLots}, {OrderPrice}, {Side}")
        if Side == 'long':
            position_info = self.CurrentPosition_L.get(ContractCode)
            if position_info is None:
                if ToLots > 0:
                    position_info = LBY_PositionInfo(ContractCode)
                    self.CurrentPosition_L[ContractCode] = position_info
                    # 全新开仓
                    order_lots = abs(ToLots - position_info.current_contracts)
                    self.LBY_Buy(OrderTime, ContractCode, order_lots, OrderPrice, MarketType='QH')
            else:
                # 已有持仓信息
                if ToLots > position_info.current_contracts:
                    # 加仓
                    order_lots = abs(ToLots - position_info.current_contracts)
                    self.LBY_Buy(OrderTime, ContractCode, order_lots, OrderPrice, MarketType='QH')
                elif ToLots < position_info.current_contracts:
                    # 减仓
                    order_lots = abs(ToLots - position_info.current_contracts)
                    self.LBY_Sell(OrderTime, ContractCode, order_lots, OrderPrice, MarketType='QH')

        elif Side == 'short':
            position_info = self.CurrentPosition_S.get(ContractCode)
            if position_info is None:
                if ToLots > 0:
                    position_info = LBY_PositionInfo(ContractCode)
                    self.CurrentPosition_S[ContractCode] = position_info
                    # 全新开仓
                    order_lots = abs(ToLots - position_info.current_contracts)
                    self.LBY_SellShort(OrderTime, ContractCode, order_lots, OrderPrice, MarketType='QH')
            else:
                # 已有持仓信息
                if ToLots > position_info.current_contracts:
                    # 加仓
                    order_lots = abs(ToLots - position_info.current_contracts)
                    self.LBY_SellShort(OrderTime, ContractCode, order_lots, OrderPrice, MarketType='QH')
                elif ToLots < position_info.current_contracts:
                    # 减仓
                    order_lots = abs(ToLots - position_info.current_contracts)
                    self.LBY_BuyToCover(OrderTime, ContractCode, order_lots, OrderPrice, MarketType='QH')

    # 调仓
    def LBY_RebalancePosition(self, current_dt):
        if len(self.intraday_1m_bars) > 0:
            # bar_times = list(self.intraday_1m_bars['open'].index)
            for rebalance_time_str in self.BackTestRebalanceTimeList:
                strs = rebalance_time_str.split(":", 1)
                rebalance_time = datetime.datetime(current_dt.year, current_dt.month, current_dt.day, int(strs[0]),
                                                   int(strs[1]), 0)
                self.traded_counter = self.traded_counter + 1
                for index, row in self.order_info_long.iterrows():
                    contract_code = row[self.order_info_columns[0]]
                    from_lots = row[self.order_info_columns[1]]
                    to_lots = row[self.order_info_columns[2]]
                    each_lots = math.ceil(abs(to_lots - from_lots) / len(self.BackTestRebalanceTimeList))
                    each_lots = max(1, each_lots)


                    order_price = self.intraday_1m_bars.loc[rebalance_time, contract_code]



                    if from_lots > to_lots:
                        # 减仓
                        self.LBY_OrderToTarget(rebalance_time, contract_code,
                                               max(to_lots, from_lots - each_lots * self.traded_counter), order_price,
                                               Side='long')
                    elif from_lots < to_lots:
                        # 加仓
                        self.LBY_OrderToTarget(rebalance_time, contract_code,
                                               min(to_lots, from_lots + each_lots * self.traded_counter), order_price,
                                               Side='long')
                for index, row in self.order_info_short.iterrows():
                    contract_code = row[self.order_info_columns[0]]
                    from_lots = row[self.order_info_columns[1]]
                    to_lots = row[self.order_info_columns[2]]
                    each_lots = math.ceil(abs(to_lots - from_lots) / len(self.BackTestRebalanceTimeList))
                    each_lots = max(1, each_lots)
                    # order_price = self.intraday_1m_bars['close'].loc[rebalance_time, contract_code]

                    order_price = self.intraday_1m_bars.loc[rebalance_time, contract_code]

                    if from_lots > to_lots:

                        # 减仓
                        self.LBY_OrderToTarget(rebalance_time, contract_code,
                                               max(to_lots, from_lots - each_lots * self.traded_counter), order_price,
                                               Side='short')
                    elif from_lots < to_lots:
                        # 加仓
                        self.LBY_OrderToTarget(rebalance_time, contract_code,
                                               min(to_lots, from_lots + each_lots * self.traded_counter), order_price,
                                               Side='short')

    # 增加后续分析结果生成的部分
    def Trading_Analysis(self):
        if not os.path.exists("type_res"):
            os.makedirs("type_res")

        file_name = str(datetime.datetime.now())[:19]
        file_name = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", file_name)
        file_path = "type_res/"+file_name+self.parameter_symbol_type_list[0]

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # 交易记录
        order_log = self.AllOrderRecordList_specific
        order_log = pd.DataFrame(order_log)
        order_log.to_excel(file_path+"/order_log.xlsx")
        start_time = pd.to_datetime(str(order_log['time'][0])[:10])

        # 净值计算
        PNL = self.DailyTotalPortfolioCapital
        PNL = pd.DataFrame(PNL, index=['净值'])
        PNL = PNL.T
        PNL = PNL[PNL.index>=start_time]
        PNL['净值'] = PNL['净值'].apply(lambda x: x / self.BackTestStartCash)
        PNL['回撤'] = PNL['净值']-PNL['净值'].cummax()
        fig = plt.figure(figsize=(32,20))
        ax1 = plt.subplot(111)
        PNL['净值'].plot(color='b')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc='upper left',fontsize=20)
        ax2 = plt.twinx()
        PNL['回撤'].plot(color='y',alpha=0.5)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig(file_path+"/PNL.png")

        # 交易结果的指标计算
        during_date = (pd.to_datetime(self.BackTestEndDate)-start_time).days
        during_year = during_date / 365.0
        res = {}
        res['收益%'] = (PNL['净值'][-1]-1)*100
        res['年化收益%'] = res['收益%']/during_year
        res['最大回撤%'] = abs(PNL['回撤'].min())*100
        res['收益回撤比'] = res['收益%']/res['最大回撤%']
        res = pd.DataFrame(res,index=['回测统计'])
        res.to_excel(file_path+"/result_analysis.xlsx")
        PNL.to_excel(file_path+"/PNL.xlsx")

    def LBY_DoBackTest(self):
        # print('Working...\n')
        ''' 预加载平台数据开始 如果没数据，请重新正确运行名为00的notebook '''
        file_path_dominant_contract_code = 'C:/Project/FuturesBackTester/pickle/dominant_contract_code.pickle'
        f = open(file_path_dominant_contract_code, 'rb')
        self.AllDominantContractCodeDic = pickle.load(f)
        f.close()
        ''' 预加载平台数据结束 '''

        self.reset_order_info()
        self.LBY_PrepareData()

        # 获取交易日列表
        # now_time = datetime.datetime.now()
        # rqdatac.init()
        # HS300 = get_price('000001.XSHE', start_date='2010-01-04',
        #                   end_date=datetime.datetime.strftime(now_time, "%Y-%m-%d"), fields='close')
        HS300 = self.HS300

        HS300.index = pd.to_datetime(HS300.index)


        self.trading_day = HS300
        self.trading_day = self.trading_day[(self.trading_day.index>=self.BackTestBeginDate)&(self.trading_day.index<=self.BackTestEndDate)]
        trading_date_list = list(self.trading_day.index)
        current_year = 0

        for trading_date in trading_date_list:
            # print(f"{trading_date}")
            # 获取当日(白天，不做夜盘)涉及到的所有合约的1分钟K线信息

            self.DominantContractCodeDic = self.AllDominantContractCodeDic[trading_date]
            if current_year == 0 or current_year != trading_date.year:
                print(f"dealing with {trading_date.year}")
                current_year = trading_date.year
                f = open(f"C:/Project/FuturesBackTester/pickle/1m_kbars_{current_year}.pickle", 'rb')
                self.CurrentYear_1mKbarInfoDic = pickle.load(f)
                f.close()

            self.intraday_1m_bars = self.CurrentYear_1mKbarInfoDic[trading_date]
            # if trading_date.year==2019 and trading_date.month==4 and trading_date.day==3:
            #    print(self.intraday_1m_bars['close'].columns)
            '''
            contract_code_list = list(self.order_info_long[self.order_info_columns[0]])+list(self.order_info_short[self.order_info_columns[0]])+list(self.CurrentPosition_L.keys())+list(self.CurrentPosition_S.keys())
            for symbol_type in self.parameter_symbol_type_list: #这里可以优化，可以提前存储每日主力合约
                contrace_code = get_dominant_future(symbol_type, trading_date)
                self.DominantContractCodeDic[symbol_type] = contrace_code
                contract_code_list = contract_code_list + [contrace_code]
            contract_code_list = list(set(contract_code_list))

            if len(contract_code_list)>0:
                if len(contract_code_list) ==1:
                    contract_code_list = contract_code_list + ['CU8888.XSGE']
                self.intraday_1m_bars = get_price(contract_code_list, start_date=datetime.datetime(trading_date.year,trading_date.month,trading_date.day, 8, 0, 0), end_date=datetime.datetime(trading_date.year,trading_date.month,trading_date.day, 15, 15, 0), fields=['open', 'close', 'high', 'low', 'volume'], frequency='1m')
            else:
                self.intraday_1m_bars = {}
            '''
            # print(self.intraday_1m_bars['close'])
            # 执行具体的业务函数
            self.LBY_BeforeTradingDayBegin(trading_date)
            # self.During_Tradingday(trading_date)
            self.LBY_RebalancePosition(trading_date)
            self.LBY_ClearingAfterTradingDayEnd(trading_date)
            self.LBY_AfterTradingDayEnd(trading_date)
            # print(f"  long: {self.order_info_long}\n  short:{self.order_info_short}\n----------\n")
            # break
        stat_res,pnl = self.Trading_Analysis()
        return stat_res,pnl

# ============================  Demo  ============================

def run_single_setting(type_list):
    # print(datetime.datetime.now())
    StartCash = 1000000000  # 初始账户资金
    RebalanceTimeList = ['09:01']  # 时间戳为列表里时间的K线完成时触发调仓逻辑，每个时间格式hh:mm,比如'09:15'
    BeginDate = "2010-01-01"
    EndDate = "2020-11-24"
    backtester = LBY_BackTester_QH(StartCash, RebalanceTimeList, BeginDate, EndDate)
    backtester.parameter_symbol_type_list = type_list
    backtester.LBY_DoBackTest()


if __name__ == '__main__':
    total_list = ['TA','RM','OI','PP','L','CU','ZN','NI','AL','FG','CF','RU','V','BU','ZC','MA','HC','RB','I','J','JM','Y','P','M','A','EG']
    run_single_setting(total_list)




