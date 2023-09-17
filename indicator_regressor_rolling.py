# -- coding: utf-8 --
"""
 @time : 2023/7/28
 @file : indicator_regressor_rolling.py
 @author : zhenghao
 @software: PyCharm
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from xgboost import XGBRegressor,XGBClassifier
from xgboost import plot_importance, plot_tree
import joblib
from Simple_Backtest_hedge import simple_backtest
import multiprocessing as mp
from functools import partial
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (21, 10)

plt.rcParams.update({'font.size':20})
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=16)


class indicator_regressor:


    def __init__(self,factor_list,model_name):
        self.factor_list = factor_list
        self.ret_y = pd.read_csv("ret_cache/Open-5DOpen.csv",index_col=0,parse_dates=[0])
        self.group_info = []
        self.group_info_pred = []
        self.model_name = model_name
        self.tradingday = pd.read_csv("all_tradeday.csv",index_col=0,parse_dates=[0])


    def fixed_timeserise_split(self,df_factor,ret_y,fixed_win=244,last_period=20):
        datetime_list = sorted(list(set(df_factor.index)))

        split_data_list = []

        df_factor = df_factor.reset_index().set_index(["datetime","symbol"])

        for i in range(fixed_win,len(datetime_list),last_period):
            if i+last_period<len(datetime_list):
                train_date = datetime_list[i]
                test_start_date = datetime_list[i+1]
                test_end_date = datetime_list[i+last_period]
            else: #如果长度不够,就把最后有多少长度的就多少当成样本外
                train_date = datetime_list[i]
                test_start_date = datetime_list[i + 1]
                test_end_date = datetime_list[-1]

            temp_train_x = np.array(df_factor.loc[:train_date])
            temp_train_y = np.array(ret_y.loc[:train_date])
            temp_test_x = np.array((df_factor.loc[test_start_date:test_end_date]))
            temp_test_y = np.array((ret_y.loc[test_start_date:test_end_date]))
            split_datetime = df_factor.loc[:train_date].index, df_factor.loc[test_start_date:test_end_date].index
            split_data_list.append([temp_train_x, temp_train_y, temp_test_x, temp_test_y, split_datetime])

        return split_data_list


    def rolling_timeserise_split(self,df_factor,ret_y,train_batch_size,test_batch_size,pred_len):

        datetime_list = sorted(list(set(df_factor.index)))
        split_data_list = []

        one_hot_variable = pd.get_dummies(df_factor['symbol'])
        df_factor = pd.concat([df_factor,one_hot_variable],axis=1)
        df_factor = df_factor.reset_index().set_index(["datetime", "symbol"])

        for i in range(train_batch_size,len(datetime_list),test_batch_size):
            train_start_date = datetime_list[i-train_batch_size]
            train_end_date = datetime_list[i-pred_len]

            test_start_date = datetime_list[min(i+1,len(datetime_list)-1)]
            test_end_date = datetime_list[min(i+test_batch_size,len(datetime_list)-1)]

            temp_train_x = np.array(df_factor.loc[train_start_date:train_end_date])
            temp_train_y = np.array(ret_y.loc[train_start_date:train_end_date])
            temp_test_x = np.array((df_factor.loc[test_start_date:test_end_date]))
            temp_test_y = np.array((ret_y.loc[test_start_date:test_end_date]))
            split_datetime = df_factor.loc[train_start_date:train_end_date].index, df_factor.loc[
                                                                                   test_start_date:test_end_date].index
            split_data_list.append([temp_train_x, temp_train_y, temp_test_x, temp_test_y, split_datetime])

        return split_data_list


    def adjust_tradingday(self,date,shift=1):
        """
        传入的date需要时pd.datetime后的格式，Timestamp
        """
        pos = list(self.tradingday.index).index(date)
        aj_pos = pos - shift
        return list(self.tradingday.index)[aj_pos]

    def preprocess(self,starttime,endtime,train_size,test_size,pred_len,method):

        df_factor = []



        # ret_y = self.ret_y.stack()

        #截面rank_y
        ret_y_std = self.ret_y.apply(lambda x:(x-x.mean())/x.std(),axis=1)
        ret_y = ret_y_std.stack()



        for factor in self.factor_list:
            temp = pd.read_csv(f"factor_res/{factor}/{factor}_val.csv",index_col=0,parse_dates=[0])
            temp = temp[(temp.index>=pd.to_datetime(starttime))&(temp.index<pd.to_datetime(endtime))]
            df_factor.append(temp.stack())

        df_factor = pd.concat(df_factor,axis=1).sort_index()
        ret_y = ret_y.reindex(df_factor.index).fillna(0)



        df_factor = df_factor.reset_index().rename(columns={'level_0': 'datetime','level_1': 'symbol'})
        df_factor = df_factor.set_index("datetime")

        if method == 'fixed':
            split = self.fixed_timeserise_split(df_factor=df_factor,ret_y=ret_y,fixed_win=rolling_win,last_period=last_win)
        elif method == 'rolling':
            split = self.rolling_timeserise_split(df_factor=df_factor, ret_y=ret_y, train_batch_size=train_size,test_batch_size=test_size,pred_len=pred_len)



        scaled_dataset = []
        for dataset in split:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(dataset[0])
            X_test_scaled = scaler.transform(dataset[2])
            X_train_scaled = np.nan_to_num(X_train_scaled)
            X_test_scaled = np.nan_to_num(X_test_scaled)

            # y这里还没有做处理，需要考虑后面
            y_train = dataset[1]
            y_test = dataset[3]
            split_time = dataset[4]
            scaled_dataset.append([X_train_scaled,y_train,X_test_scaled,y_test,split_time])

        return scaled_dataset



    def train(self,X_train_scaled,y_train, model_name='lr000'):
        print("%s training......" % model_name)

        model = model_name.split("_")[0]

        if model == 'ew':
            pass

        if model == 'lr':
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train_scaled, y_train)
            joblib.dump(reg, 'models/'+model_name)

        if model == 'lasso':
            alpha = 0.3 # L1 正则化项的权重，调整 alpha 可以控制稀疏性
            reg = Lasso(alpha=alpha)
            reg.fit(X_train_scaled, y_train)
            joblib.dump(reg, 'models/' + model_name)



        if model == 'xgb':
            reg = XGBRegressor(max_depth = 10, n_estimators = 20)
            reg.fit(X_train_scaled,y_train)
            joblib.dump(reg, 'models/'+model_name)
            # parameters = {
            #     'n_estimators': [100, 200, 300, 400],
            #     'learning_rate': [0.001, 0.005, 0.01, 0.05],
            #     'max_depth': [8, 10, 12, 15],
            #     'gamma': [0.001, 0.005, 0.01, 0.02],
            #     'random_state': [42],
            # }
            #
            # eval_set = [(X_train_scaled, y_train), (X_valid_scaled, y_valid)]
            # model = XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
            # clf = GridSearchCV(model, parameters)
            # clf.fit(X_train_scaled, y_train)
            #
            # print(f'Best params: {clf.best_params_}')
            # print(f'Best validation score = {clf.best_score_}')
            #
            # ##得到最佳参数后训练模型
            # model = XGBRegressor(**clf.best_params_, objective='reg:squarederror')
            # model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
            # joblib.dump(model, 'models/xgb')
            #
            # plot_importance(model)

        if model == 'adaboost':
            rng = np.random.RandomState(1)
            reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),
                                    n_estimators=50, random_state=rng)  # AdaBoost回归
            reg.fit(X_train_scaled, y_train)
            joblib.dump(reg, 'models/'+model_name)

        return reg

    def predict(self,X_test_scaled, model_name='lr000'):

        model = model_name.split("_")[0]

        if model == 'ew':
            y_predict = X_test_scaled.mean(axis=1)
            return y_predict

        else:
            model_ = joblib.load(f'models/{model_name}')
            y_predict = model_.predict(X_test_scaled)

            return y_predict


    def out_sample_predict(self,predict_start_time):
        df_factor = []
        for factor in self.factor_list:
            temp = pd.read_csv(f"factor_res/{factor}/{factor}_val.csv", index_col=0, parse_dates=[0])
            temp = temp.loc[predict_start_time:]
            df_factor.append(temp.stack())

        df_factor = pd.concat(df_factor,axis=1).sort_index()
        pred_X = np.array(df_factor)
        loaded_scaler = joblib.load('scalers/scaler')
        pred_scaled_x = loaded_scaler.transform(pred_X)
        pred_scaled_x = np.nan_to_num(pred_scaled_x)

        model_ = joblib.load(f'models/{self.model_name}')
        y_predict = model_.predict(pred_scaled_x)
        res = pd.DataFrame({'predict': y_predict}).set_index(df_factor.index)
        pred = res['predict'].unstack().fillna(0)
        pred.apply(lambda x: self.handle_one_bar(x, 5,flag=1), axis=1)
        self.group_info_pred = pd.concat(self.group_info_pred)
        self.backtest(self.group_info_pred,model_name=f"{self.model_name}outsample")



    def handle_one_bar(self,df, N,ascending=False):
        groups, bins = pd.qcut(df.rank(method='first', ascending=ascending), q=N, labels=False, retbins=True)
        return groups

    def get_longshort_signal(self,group_info):
        group_one = group_info[group_info == 0].applymap(lambda x: 1 if x == 0 else 0)
        group_five = group_info[group_info == 4].fillna(0).applymap(lambda x: np.sign(x))
        group_longshort = group_one - group_five
        group_longshort.fillna(0,inplace=True)
        return group_longshort



    def load_pipeline(self,starttime,endtime,train_size,test_size,last_win,method=None):

        # 根据分组，直接单因子之间净值整合回测
        if self.model_name=="ew_pnl":
            signal_list = []
            for k in self.factor_list:
                temp = pd.read_csv(f"factor_res/{k}/{k}_group.csv",index_col=0,parse_dates=[0])
                temp_signal = self.get_longshort_signal(temp)
                signal_list.append(temp_signal)

            pool = mp.Pool(processes=16)
            run_single_test_p = partial(simple_backtest,tradingday=self.tradingday, adjust_days=last_win, trading_fee=True)
            res = pool.map(run_single_test_p, signal_list)
            pool.close()
            pool.join()

            stat_df = []
            pnl_df = []
            for k in list(res):
                stat_df.append(k[0])
                pnl_df.append(k[1]['净值'])
            stat_df = pd.concat(stat_df,axis=0)
            stat_df.index = self.factor_list

            pnl_df = pd.concat(pnl_df,axis=1)
            pnl_df.columns = self.factor_list
            pnl_df['均值组合'] = pnl_df.diff().mean(axis=1).cumsum()+1

            append_df = {}
            append_df['收益%'] = 100*(pnl_df['均值组合'].iloc[-1]-1)
            append_df['年化收益%'] = pnl_df['均值组合'].diff().mean()*244*100
            append_df['年化波动%'] = pnl_df['均值组合'].diff().std()*np.sqrt(244)*100
            append_df['sharpe'] = append_df['年化收益%']/append_df['年化波动%']
            append_df['最大回撤%'] = abs((pnl_df['均值组合']-pnl_df['均值组合'].cummax()).min())*100
            append_df['收益回撤比'] = append_df['年化收益%']/append_df['最大回撤%']
            append_df = pd.DataFrame(append_df,index=['均值组合'])
            stat_df = pd.concat([stat_df,append_df],axis=0)
            stat_df = round(stat_df,2)


            if not os.path.exists(f"backtest_res/{self.model_name}_{str(last_win)}/"):
                os.makedirs(f"backtest_res/{self.model_name}_{str(last_win)}/")
            pnl_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}/pnl.xlsx")
            stat_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}/stat.xlsx")
            # fig = plt.figure(figsize=(32,20))
            # pnl_df.plot(subplots=True,layout=(4,4),title=f'因子绩效_{str(last_win)}Days')
            # plt.tight_layout()
            # plt.savefig(f"backtest_res/{self.model_name}_{str(last_win)}/"+"pnl_plot.jpg")


        # 直接信号相加，相比第一种省了手续费，自己先轧差
        elif self.model_name=='ew_signal':
            signal_df = pd.DataFrame()
            root = "C:/Project/截面因子测试/"
            for k in self.factor_list:
                temp = pd.read_csv(root+f"factor_res/{k}/{k}_group.csv", index_col=0, parse_dates=[0])
                temp_signal = self.get_longshort_signal(temp)
                signal_df = signal_df.add(temp_signal,fill_value=0)

            stat_df,pnl_df = simple_backtest(pos=signal_df,tradingday=self.tradingday, adjust_days=last_win, trading_fee=True)

            if not os.path.exists(f"backtest_res/{self.model_name}_{str(last_win)}"):
                os.makedirs(f"backtest_res/{self.model_name}_{str(last_win)}")

            stat_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}/stat.xlsx")
            pnl_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}/pnl.xlsx")


        # 先信号rank相加排序，再给出最后信号
        elif self.model_name=='ew_signal_rank':
            signal_rank = pd.DataFrame()
            count = pd.DataFrame()
            for k in self.factor_list:
                temp = pd.read_csv(f"factor_res/{k}/{k}_group.csv", index_col=0, parse_dates=[0])
                temp_count = temp.applymap(lambda x : 0 if np.isnan(x) else 1)
                count = count.add(temp_count,fill_value=0)
                signal_rank = signal_rank.add(temp, fill_value=0)

            signal_score = signal_rank/count
            signal_group = signal_score.apply(lambda x:self.handle_one_bar(x, 5,ascending=True))
            signal_df = self.get_longshort_signal(signal_group)

            stat_df, pnl_df = simple_backtest(pos=signal_df, tradingday=self.tradingday, adjust_days=last_win,
                                              trading_fee=True)

            if not os.path.exists(f"backtest_res/{self.model_name}_{str(last_win)}"):
                os.makedirs(f"backtest_res/{self.model_name}_{str(last_win)}")

            stat_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}/stat.xlsx")
            pnl_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}/pnl.xlsx")


        elif self.model_name in ['lr','lasso','adaboost']:
            train_dataset = self.preprocess(starttime=starttime,endtime=endtime,train_size=train_size,
                                            test_size=test_size,pred_len=last_win,method=method)
            pred_df = []
            epoch = 0
            for dataset in train_dataset:
                reg = self.train(dataset[0],dataset[1],model_name=self.model_name+f"_{epoch}")
                epoch+=1
                y_predict = reg.predict(dataset[2])
                y_predict = pd.DataFrame({'predict': y_predict, 'ytrue': dataset[3]}).set_index(dataset[4][1])
                pred_df.append(y_predict)


            pred_df = pd.concat(pred_df)
            print(pred_df.corr(method='spearman'))

            #直接分组进行回测
            pred = pred_df['predict'].unstack()
            pred_group = pred.apply(lambda x:self.handle_one_bar(x, 5), axis=1) # 比例切分会导致一天被拆开
            signal_df = self.get_longshort_signal(pred_group)

            stat_df, pnl_df = simple_backtest(pos=signal_df, tradingday=self.tradingday, adjust_days=last_win,trading_fee=True)
            if not os.path.exists(f"backtest_res/{self.model_name}_{str(last_win)}_{str(train_size)}_{str(test_size)}"):
                os.makedirs(f"backtest_res/{self.model_name}_{str(last_win)}_{str(train_size)}_{str(test_size)}")

            stat_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}_{str(train_size)}_{str(test_size)}/stat.xlsx")
            pnl_df.to_excel(f"backtest_res/{self.model_name}_{str(last_win)}_{str(train_size)}_{str(test_size)}/pnl.xlsx")







if __name__ == '__main__':
    # A = indicator_regressor(['Z26','Z16','Z11','Z10'])

    fatcor_list =  ['cangdan','cangdan_huanbi','member_oi','jicha_mom','kucun_tongbi','kucun_shuiwei','kucun_huanbi_week'
                        ,'kucun_huanbi_month','kucun_huanbi_diff_week','kucun_huanbi_diff_month',
              'rtn_skew','ret_skew','rtn_kurt','rv_umd','rtn_skew_std','ERR']
    A = indicator_regressor(fatcor_list,model_name='ew_pnl')
    A.load_pipeline(starttime="2020-01-04",endtime="2023-09-01",train_size=244*3,test_size=120,last_win=5,method='rolling')
    # A = indicator_regressor(factor_list=['rtn_dw','Z11','Z10','rtn_skew','rtn_skew_std','rtn_kurt','rv_umd','nos_gs','exRTN_maxVal'],model_name='lr')


    # A = indicator_regressor(factor_list=['Z10','Z11','rv_umd','rtn_skew_std'],model_name='lr')
    # A.load_pipeline(starttime="2018-01-04",endtime="2023-07-01",rolling_win=244,last_win=5,method='rolling') # ew,lr, xgb, adaboost




