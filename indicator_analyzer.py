# -- coding: utf-8 --
"""
 @time : 2023/7/25
 @file : indicator_analyzer.py
 @author : zhenghao
 @software: PyCharm

根据因子生成的分组文件与分组信息直接生成对应的报告
"""
import glob
import os

import numpy as np
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Solve the minus sign problems

def hecheng_signal(df,last=5):
    res = pd.DataFrame()
    for i in range(last):
        res = res.add(df.shift(i),fill_value=0)
    res = res/last
    return res

def cal_icir(factor_group,factor_val,data,start_date,end_date):

    factor_group = pd.read_csv(factor_group,index_col=0,parse_dates=[0])
    factor_val = pd.read_csv(factor_val,index_col=0,parse_dates=[0])

    factor_group = factor_group.loc[start_date:end_date]
    factor_val = factor_val.loc[start_date:end_date]

    ret_dict = {}
    ic_dict = {}

    one_signal = factor_group[factor_group == 0].applymap(lambda x: 0 if np.isnan(x) else 1)
    two_signal = np.sign(factor_group[factor_group == 1]).fillna(0)
    three_signal = np.sign(factor_group[factor_group == 2]).fillna(0)
    four_signal = np.sign(factor_group[factor_group == 3]).fillna(0)
    five_signal = np.sign(factor_group[factor_group == 4]).fillna(0)

    for freq in data.keys():
        back_str= freq.split("-")[1]
        N = int(back_str[0])
        if 'Open' in back_str:
            df = data['Open-1DOpen']
        elif 'MorningOpen' in back_str:
            df = data['MorningOpen-1DMorningOpen']
        elif 'close' in back_str:
            df = data['Open-1DClose']
        else:
            pass


        one_signal = hecheng_signal(one_signal,N).replace(0,np.nan)
        two_signal = hecheng_signal(two_signal,N).replace(0,np.nan)
        three_signal = hecheng_signal(three_signal,N).replace(0,np.nan)
        four_signal = hecheng_signal(four_signal,N).replace(0,np.nan)
        five_signal = hecheng_signal(five_signal,N).replace(0,np.nan)

        one_group = (df*one_signal).mean(axis=1)
        two_group = (df*two_signal).mean(axis=1)
        three_group = (df*three_signal).mean(axis=1)
        four_group = (df*four_signal).mean(axis=1)
        five_group = (df*five_signal).mean(axis=1)

        IC_corr = factor_val.T.corrwith(df.T,method='pearson').dropna()
        Rankic_corr = factor_val.T.corrwith(df.T,method='spearman').dropna()
        total = pd.concat([one_group,two_group,three_group,four_group,five_group],axis=1)
        ret_dict[freq] = total
        ic_dict[freq] = pd.concat([IC_corr,Rankic_corr],axis=1)

    return ret_dict,ic_dict,factor_val

def plot(ret_dict_list,ic_dict_list,indicator_val_list,factor_name,factor_names):

    nowtime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    analysis_dict = {}
    if os.path.exists(f"report_Twap/{factor_name}"):
        pass
    else:
        os.makedirs(f"report_Twap/{factor_name}")
    out_file = f"report_Twap/{factor_name}/{nowtime}{factor_name}.pdf"
    pdf_pages = PdfPages(out_file)


    total_stat = []
    for k in range(len(ret_dict_list)):
        ret_dict = ret_dict_list[k]
        ic_dict = ic_dict_list[k]
        indicator_val = indicator_val_list[k]
        factor_attention = os.path.split(factor_names[k])[1][:-4]

        for freq, daliy_ret in ret_dict.items():
            print(factor_attention+freq)
            pnl_df = daliy_ret.cumsum() + 1
            IR_df = (daliy_ret.mean() * 244) / (daliy_ret.std() * np.sqrt(244))
            ret_df = daliy_ret.mean() * 244
            std_df = daliy_ret.std() * np.sqrt(244)

            buckets_key_stat = pd.concat([ret_df, std_df, IR_df], axis=1)
            buckets_key_stat = buckets_key_stat.reset_index()
            buckets_key_stat.columns = ['bucket', 'ret%', "std%", 'ir']
            buckets_key_stat['ret%'] = buckets_key_stat['ret%'].apply(lambda x: 100 * x)
            buckets_key_stat['std%'] = buckets_key_stat['std%'].apply(lambda x: 100 * x)

            # 计算max-min & first - last
            max_ir_id = buckets_key_stat['ir'].argmax()
            min_ir_id = buckets_key_stat['ir'].argmin()

            max_min_daily_ret = daliy_ret[max_ir_id] - daliy_ret[min_ir_id]
            first_last_daily_ret = daliy_ret[0] - daliy_ret[4]
            append_df = pd.concat([max_min_daily_ret, first_last_daily_ret], axis=1)
            append_df.columns = ['max-min', 'first-last']
            pnl_append_df = append_df.cumsum() + 1
            IR_append_df = (append_df.mean() * 244) / (append_df.std() * np.sqrt(244))
            ret_append_df = (append_df.mean() * 244)
            std_append_df = append_df.std() * np.sqrt(244)
            append_key_stat = pd.concat([ret_append_df, std_append_df, IR_append_df], axis=1)
            append_key_stat = append_key_stat.reset_index()
            append_key_stat.columns = ['bucket', 'ret%', "std%", 'ir']
            append_key_stat['ret%'] = append_key_stat['ret%'].apply(lambda x: 100 * x)
            append_key_stat['std%'] = append_key_stat['std%'].apply(lambda x: 100 * x)

            total_key_stat = round(pd.concat([buckets_key_stat, append_key_stat], axis=0), 2)
            pnl_total_df = pd.concat([pnl_df, pnl_append_df], axis=1)
            # pnl_total_df.to_csv(f"report_Twap/{factor_name}/{nowtime}{factor_attention}.csv")


            ## 下面计算IC
            ic_ts = ic_dict[freq].iloc[:,0]
            rankic_ts = ic_dict[freq].iloc[:,1]

            ic_ts = ic_ts.to_frame()
            ic_ts.columns = ['ic_ts']

            rankic_ts = rankic_ts.to_frame()
            rankic_ts.columns = ['rankic_ts']

            # 统计量保存
            items = {
                "多空收益%": round(pnl_total_df['first-last'].diff().mean() * 244, 3)*100,
                "多空IR": total_key_stat.loc[total_key_stat['bucket'] == 'first-last', 'ir'].values[0],
                "IC大于0比例": len(ic_ts[ic_ts > 0].dropna()) / len(ic_ts),
                'RANKIC大于0比例': len(rankic_ts[rankic_ts > 0].dropna()) / len(rankic_ts),
            }
            analysis_dict[freq] = items
            items = pd.DataFrame(items,index=[factor_attention+freq])
            total_stat.append(items)


            ## 画图
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes = axes.flatten()
            axes[0].axis("off")
            axes[0].set_title(f"{factor_attention}||{freq} IR分组统计")
            table = axes[0].table(cellText=total_key_stat.values, colLabels=total_key_stat.columns, loc='center')
            table.set_fontsize(12)
            table.auto_set_font_size(False)
            # table.set_fontstyle('italic')
            for i in range(len(total_key_stat.values) + 1):
                for j in range(4):
                    table[(i, j)].set_text_props(ha='center', va='center')
            table.scale(1, 2)

            axes[1].set_title("分组效果回测")
            pnl_total_df[pnl_total_df.columns.difference(['max-min', 'first-last'])].plot(ax=axes[1], legend=True)

            axes[2].set_title("多空/max_min回测")
            pnl_total_df[['max-min', 'first-last']].plot(ax=axes[2], legend=True)

            total_key_stat_plot = total_key_stat.iloc[:-2, :]
            axes[3].set_title(f"{freq} IR")
            total_key_stat_plot['ir'].plot(ax=axes[3], kind='bar')
            axes[4].set_title(f"{freq} RET")
            total_key_stat_plot['ret%'].plot(ax=axes[4], kind='bar')

            ic_ts['ic_ma'] = ic_ts['ic_ts'].rolling(5).mean()
            ic_ts['ic_cumsum'] = ic_ts['ic_ts'].cumsum()
            rankic_ts['rankic_ma'] = rankic_ts['rankic_ts'].rolling(5).mean()
            rankic_ts['rankic_cumsum'] = rankic_ts['rankic_ts'].cumsum()

            axes[5].set_title("因子hist分布图")
            axes[5].hist(indicator_val.values.flatten())
            axes[5].set_xlabel("因子值")
            axes[5].set_ylabel("出现频次")

            axes[6].set_title(f"{freq} IC")
            ic_ts['ic_ts'].plot(ax=axes[6], legend=True, alpha=0.3)
            ic_ts['ic_ma'].plot(ax=axes[6], legend=True)
            axes[6].axhline(y=0, ls='--', color='r')

            axes[7].set_title(f"{freq} IC")
            rankic_ts['rankic_ts'].plot(ax=axes[7], legend=True, alpha=0.3)
            rankic_ts['rankic_ma'].plot(ax=axes[7], legend=True)
            axes[7].axhline(y=0, ls='--', color='r')

            axes[8].set_title(f"{freq} IC/RANKIC 累计图")
            ic_ts['ic_cumsum'].plot(ax=axes[8], legend=True)
            rankic_ts['rankic_cumsum'].plot(ax=axes[8], legend=True)

            # plt.tight_layout()
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
            pdf_pages.savefig(fig)




           # # 整合信息也画一张图
        # ls_ret = {}
        # ls_ir = {}
        # ic_ratio = {}
        # rankic_ratio = {}
        # for freq, items in analysis_dict.items():
        #     ls_ret[freq] = items['多空收益%']
        #     ls_ir[freq] = items['多空IR']
        #     ic_ratio[freq] = items['IC大于0比例']
        #     rankic_ratio[freq] = items['RANKIC大于0比例']
        # ls_ret = pd.DataFrame(ls_ret, index=['多空收益%'])
        # ls_ir = pd.DataFrame(ls_ir, index=['多空IR'])
        # ic_ratio = pd.DataFrame(ic_ratio, index=['IC大于0比例'])
        # rankic_ratio = pd.DataFrame(rankic_ratio, index=['RANKIC大于0比例'])
        #
        # fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        # axes = axes.flatten()
        # ls_ret.T.plot(kind='bar', legend=True, ax=axes[0])
        # axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        # ls_ir.T.plot(kind='bar', legend=True, ax=axes[1])
        # axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        # ic_ratio.T.plot(kind='bar', legend=True, ax=axes[2])
        # axes[2].set_ylim(0, 1)
        # axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
        # axes[2].axhline(y=0.5, ls='--', color='r')
        # rankic_ratio.T.plot(kind='bar', legend=True, ax=axes[3])
        # axes[3].set_ylim(0, 1)
        # axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=45)
        # axes[3].axhline(y=0.5, ls='--', color='r')
        # pdf_pages.savefig(fig)
    pdf_pages.close()
    total_stat = pd.concat(total_stat,axis=0)
    total_stat = round(total_stat,2)
    total_stat.to_excel(f"report_Twap/{factor_name}/{factor_name}_stat.xlsx")


    # plt.show()
    pass

def generator_report(factor_name,data,start_date,end_date):
    factor_groups = glob.glob(f"factor_res/{factor_name}/{factor_name}_group_*.csv")
    factor_vals = glob.glob(f"factor_res/{factor_name}/{factor_name}_val_*.csv")
    ret_dict_list = []
    ic_dict_list = []
    factor_val_list = []
    for i in range(len(factor_groups)):
        ret_dict, ic_dict, factor_val = cal_icir(factor_groups[i],factor_vals[i],data,start_date,end_date)
        ret_dict_list.append(ret_dict)
        ic_dict_list.append(ic_dict)
        factor_val_list.append(factor_val)
    plot(ret_dict_list, ic_dict_list, factor_val_list,factor_name,factor_vals)


def get_total_stat(factor_class):
    res = []
    factor_list = glob.glob(f"report_Twap/{factor_class}*/*{factor_class}*_stat.xlsx")
    for file in factor_list:
        res.append(pd.read_excel(file,index_col=0))
    res = pd.concat(res,axis=0)
    res = res.sort_values(by='多空收益%',ascending=False)
    res.to_excel("total_look.xlsx")


if __name__ == '__main__':
    import multiprocessing as mp
    from functools import partial

    factor_rolling_method = {
        'rv_umd': 'MA-10',
        'rv_umd_u': 'MA-10',
        'rv_umd_d': 'MA-10',
        'rv_umd_ud': 'MA-10',
    }

    factors = list(factor_rolling_method.keys())


    freq_list = ['Open-1DOpen','Open-5DOpen','Open-1DClose','MorningOpen-1DMorningOpen',  'MorningOpen-5DMorningOpen', ]
    data = {}
    for k in freq_list:
        data[k] = pd.read_csv("ret_cache/" + k + ".csv", index_col=0, parse_dates=[0])

    generator_report_partial = partial(generator_report,data=data,start_date="2015-01-01",end_date='2022-01-01')
    pool = mp.Pool(processes=8)
    pool.map(generator_report_partial,factors)
    pool.close()
    pool.join()


    factor_class = 'rv_umd'
    get_total_stat(factor_class)
