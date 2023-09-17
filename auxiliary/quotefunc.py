# -*- coding: utf-8 -*-
import os
import re
import sys
import glob
import logging
import datetime
import traceback
import numpy as np
import pandas as pd
import multiprocessing
try:
    import quotelog
    # import quotetime
    import Queue
except ImportError:
    from . import quotelog
    # from . import quotetime
    import queue as Queue


def date_converter(x):
    if isinstance(x, str):
        return datetime.datetime.strptime(x.replace('-', ''), '%Y%m%d')
    else:
        return x


def list_converter(x):
    if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, type(np.array)):
        return [v for v in x]
    else:
        return [x,]


def combinefile(source, groupfunc=None):
    '''
    合并h5或csv文件
    :param source: 目标目录
    :param groupfunc: 分组函数，参数包含文件名和dataframe数据
    :return: 没有分组函数直接返回{'all': dataframe}，有分组函数则返回 {groupname: dataframe}
    '''
    groups = {}
    h5files = glob.glob(os.path.join(source, '*.*'))
    for file in h5files:
        try:
            if file.lower().endswith('.h5'):
                dfdata = pd.read_hdf(file, mode='r')
            elif file.lower().endswith('.csv'):
                dfdata = pd.read_csv(file, mode='r')
            else:
                logging.warning('unrecognized file {0}'.format(file))
                continue

            if groupfunc is not None:
                filename = os.path.basename(file)
                groupname = groupfunc(filename, dfdata)
            else:
                groupname = 'all'
            if groupname not in groups:
                groups[groupname] = []
            dflist = groups[groupname]
            dflist.append(dfdata)
        except:
            logging.error(traceback.format_exc())

    combines = {groupname: pd.concat(dflist) for groupname, dflist in groups.items()}
    del groups
    return combines

#
# def quoterun(start, end, func, **kwargs):
#     '''
#     按交易日 day by day 执行任务
#     :param start: 开始日期
#     :param end: 结束日期
#     :param func: 调用函数
#     :param kwargs: 参数
#     :return:
#     '''
#     results = {}
#     curdate = start
#     while curdate <= end:
#         if not quotetime.is_tradingdate(curdate):
#             curdate = curdate + datetime.timedelta(days=1)
#             continue
#         logging.info(curdate)
#         kwargs['date'] = curdate
#         result = func(**kwargs)
#         if result != None:
#             results[curdate] = result
#         curdate = curdate + datetime.timedelta(days=1)
#     return results


def commonstart(mainfile=None):
    '''
    进程通用启动修饰，包含初始化log配置，异常退出时发通知等
    :param mainfile: 入口文件，通常用__file__
    :return:
    '''
    def inner1(func):
        def inner2(*args, **kwargs):
            try:
                if mainfile is not None:
                    dirname = os.path.dirname(mainfile)
                    filename = os.path.basename(mainfile)
                    sys.path.append(dirname)
                    quotelog.load_logconfig(filename.split('.')[0])
                else:
                    quotelog.load_logconfig()
                logging.info('start')
                func(*args, **kwargs)
                logging.info('end')
            except Exception as e:
                logging.fatal(traceback.format_exc())
                # raise
        return inner2
    return inner1


def _taskhandle(queue_task, queue_result, func):
    logging.debug('start process {0}'.format(os.getpid()))
    try:
        while not queue_task.empty():
            try:
                kwargs = queue_task.get(False)
                if kwargs is not None:
                    result = func(**kwargs)
                    if result is not None:
                        queue_result.put(result)
            except Queue.Empty:
                pass
            except:
                logging.error(traceback.format_exc())
    except:
        logging.fatal(traceback.format_exc())


def multask(tasks, func, process=0):
    '''
    启动多进程任务
    :param tasks: 任务参数列表
    :param func: 调用函数
    :param process: 进程数，默认逻辑核心数的一半
    :return:
    '''
    if process == 0:
        process = int(multiprocessing.cpu_count() / 2)
    pool = multiprocessing.Pool()
    queue_task = multiprocessing.Manager().Queue(len(tasks))
    queue_result = multiprocessing.Manager().Queue(len(tasks))

    for task in tasks:
        queue_task.put(task)
    for i in range(process):
        pool.apply_async(func=_taskhandle, args=(queue_task, queue_result, func))
    pool.close()
    pool.join()
    results = []
    while not queue_result.empty():
        results.append(queue_result.get())
    return results


def _test_multask(a, b):
    import time
    import random
    time.sleep(random.random())
    return (a, b)

def _test_quoteruntest(date, a, b):
    import time
    import random
    time.sleep(random.random())
    print(date, a, b)

if __name__ == '__main__':
    pass

    tasks = []
    for i in range(100):
        i = int(i)
        tasks.append({'a': i, 'b': i * i})

    print(multask(tasks, _test_multask))
    # quoterun(datetime.date(2019, 1, 1), datetime.date(2019, 3, 1), _test_quoteruntest, a=1, b=2)


