import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as t



def scissors(data: pd.DataFrame,
             window_len: int,
             next_len: int=1,
             index_name = 'stock_code'):
    '''
    将一个dataFrame根据window_len在index上生成List of tensor
    '''
    assert isinstance(window_len, int)
    p = data.index.names.index(index_name)
    stocks = np.unique(data.index.get_level_values(p))
    res = []
    for stock in tqdm(stocks):
        data_temp = data.loc[stock]
        if len(data_temp)> window_len:
            i = window_len
            while i < len(data_temp):
                res.append(t.tensor(data_temp.iloc[i-window_len:i].to_numpy()).unsqueeze_(0))
                i+=next_len
    return res


def scissors_nonoverlap(data: pd.DataFrame,
             window_len: int,
             index_name = 'stock_code'):
    '''
    将一个dataFrame根据window_len在index上生成List of tensor
    '''
    assert isinstance(window_len, int)
    p = data.index.names.index(index_name)
    stocks = np.unique(data.index.get_level_values(p))
    res = []
    for stock in tqdm(stocks):
        data_temp = data.loc[stock]
        if len(data_temp)> window_len:
            i = window_len
            while i < len(data_temp):
                res.append(t.tensor(data_temp.iloc[i-window_len:i].to_numpy()).unsqueeze_(0))
                i+=window_len
    return res