#!/usr/miniconda/bin python3.8
# -*- encoding: utf-8 -*-
'''
@File    :   BT_system.py
@Time    :   2021/10/14 23:18:32
@Author  :   Jinxing 
@Version :   1.0
'''
from typing import Dict
import pandas as pd
import numpy as np
import torch as t
from torch import nn
from TechfinTorchAPI.dataloader.pandas_dataloader import PandasDataset
import sys,copy
sys.path.append('/root/persistence_data')
from logger import get_logger



class BtBase:
    matrics: Dict
    def __init__(self,
                 dataset: PandasDataset) -> None:
        '''
        可定制
        '''
        self.dataset = dataset
        
    def register(self,
                 matrics: Dict):
        self.matrics = matrics
    
    def __call__(self,
                 model: nn.Module) -> Dict:
        logger = get_logger(__name__)
        result_data = self._test(model)
        res = {}
        for key, func in self.matrics.items():
            try:
                res[key] = func(result_data)
            except:
                logger.debug('An error happens in the {} evaluation'.format(key))
        
        return res
    
    def _test(self,
              model: nn.Module) -> pd.DataFrame:
        '''
        数据的调用，预测与拼装。
            期望返回一个dataframe，供TechfinTorchAPI.matrics.back_testing里的调用
        '''
        pass


class BTSystem(BtBase):
    # 适用于截面模型(至少不需要滚动窗口)
    def __init__(self,
                 dataset: PandasDataset) -> None:
        self.dataset = dataset
    
    def _test(self,
             model: nn.Module) -> pd.DataFrame:
        data = copy.deepcopy(self.dataset.data)
        prediction = pd.DataFrame(np.zeros((len(data),1)), columns = ['prediction'], index = data.index)
        iterator = iter(self.dataset)
        while True:
            try:
                
                x = next(iterator)[0]   # 因为我们默认任何时候第一个输出的就是x，只输出x避免multi-task的时候error
                with t.no_grad():
                    pred = model(x).reshape(-1,1).cpu().numpy()
                # 提取该batch下的index
                item = self.dataset.item   
                index = self.dataset.index[item]  
                prediction.loc[index] = pred
                
            except StopIteration:
                break
        return data.join(prediction)


        
        
        

        
        
        




        
        
        
        
    
    
        