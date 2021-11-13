'''
@Editor: Jinxing
@Description:
'''
import pandas as pd
from TechfinDataAPI.system.pandas_system import PdSys
from typing import Union
import os

__all__ = ['pdsys_pkl']

def pdsys_pkl(data_sys: PdSys,
            dir: str):
    """
    将pd.Data数据保存成pkl方便后续的提取
    
    Args:
        data: pdSys格式，就会对backUp的所有数据进行保存，名字使用backUp_Note的名字
    """
    assert type(data_sys) == PdSys
    

    if not os.path.exists(dir):
        os.makedirs(dir)
    
    for i,(data, note) in enumerate(data_sys.backup_):
        try:
            file_name = '_'.join(note.split(' '))
            data.to_pickle(os.path.join(dir,file_name))
        except:
            raise Exception('backup {} goes wrong.'.format(i))
    
    print('finishes')
        

