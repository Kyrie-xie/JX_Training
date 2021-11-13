'''
一些常用的API，避免多余的加载
'''



####torch####
import torch as t
import torch

#model
from torch.nn import functional
from torch import nn
from torch import optim

#data
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# typing
from typing import Union, Tuple, List, Optional, Callable, Any, Dict

# logger
import logging


def get_logger(cfg):
    logging.basicConfig(format='%(asctime)s - [%(funcName)s] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    logger = logging.getLogger(cfg)
    return logger



__all__ = ['get_logger', 'data', 'DataLoader', 'Dataset', 'nn', 'optim', 't', 'torch', 'functional',\
           'Union', 'Tuple', 'List', 'Optional', 'Callable', 'Any']
