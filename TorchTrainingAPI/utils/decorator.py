'''
@Editor: Jinxing
@Description:
'''
'''
@Editor: Jinxing
@Description:
    一些装饰器
'''
import functools, time
from prettytable import PrettyTable

def time_consumption(func):
    '''
    装饰器：
        在结束func后print所用的时间
    '''
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0 #用时
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r'%(k,w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[using time: {}s]\t {}({})'.format(elapsed, name, arg_str))
        return result
    return clocked


def num_params(func):
    """
    这个decorator使用于定义network的__init__上边，用来在初始化的同时展示模型的参数。
        template: TechfinTorch.test.decorator_example.py
    :param func: DNN的__init__方程
    :return: 1.parameter的表格
    """
    @functools.wraps(func)
    def paramCount(*args, **kwargs):
        # 计算神经网络中的参数量
        result = func(*args, **kwargs)
        model = args[0]
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return result
    return paramCount


def main_flow(func):
    '''
    涵盖了time 和 try
    '''
    @functools.wraps(func)
    def flow(*args, **kwargs):
        t0 = time.time()
        name = func.__name__
        try:
            result = func(*args, **kwargs)
        except:
            raise Exception('In {} some errors occur!'.format(name))
        elapsed = time.time() - t0 #用时
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r'%(k,w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[using time: {}s]\t {}'.format(elapsed, name))
        return result
    return flow



if __name__ == '__main__':
    # Example
    @time_consumption
    def s(x,y):
        return x+y
    print( s(x = 3,y=4))