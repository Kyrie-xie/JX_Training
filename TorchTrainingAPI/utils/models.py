'''
@Editor: Jinxing
@Description:
'''
from prettytable import PrettyTable

# Code
def paramCount(model):
    #计算神经网络中的参数量
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
