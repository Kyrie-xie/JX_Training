'''
@Editor: Jinxing
@Description:
'''
import torch
import torch.nn as nn


def checkpoint_writer(model: nn.Module,
                      optimizer: torch.optim,
                      epoch: int,
                      save_dir: str
                      ) -> None:
    checkpoint = {'model': model,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoch}
    torch.save(checkpoint, '{}/checkpoint_{}.pkl'.format(save_dir,
                                                         epoch))


def load_checkpoint(filepath: str):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer = TheOptimizerClass()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    return model, optimizer