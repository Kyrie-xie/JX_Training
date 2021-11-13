'''
@Editor: Jinxing
@Description:
'''
import collections

from TechfinTorchAPI.config import *
from .train_loop import TrainerBase



class SimpleTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 data_loader: DataLoader,
                 optimizer: optim,
                 loss_function: Callable[[t.tensor, t.tensor], t.tensor]
                 ):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self._data_loader = iter(data_loader)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.iteration_dict = collections.defaultdict(lambda : 0)

    def run_step(self):
        self.model.train()

        try:
            train_x, train_y = next(self._data_loader)
        except StopIteration:
            self._data_loader = iter(self.data_loader)
            train_x, train_y = next(self._data_loader)

        pred_y = self.model(train_x)

        loss = self.loss_function(pred_y, train_y)

        self.optimizer.zero_grad()

        self.iteration_dict['loss'] = loss.detach().cpu()
        
        self.iteration_dict['average_loss'] = self.iteration_dict['average_loss'] * 999/1000 + self.iteration_dict['loss']/1000

        loss.backward()
        
        self.optimizer.step()

        





