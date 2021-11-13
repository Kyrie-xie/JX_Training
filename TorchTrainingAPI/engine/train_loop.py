'''
@Editor: Jinxing
@Description:
'''
import weakref
from typing import List, Optional

from TechfinTorchAPI.config import *
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

__all__ = ["HookBase", "TrainerBase", logging_redirect_tqdm]


class HookBase:
    """
    Hook中会对trainer使用弱引用来共享参数，然后在trainer中被调用。

    在trainer之中，
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step() #trainer中的训练步骤
            hook.after_step()
        iter += 1
        hook.after_train()
    Notes:
        Hook因为会调用到trainer中的参数，因此，在不同环境下中， Hook应当被定制
    """

    trainer: 'TrainerBase'  # 这个在TrainerBase的register_hooks中自己定义

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    针对后续的自定义的Trainer，只需要overwrite其中的run_step模块以及__init__模块
        example： .trainer.py/SimpleTrainer
    """

    def __init__(self):
        """
        一个关于Train Loop的基础框架

        ：note： 一个高级的框架需要另外定义Attributs:
            self.model: 深度学习模型
            self.optimizer
            self.loss_function
            self.T: 在训练的时候的iteration计数器，用于hook部分的计数与展示功能
            self.iteration_dict: 那些参数在进行iteration的时候将被记录下来。可以在Hook中定制

        """
        self.model: nn.Module
        self._hooks: List[HookBase] = []
        self.iter: int
        self.max_iter: int
        self.T: trange
        self.model: nn.Module
        self.optimizer: optim
        self.data_loader: DataLoader
        self.loss_function: Callable[[t.tensor], t.tensor]
        self.start_iter: int = 0
        self.iteration_dict: dict = {}
        self.writer = None

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self,
              max_iter: int,
              start_iter: int = 0, ):
        """
        Args:
            max_iter: int 末尾的epoch
            start_iter: int 开始的epoch
        """
        logger = get_logger(__name__)  # 在.config中
        if start_iter >= max_iter:
            logger.exception('The start epoch is greater than max epoch number')

        self.start_iter = start_iter
        self.max_iter = max_iter

        logger.info('start training from the epoch {}'.format(self.start_iter))
        try:
            with logging_redirect_tqdm():
                with trange(start_iter, max_iter, leave = False) as self.T:
                    self.before_train()
                    for self.iter in self.T:
                        self.before_step()
                        self.run_step()
                        self.after_step()
        except:
            logger.exception("Exception during training:")
            raise
        finally:
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()
        self.model.eval()
        if self.writer is not None: self.writer.close()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        for h in self._hooks:
            h.before_step()
        self.model.train()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError


# test
if __name__ == '__main__':
    pass
