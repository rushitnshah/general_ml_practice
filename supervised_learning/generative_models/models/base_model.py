from abc import abstractmethod
from typing import Union

import numpy as np
import torch as th


class BaseModel(th.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self, train_data, test_data, full_dataset, 
        nepochs: int = 10, eval_freq: int = 1, plot_at_eval=False):
        raise NotImplementedError
    
    @abstractmethod
    def estimate_loss(self, dataloader):
        raise NotImplementedError

    @abstractmethod
    def generate(self, x_true):
        raise NotImplementedError
    
    def _to_tensor(self, x: Union[th.Tensor, np.ndarray]):
        if isinstance(x, th.Tensor):
            return x.to(self.device)
        else:
            return th.tensor(x, device=self.device, dtype=th.float)
    
    def print_to_terminal(self, epoch, train_loss, test_loss):
        prnt_str = f"|  epoch  {epoch:5d}  "
        prnt_str += f"|  train loss {train_loss:.4f}  "
        prnt_str += f"|  test loss {test_loss:.4f}  |"
        print(prnt_str)