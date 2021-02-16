import torch
from .loss_landscapes.metrics.metric import Metric
from .loss_landscapes.model_interface.model_wrapper import ModelWrapper


class Tac2NVLossWrapper(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor, target_lens: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target
        self.target_lens = target_lens

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        model_out = model_wrapper.forward(self.inputs)
        return self.loss_fn(model_out, self.target, self.target_lens).item()


class Tac2NVWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__([model])

    def forward(self, x):
        return self.modules[0](**x)
    