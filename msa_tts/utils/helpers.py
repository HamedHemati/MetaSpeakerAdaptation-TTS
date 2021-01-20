import torch.optim as optim
import torch

from .wavernn.wavernn import WaveRNN


def get_wavernn(device, **params):
    r"""Intializes WaveRNN vocoder with the given params and returns it."""
    # Init model
    wavernn = WaveRNN(**params).to(device)

    # Load checkpoint
    state_dict = torch.load(params["checkpoint_path"], map_location=lambda storage, loc: storage)
    wavernn.load_state_dict(state_dict, strict=False)
    print("Loaded WaveRNN checkpoint.\n")
    
    return wavernn


def get_optimizer(model, **params):
    r"""Returns an optimizer with the specified parmeters."""
    Optimizer = getattr(optim, params["optimizer_name"])
    optim_params = {a:eval(b) for (a,b) in params["optim_params"].items()} 
    optimizer = Optimizer(model.parameters(), **optim_params)
    
    return optimizer
