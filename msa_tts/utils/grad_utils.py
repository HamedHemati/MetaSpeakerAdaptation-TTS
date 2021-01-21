#################################################
# Adapted from https://github.com/sungyubkim/GBML
#################################################

import torch


def apply_grad(model, grad):
    '''
    Assign gradient to model(nn.Module) instance. Return the norm of gradient.
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()


def mix_grad(grad_list, weight_list):
    '''
    Calculate weighted average of gradients.
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad

