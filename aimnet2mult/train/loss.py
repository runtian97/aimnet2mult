import torch
from torch import Tensor
from typing import Dict, Any
from ..config import get_module
from functools import partial


class MTLoss:
    """ Multi-target loss function with fixed weights.

    This class allows for the combination of multiple loss functions, each with a specified weight.
    The weights are normalized to sum to 1. The loss functions are applied to the predictions and 
    true values, and the weighted sum of the losses is computed.

    Loss functions definition must contain keys:
        name (str): The name of the loss function.
        fn (str): The loss function (e.g. `aimnet2.train.loss.mse_loss_fn`).
        weight (float): The weight of the loss function.
        kwargs (Dict): Optional, additional keyword arguments for the loss function.

    Methods:
        __call__(y_pred, y_true):
            Computes the weighted sum of the losses from the individual loss functions.
            Args:
                y_pred (Dict[str, Tensor]): Predicted values.
                y_true (Dict[str, Tensor]): True values.
            Returns:
                Dict[str, Tensor]: total loss under key 'loss' and values for individual components.
    """

    def __init__(self, components: Dict[str, Any]):
        w_sum = sum(c['weight'] for c in components.values())
        self.components = dict()
        for name, c in components.items():
            kwargs = c.get('kwargs', dict())
            fn = partial(get_module(c['fn']), **kwargs)
            self.components[name] = (fn, c['weight'] / w_sum)

    def __call__(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss = dict()
        for name, (fn, w) in self.components.items():
            l = fn(y_pred=y_pred, y_true=y_true)
            loss[name] = l * w
        # special name for the total loss
        loss['loss'] = sum(loss.values())
        return loss
    

class MTAdaWLoss:
    """ Multi-target loss function with adaptive weights.

    This class allows for the combination of multiple loss functions, each with a specified weight.
    Scales of of the weights are adapted based on the loss values to match specified target weight.
    The scale of the first component is fixed and the scale of the other components are adjusted accordingly.
    The loss functions are applied to the predictions and true values, and the weighted sum of the losses is computed.

    Loss functions definition must contain keys:
        name (str): The name of the loss function.
        fn (str): The loss function (e.g. `aimnet2.train.loss.mse_loss_fn`).
        weight (float): The weight of the loss function.
        scale (float): The initial scale of the loss function.
        kwargs (Dict): Optional, additional keyword arguments for the loss function.

    Methods:
        __call__(y_pred, y_true):
            Computes the weighted sum of the losses from the individual loss functions.
            Args:
                y_pred (Dict[str, Tensor]): Predicted values.
                y_true (Dict[str, Tensor]): True values.
            Returns:
                Dict[str, Tensor]: total loss under key 'loss' and values for individual components.
    """

    def __init__(self, components: Dict[str, Any], eta=0.001):
        self.eta = eta
        s = sum(c['weight'] for c in components.values())
        self.components = dict()
        for name, c in components.items():
            kwargs = c.get('kwargs', dict())
            fn = partial(get_module(c['fn']), **kwargs)
            self.components[name] = [fn, c['weight'] / s, c['scale']]

    def __call__(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss = dict()
        for name, (fn, w, s) in self.components.items():
            l = fn(y_pred=y_pred, y_true=y_true)
            loss[name] = l * s
            loss[f'{name}_scale'] = s
        # special name for the total loss
        loss['loss'] = sum(loss[k] for k in self.components)

        # Update weights - batch the GPU->CPU transfer
        # Detach and compute ratios on GPU, then single transfer
        with torch.no_grad():
            total_loss = loss['loss'].detach()
            component_names = list(self.components.keys())[1:]  # Skip first
            if component_names:
                # Compute all ratios on GPU
                ratios = torch.stack([loss[name].detach() / total_loss for name in component_names])
                target_weights = torch.tensor([self.components[name][1] for name in component_names],
                                              device=ratios.device)
                # Single GPU->CPU transfer for comparison
                above_target = (ratios > target_weights).cpu().tolist()

                # Update scales based on comparison
                for i, name in enumerate(component_names):
                    mult = (1.0 - self.eta) if above_target[i] else (1.0 + self.eta)
                    self.components[name][2] *= mult

        return loss
def _get_sample_mask(y_true: Dict[str, Tensor], key_true: str, target: Tensor) -> Tensor:
    mask = y_true.get(f'{key_true}_mask')
    if mask is None:
        return None
    mask = mask.to(target)
    if mask.dim() == 0:
        mask = mask.view(1, 1)
    elif mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    return mask


def _apply_sample_mask(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.squeeze(-1)
    denom = mask.sum()
    # Avoid .item() - use torch.where for GPU-friendly conditional
    masked_sum = (values * mask).sum()
    # Return 0 if denom is 0, otherwise return masked mean
    return torch.where(denom > 0, masked_sum / denom, torch.zeros((), device=values.device, dtype=values.dtype))


def mse_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    """ General MSE loss function.
    Returns zero loss if the key is missing from y_true (masked out entirely).
    """
    if key_true not in y_true:
        return torch.zeros((), device=y_pred[key_pred].device, dtype=y_pred[key_pred].dtype)
    x = y_true[key_true]
    y = y_pred[key_pred]
    mask = _get_sample_mask(y_true, key_true, x)
    if mask is None:
        l = torch.nn.functional.mse_loss(x, y)
    else:
        diff2 = (x - y).pow(2).view(x.shape[0], -1).mean(dim=-1)
        l = _apply_sample_mask(diff2, mask)
    return l


def peratom_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    """ MSE loss function with per-atom normalization correction.
    Suitable when some of the values are zero both in y_pred and y_true due to padding of inputs.
    Returns zero loss if the key is missing from y_true or y_pred.
    """
    if key_true not in y_true or key_pred not in y_pred:
        # Key not present - return zero loss
        # Find any tensor to get device/dtype
        for v in y_pred.values():
            if isinstance(v, torch.Tensor):
                return torch.zeros((), device=v.device, dtype=v.dtype)
        return torch.zeros(())
    x = y_true[key_true]
    y = y_pred[key_pred]

    # Align shapes: squeeze trailing dimension if present in one but not the other
    if x.dim() > y.dim() and x.shape[-1] == 1:
        x = x.squeeze(-1)
    elif y.dim() > x.dim() and y.shape[-1] == 1:
        y = y.squeeze(-1)

    mask = _get_sample_mask(y_true, key_true, x)
    if y_pred['_natom'].numel() == 1:
        if mask is None:
            l = torch.nn.functional.mse_loss(x, y)
        else:
            per_sample = (x - y).pow(2).view(x.shape[0], -1).mean(dim=-1)
            l = _apply_sample_mask(per_sample, mask)
    else:
        diff2 = (x - y).pow(2).view(x.shape[0], -1)
        dim = diff2.shape[-1]
        scaled = diff2 * (y_pred['_natom'].unsqueeze(-1) / dim)
        per_sample = scaled.mean(dim=-1)
        if mask is None:
            l = per_sample.mean()
        else:
            l = _apply_sample_mask(per_sample, mask)
    return l


def energy_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'energy', key_true: str = 'energy') -> Tensor:
    """MSE loss normalized by the number of atoms.
    Returns zero loss if the key is missing from y_true (masked out entirely).
    """
    if key_true not in y_true:
        return torch.zeros((), device=y_pred[key_pred].device, dtype=y_pred[key_pred].dtype)
    x = y_true[key_true]
    y = y_pred[key_pred]
    s = y_pred['_natom'].sqrt()
    mask = _get_sample_mask(y_true, key_true, x)
    if mask is None:
        if y_pred['_natom'].numel() > 1:
            l = ((x - y).pow(2) / s).mean()
        else:
            l = torch.nn.functional.mse_loss(x, y) / s
    else:
        tmp_s = s
        while tmp_s.dim() < x.dim():
            tmp_s = tmp_s.unsqueeze(-1)
        scaled = ((x - y).pow(2) / tmp_s).view(x.shape[0], -1).mean(dim=-1)
        l = _apply_sample_mask(scaled, mask)
    return l


def dipole_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'charges', key_true: str = 'dipole') -> Tensor:
    x = y_true[key_true]
    # calculate dipole
    # detach coord, gradiant flows only to charges
    c = y_pred['coord'].detach()
    q = y_pred[key_pred]
    y = (c * q.unsqueeze(-1)).sum(-2)
    # in-place update y_pred for metrics
    y_pred[key_true] = y

    mask = _get_sample_mask(y_true, key_true, x)
    if y_pred['_natom'].numel() == 1:
        if mask is None:
            l = torch.nn.functional.mse_loss(x, y)
        else:
            per_sample = (x - y).pow(2).view(x.shape[0], -1).mean(dim=-1)
            l = _apply_sample_mask(per_sample, mask)
    else:
        diff2 = (x - y).pow(2).view(x.shape[0], -1)
        dim = diff2.shape[-1]
        scaled = diff2 * (y_pred['_natom'].unsqueeze(-1) / dim)
        per_sample = scaled.mean(dim=-1)
        if mask is None:
            l = per_sample.mean()
        else:
            l = _apply_sample_mask(per_sample, mask)
    return l


def quadrupole_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'charges', key_true: str = 'quadrupole') -> Tensor:
    x = y_true[key_true]
    # calculate quadrupole
    # detach coord, gradiant flows only to charges
    c = y_pred['coord'].detach()
    q = y_pred[key_pred]
    quad_ii = (c.pow(2) * q.unsqueeze(-1)).sum(-2)
    quad_ii = quad_ii - quad_ii.mean(dim=-1, keepdim=True)
    quad_ij = (c * c.roll(-1, -1) * q.unsqueeze(-1)).sum(-2)
    y = torch.cat([quad_ii, quad_ij], dim=-1)
    # in-place update y_pred for metrics
    y_pred[key_true] = y

    mask = _get_sample_mask(y_true, key_true, x)
    if y_pred['_natom'].numel() == 1:
        if mask is None:
            l = torch.nn.functional.mse_loss(x, y)
        else:
            per_sample = (x - y).pow(2).view(x.shape[0], -1).mean(dim=-1)
            l = _apply_sample_mask(per_sample, mask)
    else:
        diff2 = (x - y).pow(2).view(x.shape[0], -1)
        dim = diff2.shape[-1]
        scaled = diff2 * (y_pred['_natom'].unsqueeze(-1) / dim)
        per_sample = scaled.mean(dim=-1)
        if mask is None:
            l = per_sample.mean()
        else:
            l = _apply_sample_mask(per_sample, mask)
    return l
