__all__ = ('BPRLoss',)

import torch
from torch import nn
from torch.nn import functional as F


class BPRLoss(nn.Module):
  def __init__(self, reduction='sum'):
    super().__init__()
    self.reduction = _get_reduction_by_name(reduction)

  def forward(self, pos_scores, neg_scores, sample_weight=None):
    sample_loss = -F.logsigmoid(pos_scores - neg_scores)
    if sample_weight is not None:
      # note that we do not consider weight when reducing the sample loss
      sample_loss = sample_weight * sample_loss
    return self.reduction(sample_loss.sum(dim=-1))


def _get_reduction_by_name(name):
  if name == 'sum':
    return torch.sum
  elif name == 'mean':
    return torch.mean
  elif name == 'none':
    def identity(x): return x
    return identity
  else:
    raise ValueError(f'Unknown reduction: {name}')
