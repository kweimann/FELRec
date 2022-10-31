__all__ = ('rank', 'normalized_rank', 'reciprocal_rank', 'NDCG', 'DCG', 'HR')

import torch


def rank(scores, targets, descending=True):
  """Return the rank of relevant item."""
  assert targets.ndim == 1
  relevance = _binary_relevance(scores, targets, descending=descending)
  return _rank(relevance)


def normalized_rank(scores, targets, descending=True):
  """Return the rank of relevant item normalized for the number of recommendations."""
  assert targets.ndim == 1
  relevance = _binary_relevance(scores, targets, descending=descending)
  return _normalized_rank(relevance)


def reciprocal_rank(scores, targets, descending=True):
  assert targets.ndim == 1
  relevance = _binary_relevance(scores, targets, descending=descending)
  return _reciprocal_rank(relevance)


def NDCG(scores, targets, k, descending=True):
  """Normalized discounted cumulative gain."""
  relevance = _binary_relevance(scores, targets, descending=descending)
  return _NDCG(relevance, k)


def DCG(scores, targets, k, descending=True):
  """Discounted cumulative gain."""
  relevance = _binary_relevance(scores, targets, descending=descending)
  return _DCG(relevance, k)


def HR(scores, targets, k, descending=True):
  """Hit rate."""
  assert k > 0
  relevance = _binary_relevance(scores, targets, descending=descending)
  return _HR(relevance, k)


def _rank(binary_relevance):
  _, indices = torch.nonzero(binary_relevance, as_tuple=True)
  return indices + 1


def _normalized_rank(binary_relevance):
  return (_rank(binary_relevance) - 1) / max(1, binary_relevance.size(1) - 1)


def _reciprocal_rank(binary_relevance):
  return _rank(binary_relevance).float().reciprocal()


def _NDCG(binary_relevance, k):
  assert k > 0
  return _DCG(binary_relevance, k) / _IDCG(binary_relevance, k)


def _DCG(binary_relevance, k):
  assert k > 0
  k = min(k, binary_relevance.size(1))
  i = torch.arange(1, k + 1, device=binary_relevance.device)
  nominator = binary_relevance[:, :k]
  denominator = torch.log2(i + 1.).unsqueeze(0)
  score = (nominator / denominator).sum(dim=1)
  return score


def _IDCG(binary_relevance, k):
  assert k > 0
  k = min(k, binary_relevance.size(1))
  i = torch.arange(1, k + 1, device=binary_relevance.device)
  k = torch.as_tensor(k, device=binary_relevance.device)
  score = torch.log2(i + 1.).reciprocal().cumsum(dim=0)
  k_relevant = torch.minimum(binary_relevance.sum(dim=1), k)
  return score[k_relevant - 1]


def _HR(binary_relevance, k):
  assert k > 0
  return binary_relevance[:, :k].any(dim=1)


def _binary_relevance(scores, targets, descending=True):
  """Return binary relevance of item at each rank."""
  assert scores.ndim == 2 and scores.dtype == torch.float
  if targets.ndim == 1:
    assert targets.dtype == torch.long \
           and targets.max() < scores.size(1)
  elif targets.ndim == 2:
    assert targets.dtype == torch.bool \
           and targets.shape == scores.shape
  else:
    raise AssertionError
  order = scores.argsort(dim=1, descending=descending)
  if targets.ndim == 1:
    relevance = targets.reshape(-1, 1) == order
  elif targets.ndim == 2:
    relevance = torch.gather(targets, 1, order)
  # noinspection PyUnboundLocalVariable
  return relevance
