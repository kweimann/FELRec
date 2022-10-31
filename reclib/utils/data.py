__all__ = ('split', 'remap_ids', 'drop_unavailable', 'lookup_table')

import torch


def split(events, val_size, test_size, divisible_by=None):
  n = len(events)
  val_size = int(n * val_size)
  test_size = int(n * test_size)
  val_end = n - test_size
  train_end = n - test_size - val_size
  if divisible_by is not None:
    train_start = train_end % divisible_by
  else:
    train_start = 0
  train = events.iloc[train_start:train_end]
  val = events.iloc[train_end:val_end]
  test = events.iloc[val_end:n]
  return train, val, test


def remap_ids(events, col, inplace=True, drop=False):
  if not inplace:
    events = events.copy()
  if not drop:
    events[f'_{col}'] = events[col]
  value_dict = {val: i for i, val in enumerate(events[col].unique(), start=1)}
  events[col] = events[col].map(value_dict)
  return events


def drop_unavailable(train, test, col):
  # drop rows with `col` values that exist in test but not in train
  train_col = set(train[col].unique())
  exists_in_train = test[col].isin(train_col)
  return test[exists_in_train]


def lookup_table(series, padding=0):
  """Build a tensor that acts as a lookup table for the series."""
  num_nodes = series.index.max() + 1
  index = series.index.to_numpy()
  values = torch.from_numpy(series.to_numpy())
  x = torch.empty(num_nodes, dtype=values.dtype).fill_(padding)
  x[index] = values
  return x
