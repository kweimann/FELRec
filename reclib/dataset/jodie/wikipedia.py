from os import path, makedirs
from urllib.request import urlretrieve

import pandas as pd


def load_data(file, download=True):
  """https://snap.stanford.edu/jodie/#datasets"""
  if not path.exists(file):
    if download:
      makedirs(path.dirname(file), exist_ok=True)
      urlretrieve('http://snap.stanford.edu/jodie/wikipedia.csv', file)
    else:
      raise ValueError(f'Dataset does not exist: {file}')
  # read dataframe: we skip the header because it does not account for the feature list
  interactions = pd.read_csv(file, header=None, skiprows=1)
  # we ignore features and state changes
  interactions = interactions[[0, 1, 2]].rename(columns={
    0: 'user',
    1: 'item',
    2: 'timestamp'
  }).rename_axis('event_id')
  return interactions
