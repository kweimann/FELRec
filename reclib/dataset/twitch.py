__all__ = ('load_data', 'read_streams')

from os import path, makedirs

import pandas as pd

from reclib.dataset import utils

BENCHMARK_FILE_ID = '13lbPDxlEI-ceKCR0H2a9glrgAOukV_Vt'


def load_data(file, download=True, rename_cols=True):
  """Reference: https://cseweb.ucsd.edu/~jmcauley/datasets.html"""
  # download data if it does not exist
  if not path.exists(file):
    if download:
      if path.basename(file) != '100k_a.csv':
        raise ValueError("Only the benchmark dataset '100k_a.csv' can be downloaded.")
      makedirs(path.dirname(path.abspath(file)), exist_ok=True)
      utils.download_file_from_google_drive(
        id=BENCHMARK_FILE_ID,
        destination=file)
    else:
      raise ValueError(f'Dataset does not exist: {file}')
  streams = read_streams(file)
  if rename_cols:
    streams = streams.rename(columns={
      'userId': 'user',
      'streamer': 'item',
      'start': 'timestamp'
    }).rename_axis('event_id')
  return streams


def read_streams(file):
  return pd.read_csv(
    file, header=0,
    names=['userId', 'streamId', 'streamer',
           'start', 'end'])
