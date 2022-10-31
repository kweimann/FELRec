__all__ = ('load_data', 'read_ratings', 'download_data', 'check_dataset')

import shutil
from os import path

import pandas as pd

from reclib.dataset import utils


def load_data(data_dir, dataset='infer', download=True, rename_cols=True):
  """Reference: https://grouplens.org/datasets/movielens/"""
  if dataset == 'infer':
    dataset = path.basename(data_dir)
  check_dataset(dataset)
  # download data if it does not exist
  if not path.exists(data_dir):
    if download:
      dl_dir = download_data(dataset, extract_to=path.dirname(data_dir))
      shutil.move(dl_dir, data_dir)
    else:
      raise ValueError(f'Dataset does not exist: {data_dir}')
  # read dataframe
  if dataset == 'ml-1m':
    ratings = read_ratings(path.join(data_dir, 'ratings.dat'), dataset=dataset)
  elif dataset == 'ml-25m':
    ratings = read_ratings(path.join(data_dir, 'ratings.csv'), dataset=dataset)
  else:
    raise ValueError(f'Unsupported dataset: {dataset}')
  if rename_cols:
    ratings = ratings.rename(columns={
      'userId': 'user',
      'movieId': 'item'
    }).rename_axis('event_id')
  return ratings


def read_ratings(file, dataset='infer'):
  if dataset == 'infer':
    dataset = path.basename(path.dirname(file))
  check_dataset(dataset)
  if dataset == 'ml-1m':
    return pd.read_csv(
      file, sep='::', header=None, engine='python',
      names=['userId', 'movieId', 'rating', 'timestamp'])
  elif dataset == 'ml-25m':
    return pd.read_csv(
      file, header=0,
      names=['userId', 'movieId', 'rating', 'timestamp'])
  else:
    raise ValueError(f'Unsupported dataset: {dataset}')


def download_data(dataset, extract_to='.'):
  check_dataset(dataset)
  url = f'https://files.grouplens.org/datasets/movielens/{dataset}.zip'
  utils.download_and_unzip(url, extract_to=extract_to)
  return path.join(extract_to, dataset)


def check_dataset(dataset):
  if dataset not in ['ml-1m', 'ml-25m']:
    raise ValueError(f"Invalid dataset: '{dataset}'. "
                     f"Supported datasets: 'ml-1m', 'ml-25m'")
