__all__ = ('apply', 'map_to_device', 'is_sorted', 'draw_samples',
           'LoggingFormatter', 'save_checkpoint')

import logging
import shutil
from os import path

import numpy as np
import torch


def apply(f, o, recursive=True):
  if isinstance(o, dict):
    return {k: apply(f, v, recursive) if recursive else f(v) for k, v in o.items()}
  elif isinstance(o, list):
    return [apply(f, e, recursive) if recursive else f(e) for e in o]
  elif isinstance(o, tuple):
    return (apply(f, e, recursive) if recursive else f(e) for e in o)
  elif o is not None:
    return f(o)


def map_to_device(dataloader, device):
  def tensor_to_device(x): return x.to(device)
  def batch_to_device(x): return apply(tensor_to_device, x)
  dataloader = map(batch_to_device, dataloader)
  return dataloader


def is_sorted(array):
  return np.all(np.diff(array) >= 0)


def draw_samples(scores, targets, ns, generator=None):
  assert scores.size(0) == targets.size(0)
  B, C = scores.size()
  neg_items = torch.randint(
    C, (B, ns), device=scores.device, generator=generator)
  pos_scores = torch.gather(scores, 1, targets.unsqueeze(1))
  neg_scores = torch.gather(scores, 1, neg_items)
  scores = torch.cat([pos_scores, neg_scores], dim=1)
  targets = torch.zeros_like(targets)
  return scores, targets


class LoggingFormatter(logging.Formatter):
  def format(self, record):
    elapsed = int(record.relativeCreated)
    milliseconds = elapsed % 1000
    seconds = elapsed // 1000 % 60
    minutes = elapsed // (1000 * 60) % 60
    hours = elapsed // (1000 * 60 * 60)
    record.elapsed = f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'
    return super().format(record)


def save_checkpoint(state, file='checkpoint.pth', best=False, best_file=None):
  if best_file is None:
    best_file = path.join(path.dirname(file), f'best_checkpoint.pth')
  if 'epoch' in state:
    file = file.format(state['epoch'])
    best_file = best_file.format(state['epoch'])
  torch.save(state, file)
  if best:
    shutil.copyfile(file, best_file)
