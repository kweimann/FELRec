__all__ = ('CosineLRSchedule', 'warmup_learning_rate', 'cosine_learning_rate',
           'train_epoch', 'eval_epoch')

import itertools
import math
import time

import torch
from torch import nn

from reclib import utils


class CosineLRSchedule:
  # Reference: https://arxiv.org/abs/1608.03983
  def __init__(self, optimizer, steps, warmup_steps=0,
               eta_min=0., eta_max=None, last_step=0):
    self.optimizer = optimizer
    self.steps = steps
    self.warmup_steps = warmup_steps
    self.eta_min = eta_min
    self.eta_max = eta_max
    if eta_max is None:
      for param_group in optimizer.param_groups:
        eta_max = param_group['lr']
        if self.eta_max is not None and eta_max != self.eta_max:
          raise ValueError('optimizer must use the same learning rate for all parameters')
        self.eta_max = eta_max
    self.last_step = last_step

  def step(self):
    warmup_steps = self.warmup_steps
    if self.last_step >= self.steps:
      raise ValueError('number of steps exceeded')
    if self.last_step < warmup_steps:
      lr = warmup_learning_rate(
        eta_min=self.eta_min,
        eta_max=self.eta_max,
        T_cur=self.last_step,
        T=warmup_steps)
    else:
      lr = cosine_learning_rate(
        eta_min=self.eta_min,
        eta_max=self.eta_max,
        T_cur=self.last_step - warmup_steps,
        T=max(1, self.steps - warmup_steps - 1))
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr
    self.last_step += 1

  def state_dict(self):
    return {'steps': self.steps,
            'warmup_steps': self.warmup_steps,
            'eta_min': self.eta_min,
            'eta_max': self.eta_max,
            'last_step': self.last_step}

  def load_state_dict(self, state_dict):
    self.steps = state_dict['steps']
    self.warmup_steps = state_dict['warmup_steps']
    self.eta_min = state_dict['eta_min']
    self.eta_max = state_dict['eta_max']
    self.last_step = state_dict['last_step']


def warmup_learning_rate(eta_min, eta_max, T_cur, T):
  return eta_min + (eta_max - eta_min) * (T_cur / T)


def cosine_learning_rate(eta_min, eta_max, T_cur, T):
  return eta_min + 0.5 * (eta_max - eta_min) * (1. + math.cos(math.pi * T_cur / T))


def train_epoch(model, optimizer, train_step, dataloader,
                steps_per_epoch=None, device=None):
  model.train()
  if steps_per_epoch is not None:
    dataloader = itertools.islice(dataloader, steps_per_epoch)
  if device is not None:
    dataloader = utils.map_to_device(dataloader, device)
  step_meter = utils.meter.AverageMeter('train_step')
  data_meter = utils.meter.AverageMeter('train_data')
  loss_meter = utils.meter.LossMeter('train_loss')
  step_start = time.time()
  for _, batch_data in dataloader:
    data_end = time.time()
    loss = train_step(model, batch_data)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    optimizer.step()
    loss_meter.update(loss.item())
    step_end = time.time()
    step_meter.update(step_end - step_start)
    data_meter.update(data_end - step_start)
    step_start = time.time()
  return [step_meter, data_meter, loss_meter]


@torch.no_grad()
def eval_epoch(model, metrics, eval_step, dataloader, device=None):
  model.eval()
  if device is not None:
    dataloader = utils.map_to_device(dataloader, device)
  event_ids = []
  metric_values = [[] for _ in range(len(metrics))]
  for batch_event_ids, batch_data in dataloader:
    eval_outputs = eval_step(model, batch_data)
    event_ids.append(batch_event_ids)
    for metric, values in zip(metrics, metric_values):
      batch_metric_values = metric(*eval_outputs)
      values.append(batch_metric_values)
  event_ids = torch.cat(event_ids)
  metric_values = [torch.cat(values) for values in metric_values]
  return event_ids, metric_values
