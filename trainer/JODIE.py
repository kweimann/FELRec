"""Based on the official repository: https://github.com/srijankr/jodie"""

import argparse
import logging.config
import random
from os import path, makedirs

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

import reclib
from reclib import utils
from trainer import GRU4Rec, FELRec

parser = argparse.ArgumentParser()
parser.add_argument('--job-dir', required=True, metavar='PATH', help='job directory where output is stored')
parser.add_argument('--dataset', required=True, metavar='PATH', help='path to the dataset directory')
parser.add_argument('--data-split', metavar='PATH', help='path to the file containing train/val/test indices')
parser.add_argument('--checkpoint', metavar='PATH', help='path to a checkpoint of the model')
parser.add_argument('--lr', metavar='N', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', metavar='N', default=0, type=float, help='weight decay')
parser.add_argument('--d-model', metavar='N', default=128, type=int, help='model dimensionality')
parser.add_argument('--val-ns', metavar='N', default=16384, type=int,
                    help='number of negative samples during validation')
parser.add_argument('--batch-size', metavar='N', default=1024, type=int, help='batch size')
parser.add_argument('--epochs', metavar='N', default=50, type=int, help='number of epochs')
parser.add_argument('--val-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for validation')
parser.add_argument('--test-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for testing')
parser.add_argument('--eval-drop-items', action='store_true', help='drop items that are not in the train set')
parser.add_argument('--seed', metavar='N', default=None, type=int, help='random state')


def main():
  logging.config.fileConfig('logging.ini')

  args = parser.parse_args()
  makedirs(args.job_dir, exist_ok=True)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  if args.seed is not None:
    logging.info(f'setting seed {args.seed}')
    torch.manual_seed(seed=args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

  parent_dir = path.dirname(args.dataset)
  if path.basename(args.dataset) in ['ml-1m', 'ml-25m']:
    from reclib.dataset import movielens
    load_data = movielens.load_data
  elif path.basename(parent_dir) == 'twitch':
    from reclib.dataset import twitch
    load_data = twitch.load_data
  elif path.basename(parent_dir) == 'jodie':
    from reclib.dataset import jodie
    if path.basename(args.dataset) == 'reddit.csv':
      load_data = jodie.reddit.load_data
    elif path.basename(args.dataset) == 'wikipedia.csv':
      load_data = jodie.wikipedia.load_data
    else:
      raise ValueError(f'Unknown JODIE dataset: {args.dataset}')
  else:
    raise ValueError(f'Unknown dataset: {args.dataset}')

  logging.info(f"loading data from '{args.dataset}'")
  data = load_data(args.dataset, download=True)
  data.sort_values(by=['timestamp', 'event_id'], inplace=True)

  # split data into train/val/test
  if args.data_split is None:
    train_data, val_data, test_data = utils.data.split(
      data, val_size=args.val_size, test_size=args.test_size)
  else:
    logging.info(f"loading data split from '{args.data_split}'")
    data_split = np.load(args.data_split)
    train_data = data.loc[data_split['train']]
    val_data = data.loc[data_split['val']]
    test_data = data.loc[data_split['test']]

  original_val_size = len(val_data)
  original_test_size = len(test_data)

  if args.eval_drop_items:
    val_data = utils.data.drop_unavailable(train_data, val_data, col='item')
    test_data = utils.data.drop_unavailable(train_data, test_data, col='item')

  # remap ids of remaining users and items
  data = pd.concat([train_data, val_data, test_data])
  utils.data.remap_ids(data, col='user')
  utils.data.remap_ids(data, col='item')
  data['_index'] = range(len(data))

  train_data = data.loc[train_data.index]
  val_data = data.loc[val_data.index]
  test_data = data.loc[test_data.index]

  print("data statistics:")
  print(f'train interactions: {len(train_data)}')
  print(f'val interactions: {len(val_data)} ({len(val_data) / original_val_size:.2%} available)')
  print(f'test interactions: {len(test_data)} ({len(test_data) / original_test_size:.2%} available)')
  print(f'users: {data["user"].nunique()}')
  print(f'items: {data["item"].nunique()}')

  dataset = SessionDataset(data)
  train_dataset = Subset(dataset, indices=train_data['_index'].to_numpy())
  val_dataset = Subset(dataset, indices=val_data['_index'].to_numpy())
  test_dataset = Subset(dataset, indices=test_data['_index'].to_numpy())

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=4)
  val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    num_workers=4)
  test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    num_workers=4)

  model = JODIE(
    num_users=data['user'].max() + 1,
    num_items=data['item'].max() + 1,
    d_model=args.d_model).to(device)

  optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=args.lr,
    weight_decay=args.wd)

  if args.checkpoint is not None:
    logging.info(f"loading model checkpoint from '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    initial_epoch = checkpoint['epoch']
  else:
    initial_epoch = 0

  if args.epochs > 0:
    val_rank_meter = utils.meter.SimpleMeter('val_rank')
    monitor = utils.meter.Monitor(val_rank_meter, cmp=np.less)

    checkpoint_path = path.join(args.job_dir, 'checkpoint.pth')
    best_checkpoint_path = path.join(args.job_dir, 'best_checkpoint.pth')

    for epoch in range(initial_epoch, args.epochs):
      model.reset_dynamic_embeddings()
      train_meters = reclib.optim.train_epoch(
        model=model,
        optimizer=optimizer,
        train_step=train_step(),
        dataloader=train_loader,
        device=device)
      # get model checkpoint after train epoch;
      #  after evaluation, we revert to this checkpoint because the model is updated during evaluation
      checkpoint = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
      # note: `existing_items_mask` is mutated during evaluation
      existing_items_mask = FELRec.create_mask(
        index=train_data['item'].unique(),
        size=model.num_items,
        device=device)
      _, (val_ranks,) = reclib.optim.eval_epoch(
        model=model,
        metrics=[reclib.metrics.normalized_rank],
        eval_step=eval_step(
          existing_items_mask=existing_items_mask,
          optimizer=optimizer,
          ns=args.val_ns),
        dataloader=val_loader,
        device=device)
      # save checkpoint and log performance metrics
      mean_val_rank = val_ranks.mean().item()
      val_rank_meter.update(mean_val_rank)
      new_best_found = monitor.update()
      utils.save_checkpoint({'epoch': epoch + 1,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            file=checkpoint_path, best=new_best_found)
      logging.info(f'[{epoch + 1:03d}] {"(*)" if new_best_found else "   "} '
                   f'{" ".join(map(str, train_meters))} {val_rank_meter}')
      # revert model back to the checkpoint before evaluation
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])

    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

  print('--- evaluation ---')

  # note: `existing_items_mask` is mutated during evaluation
  existing_items_mask = FELRec.create_mask(
    index=pd.concat([train_data, val_data])['item'].unique(),
    size=model.num_items,
    device=device)

  k = 10
  test_generator = torch.Generator(device=device)
  if args.seed is not None:
    test_generator.manual_seed(args.seed)

  test_event_ids, (test_ranks, test_HR) = reclib.optim.eval_epoch(
    model=model,
    metrics=[reclib.metrics.normalized_rank,
             lambda scores, targets: reclib.metrics.HR(
               *utils.draw_samples(scores, targets, ns=100, generator=test_generator), k=k)],
    eval_step=eval_step(
      existing_items_mask=existing_items_mask,
      optimizer=optimizer),
    dataloader=test_loader,
    device=device)
  test_event_ids = test_event_ids.cpu().numpy()

  recommendations = pd.DataFrame(
    index=test_data.index,
    columns=['rank', f'HR@{k}'])
  recommendations.loc[test_event_ids, 'rank'] = test_ranks.cpu().numpy()
  recommendations.loc[test_event_ids, f'HR@{k}'] = test_HR.cpu().numpy()
  recommendations.to_csv(path.join(args.job_dir, 'recommendations.csv'))

  logging.info(f'test_rank {test_ranks.mean().item():.4f} '
               f'test_HR@{k} {test_HR.float().mean().item():.4f}')


class JODIE(nn.Module):
  """Reference: https://snap.stanford.edu/jodie/"""
  def __init__(self, num_users, num_items, d_model):
    super().__init__()
    self.d_model = d_model
    self.num_users = num_users
    self.num_items = num_items
    self.register_buffer('dynamic_user_emb', torch.zeros(num_users, d_model))
    self.register_buffer('dynamic_item_emb', torch.zeros(num_users, d_model))
    self.register_buffer('is_user_new', torch.ones(num_users, 1))
    self.register_buffer('is_item_new', torch.ones(num_items, 1))
    self.static_user_emb = nn.Embedding(num_users, d_model)
    self.static_item_emb = nn.Embedding(num_items, d_model)
    self.initial_user_emb = nn.Parameter(torch.randn(1, d_model))
    self.initial_item_emb = nn.Parameter(torch.randn(1, d_model))
    self.user_rnn = nn.RNNCell(d_model + 1, d_model)
    self.item_rnn = nn.RNNCell(d_model + 1, d_model)
    # user_emb -> prev_item_emb -> prev_static_item_emb -> static_user_emb => item_emb -> static_item_emb
    self.prediction = nn.Linear(4 * d_model, 2 * d_model)
    self.timedelta_emb = nn.Linear(1, d_model)  # embedding of elapsed time between interactions

  @torch.no_grad()
  def reset_dynamic_embeddings(self):
    self.dynamic_user_emb[:] = 0.
    self.dynamic_item_emb[:] = 0.
    self.is_user_new[:] = 1.
    self.is_item_new[:] = 1.

  def forward(self, user_data, item_data):  # note that we do not use features!
    user_id, prev_item_id, time_since_prev_item = user_data
    item_id, time_since_prev_user = item_data
    # get dynamic embeddings; use initial embeddings if dynamic do not exist yet
    user_emb = self.is_user_new[user_id] * self.initial_user_emb + self.dynamic_user_emb[user_id]
    item_emb = self.is_item_new[item_id] * self.initial_item_emb + self.dynamic_item_emb[item_id]
    prev_item_emb = self.is_item_new[prev_item_id] * self.initial_item_emb + self.dynamic_item_emb[prev_item_id]
    # get static embeddings
    static_user_emb = self.static_user_emb(user_id)
    static_item_emb = self.static_item_emb(item_id)
    prev_static_item_emb = self.static_item_emb(prev_item_id)
    # project user embedding to current time
    user_proj = user_emb * (1. + self.timedelta_emb(time_since_prev_item))
    # predict item embedding for the user
    item_pred = self.prediction(torch.cat([
      user_proj, prev_item_emb, prev_static_item_emb, static_user_emb], dim=1))
    item_target = torch.cat([item_emb, static_item_emb], dim=1)
    # update embeddings
    updated_user_emb = self.user_rnn(
      torch.cat([item_emb, time_since_prev_item], dim=1), user_emb)
    updated_item_emb = self.item_rnn(
      torch.cat([user_emb, time_since_prev_user], dim=1), item_emb)
    # store updated embeddings
    self.dynamic_user_emb[user_id] = updated_user_emb.detach()
    self.dynamic_item_emb[item_id] = updated_item_emb.detach()
    self.is_user_new[user_id] = 0.
    self.is_item_new[item_id] = 0.
    return (item_pred, item_target), \
           (updated_user_emb, user_emb), \
           (updated_item_emb, item_emb)


class SessionDataset(GRU4Rec.SessionDataset):  # note that we do not use timestamp information!
  def __init__(self, events, user_col='user', item_col='item'):
    super().__init__(events, session_len=1, user_col=user_col, item_col=item_col)
    self.no_timedelta = torch.tensor([0.])

  def __getitem__(self, index):
    event, (user, input_item, target_item, _) = super().__getitem__(index)
    user_data = (user, input_item.squeeze(), self.no_timedelta)
    item_data = (target_item.squeeze(), self.no_timedelta)
    return event, (user_data, item_data)


def train_step():
  def train_step_fn(model, batch_data):
    user_data, item_data = batch_data
    ((item_pred, item_target),
     (updated_user_emb, user_emb),
     (updated_item_emb, item_emb)) = model(user_data, item_data)
    # backpropagation on the static embeddings of the user and their previous item,
    #  the parameters of prediction layer and RNNs,
    #  and the initial embedding only if either user or item are new;
    # note that we normalize the embeddings only before computing the loss or the recommendation scores
    loss = (F.mse_loss(_norm(item_pred), _norm(item_target.detach()))
            + F.mse_loss(_norm(updated_user_emb), _norm(user_emb.detach()))
            + F.mse_loss(_norm(updated_item_emb), _norm(item_emb.detach())))
    return loss
  return train_step_fn


def eval_step(existing_items_mask, optimizer, ns=None):
  @torch.enable_grad()
  def eval_step_fn(model, batch_data):
    if not model.training:
      model.train()
    user_data, item_data = batch_data
    user, _, _ = user_data
    item, _ = item_data
    num_items = existing_items_mask.sum()
    # get indices of existing items
    if ns is not None and ns < num_items - len(item):  # limit the number of negative samples
      existing_items_mask[item] = 0.  # hide items from this batch (to avoid sampling them)
      negative_samples = FELRec.draw_samples_from_mask(existing_items_mask, ns, replace=False)
      existing_items_mask[item] = 1.  # add items from this batch to the sample pool
      existing_items = torch.cat([item, negative_samples])
    else:
      existing_items_mask[item] = 1.  # add items from this batch to the sample pool
      existing_items, = existing_items_mask.nonzero(as_tuple=True)
    # compute targets while adjusting for the existing items
    targets = torch.zeros(model.num_items, dtype=torch.long, device=existing_items_mask.device)
    targets[existing_items] = torch.arange(len(existing_items), device=existing_items_mask.device)
    targets = targets[item]
    # grab embeddings of existing items
    existing_items_embeddings = torch.cat([
      model.dynamic_item_emb[existing_items],
      model.static_item_emb(existing_items).detach()], dim=1)
    # update embeddings of user and item
    ((item_pred, item_target),
     (updated_user_emb, user_emb),
     (updated_item_emb, item_emb)) = model(user_data, item_data)
    # recommend existing items to the user (we use pre-update values)
    scores = torch.einsum(
      'qd,kd->qk',
      _norm(item_pred.detach()),
      _norm(existing_items_embeddings))
    # update model parameters during evaluation
    loss = (F.mse_loss(_norm(item_pred), _norm(item_target.detach()))
            + F.mse_loss(_norm(updated_user_emb), _norm(user_emb.detach()))
            + F.mse_loss(_norm(updated_item_emb), _norm(item_emb.detach())))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    optimizer.step()
    return scores, targets
  return eval_step_fn


def _norm(x): return F.normalize(x, dim=1, p=2)


if __name__ == '__main__':
  main()
