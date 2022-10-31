import argparse
import logging.config
import random
from os import path, makedirs

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import reclib
from reclib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--job-dir', required=True, metavar='PATH', help='job directory where output is stored')
parser.add_argument('--dataset', required=True, metavar='PATH', help='path to the dataset')
parser.add_argument('--data-split', metavar='PATH', help='path to the file containing train/val/test indices')
parser.add_argument('--checkpoint', metavar='PATH', help='path to a checkpoint of the model')
parser.add_argument('--lr', metavar='N', default=0.01, type=float, help='learning rate')
parser.add_argument('--m', metavar='N', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--wd', metavar='N', default=0.001, type=float, help='weight decay')
parser.add_argument('--d-model', metavar='N', default=128, type=int, help='model dimensionality')
parser.add_argument('--ns', metavar='N', default=1, type=int, help='number of negative samples')
parser.add_argument('--batch-size', metavar='N', default=4096, type=int, help='batch size')
parser.add_argument('--epochs', metavar='N', default=100, type=int, help='number of epochs')
parser.add_argument('--val-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for validation')
parser.add_argument('--test-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for testing')
parser.add_argument('--seed', metavar='N', default=None, type=int, help='random state')


def main():
  logging.config.fileConfig('logging.ini')

  args = parser.parse_args()
  makedirs(args.job_dir, exist_ok=True)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  if args.seed:
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

  # drop rows with items and users that are not in the train set
  val_data = utils.data.drop_unavailable(train_data, val_data, col='user')
  val_data = utils.data.drop_unavailable(train_data, val_data, col='item')
  test_data = utils.data.drop_unavailable(train_data, test_data, col='user')
  test_data = utils.data.drop_unavailable(train_data, test_data, col='item')

  # remap ids of remaining users and items
  data = pd.concat([train_data, val_data, test_data])
  utils.data.remap_ids(data, col='user')
  utils.data.remap_ids(data, col='item')

  train_data = data.loc[train_data.index]
  val_data = data.loc[val_data.index]
  test_data = data.loc[test_data.index]

  print("data statistics:")
  print(f'train interactions: {len(train_data)}')
  print(f'val interactions: {len(val_data)} ({len(val_data) / original_val_size:.2%} available)')
  print(f'test interactions: {len(test_data)} ({len(test_data) / original_test_size:.2%} available)')
  print(f'users: {data["user"].nunique()}')
  print(f'items: {data["item"].nunique()}')

  train_dataset = InteractionDataset(train_data)
  val_iterator = UserIterator(val_data)

  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True)

  model = MF(
    num_users=train_data['user'].max() + 1,
    num_items=train_data['item'].max() + 1,
    embedding_dim=args.d_model).to(device)

  if args.checkpoint is not None:
    logging.info(f"loading model checkpoint from '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
  else:
    checkpoint = None

  if args.epochs > 0:
    optimizer = torch.optim.SGD(
      params=model.parameters(),
      lr=args.lr,
      momentum=args.m,
      weight_decay=args.wd)

    lr_schedule = reclib.optim.CosineLRSchedule(
      optimizer=optimizer,
      steps=args.epochs,
      eta_min=1e-7)

    bpr_loss = reclib.losses.BPRLoss(reduction='sum')

    if checkpoint is not None:
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_schedule.load_state_dict(checkpoint['lr_schedule'])
      initial_epoch = checkpoint['epoch']
    else:
      initial_epoch = 0

    val_rank_meter = utils.meter.SimpleMeter('val_rank')
    monitor = utils.meter.Monitor(val_rank_meter, cmp=np.less)

    checkpoint_path = path.join(args.job_dir, 'checkpoint.pth')
    best_checkpoint_path = path.join(args.job_dir, 'best_checkpoint.pth')

    for epoch in range(initial_epoch, args.epochs):
      lr_schedule.step()
      train_meters = reclib.optim.train_epoch(
        model=model,
        optimizer=optimizer,
        train_step=train_step(
          bpr_loss=bpr_loss,
          ns=args.ns),
        dataloader=train_loader,
        device=device)
      _, (val_ranks,) = reclib.optim.eval_epoch(
        model=model,
        metrics=[reclib.metrics.normalized_rank],
        eval_step=eval_step(),
        dataloader=val_iterator,
        device=device)
      val_rank_meter.update(val_ranks.mean().item())
      new_best_found = monitor.update()
      utils.save_checkpoint({'epoch': epoch + 1,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'lr_schedule': lr_schedule.state_dict()},
                            file=checkpoint_path, best=new_best_found)
      logging.info(f'[{epoch + 1:03d}] {"(*)" if new_best_found else "   "} '
                   f'{" ".join(map(str, train_meters))} {val_rank_meter}')

    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model'])

  print('--- evaluation ---')

  test_iterator = UserIterator(test_data)

  k = 10
  test_generator = torch.Generator(device=device)
  if args.seed is not None:
    test_generator.manual_seed(args.seed)

  test_event_ids, (test_ranks, test_HR) = reclib.optim.eval_epoch(
    model=model,
    metrics=[reclib.metrics.normalized_rank,
             lambda scores, targets: reclib.metrics.HR(
               *utils.draw_samples(scores, targets, ns=100, generator=test_generator), k=k)],
    eval_step=eval_step(),
    dataloader=test_iterator,
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


class MF(nn.Module):
  """Reference: https://arxiv.org/abs/1205.2618"""
  def __init__(self, num_users, num_items, embedding_dim):
    super().__init__()
    self.num_users = num_users
    self.num_items = num_items
    self.embedding_dim = embedding_dim
    self.user_emb = nn.Embedding(num_users, embedding_dim)
    self.item_emb = nn.Embedding(num_items, embedding_dim)

  def forward(self, user, pos_item, neg_items):
    user_emb = self.user_emb(user)
    pos_item_emb = self.item_emb(pos_item)
    neg_items_emb = self.item_emb(neg_items)
    pos_scores = torch.einsum('bd,bd->b', user_emb, pos_item_emb)
    neg_scores = torch.einsum('bd,bkd->bk', user_emb, neg_items_emb)
    return pos_scores, neg_scores

  @torch.no_grad()
  def recommend(self, user):
    user_emb = self.user_emb(user)
    scores = torch.einsum('bd,kd->bk', user_emb, self.item_emb.weight)
    return scores


class UserIterator:
  """Iterator over the item sequences of every user."""
  def __init__(self, events, user_col='user', item_col='item'):
    self.users = {}  # user_id -> ([ event_ids ], [ item_ids ])
    for user, user_events in events.groupby(user_col):
      events = torch.from_numpy(user_events.index.to_numpy()).long()
      items = torch.from_numpy(user_events[item_col].to_numpy()).long()
      self.users[user] = (events, items)

  def __iter__(self):
    for user, (events, items) in self.users.items():
      yield events, (torch.tensor(user), items)


class InteractionDataset(Dataset):
  """Dataset of user-item interactions."""
  def __init__(self, events, user_col='user', item_col='item'):
    self.events = torch.from_numpy(events.index.to_numpy()).long()
    self.user = torch.from_numpy(events[user_col].to_numpy()).long()
    self.item = torch.from_numpy(events[item_col].to_numpy()).long()

  def __len__(self):
    return len(self.events)

  def __getitem__(self, index):
    return self.events[index], (self.user[index], self.item[index])


def train_step(bpr_loss, ns):
  def train_step_fn(model, batch_data):
    user, item = batch_data
    B, = item.size()
    negative_samples = torch.randint(model.num_items, (B, ns), device=item.device)
    pos_scores, neg_scores = model(user, item, negative_samples)
    loss = bpr_loss(pos_scores.unsqueeze(1), neg_scores)
    return loss
  return train_step_fn


def eval_step():
  def eval_step_fn(model, batch_data):
    # since embeddings do not change during evaluation,
    #  we can recommend items in advance
    user, items = batch_data
    B, = items.size()
    scores = model.recommend(user.unsqueeze(0)).repeat(B, 1)
    return scores, items
  return eval_step_fn


if __name__ == '__main__':
  main()
