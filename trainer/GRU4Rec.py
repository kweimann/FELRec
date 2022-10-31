import argparse
import heapq
import logging.config
import random
from os import makedirs, path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import reclib
from reclib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--job-dir', required=True, metavar='PATH', help='job directory where output is stored')
parser.add_argument('--dataset', required=True, metavar='PATH', help='path to the dataset')
parser.add_argument('--data-split', metavar='PATH', help='path to the file containing train/val/test indices')
parser.add_argument('--checkpoint', metavar='PATH', help='path to a checkpoint of the model')
parser.add_argument('--lr', metavar='N', default=0.001, type=float, help='learning rate')
parser.add_argument('--m', metavar='N', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--wd', metavar='N', default=0, type=float, help='weight decay')
parser.add_argument('--d-model', metavar='N', default=128, type=int, help='model dimensionality')
parser.add_argument('--layers', metavar='N', default=3, type=int, help='number of Transformer layers')
parser.add_argument('--ns', metavar='N', default=8, type=int, help='number of negative samples')
parser.add_argument('--val-ns', metavar='N', default=16384, type=int,
                    help='number of negative samples during validation')
parser.add_argument('--dropout', metavar='N', default=0.1, type=float, help='dropout rate')
parser.add_argument('--session', metavar='N', default=1, type=int,
                    help='session length. If session length > 1, then the model is trained using BPTT.')
parser.add_argument('--batch-size', metavar='N', default=1024, type=int, help='batch size')
parser.add_argument('--epochs', metavar='N', default=50, type=int, help='number of epochs')
parser.add_argument('--steps-per-epoch', metavar='N', default=None, type=int, help='number of steps per epoch')
parser.add_argument('--val-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for validation')
parser.add_argument('--test-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for testing')
parser.add_argument('--seed', metavar='N', default=None, type=int, help='random state')


def main():
  logging.config.fileConfig('logging.ini')

  args = parser.parse_args()
  makedirs(args.job_dir, exist_ok=True)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  use_bptt = args.session > 1

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

  # drop rows with items that are not in the train set
  val_data = utils.data.drop_unavailable(train_data, val_data, col='item')
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

  # session iterators for evaluation
  train_session_iterator = TimeOrderedSessionIterator(train_data, batch_size=args.batch_size)
  val_session_iterator = TimeOrderedSessionIterator(val_data, batch_size=args.batch_size)

  num_items = train_data['item'].max() + 1
  num_users = data['user'].max() + 1

  if use_bptt:
    # noinspection PyTypeChecker
    model = GRU4Rec(
      num_items=num_items,
      hidden_size=args.d_model,
      num_layers=args.layers,
      use_embedding=True,
      output_activation=nn.Identity,
      dropout=args.dropout,
      input_dropout=args.dropout)
  else:
    model = GRU4Rec(
      num_items=num_items,
      hidden_size=args.d_model,
      num_layers=args.layers,
      dropout=args.dropout)
  session_model = SessionAwareWrapper(model, num_users=num_users).to(device)

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

    # note: change reduction method together with the learning rate to influence the training speed;
    #  training speed should be adjusted depending on the dataset to achieve optimal results
    bpr_loss = reclib.losses.BPRLoss(reduction='sum')

    if checkpoint is not None:
      optimizer.load_state_dict(checkpoint['optimizer'])
      initial_epoch = checkpoint['epoch']
    else:
      initial_epoch = 0

    if use_bptt:
      logging.info('training GRU4Rec with Backpropagation Through Time (BPTT)')
      train_session_dataset = SessionDataset(
        train_data, session_len=args.session)
      train_session_loader = DataLoader(
        train_session_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)
    else:
      train_session_loader = RandomOrderSessionIterator(
        train_data, batch_size=args.batch_size)

    val_rank_meter = utils.meter.SimpleMeter('val_rank')
    monitor = utils.meter.Monitor(val_rank_meter, cmp=np.less)

    checkpoint_path = path.join(args.job_dir, 'checkpoint.pth')
    best_checkpoint_path = path.join(args.job_dir, 'best_checkpoint.pth')

    for epoch in range(initial_epoch, args.epochs):
      if use_bptt:
        train_meters = reclib.optim.train_epoch(
          model=model,
          optimizer=optimizer,
          train_step=train_step_bptt(
            bpr_loss=bpr_loss,
            ns=args.ns),
          dataloader=train_session_loader,
          steps_per_epoch=args.steps_per_epoch,
          device=device)
        session_model.reset_sessions()
        reclib.optim.eval_epoch(
          model=session_model,
          metrics=[reclib.metrics.normalized_rank],
          eval_step=eval_step(ns=args.val_ns),
          dataloader=train_session_iterator,
          device=device)
      else:
        session_model.reset_sessions()
        train_meters = reclib.optim.train_epoch(
          model=session_model,
          optimizer=optimizer,
          train_step=train_step(
            bpr_loss=bpr_loss,
            ns=args.ns),
          dataloader=train_session_loader,
          device=device)
      _, (val_ranks,) = reclib.optim.eval_epoch(
        model=session_model,
        metrics=[reclib.metrics.normalized_rank],
        eval_step=eval_step(ns=args.val_ns),
        dataloader=val_session_iterator,
        device=device)
      val_rank_meter.update(val_ranks.mean().item())
      new_best_found = monitor.update()
      utils.save_checkpoint({'epoch': epoch + 1,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            file=checkpoint_path, best=new_best_found)
      logging.info(f'[{epoch + 1:03d}] {"(*)" if new_best_found else "   "} '
                   f'{" ".join(map(str, train_meters))} {val_rank_meter}')

    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    session_model.reset_sessions()

  print('--- evaluation ---')

  test_session_iterator = TimeOrderedSessionIterator(test_data, batch_size=args.batch_size)

  k = 10
  test_generator = torch.Generator(device=device)
  if args.seed is not None:
    test_generator.manual_seed(args.seed)

  _, (train_ranks,) = reclib.optim.eval_epoch(
    model=session_model,
    metrics=[reclib.metrics.normalized_rank],
    eval_step=eval_step(),
    dataloader=train_session_iterator,
    device=device)
  _, (val_ranks,) = reclib.optim.eval_epoch(
    model=session_model,
    metrics=[reclib.metrics.normalized_rank],
    eval_step=eval_step(),
    dataloader=val_session_iterator,
    device=device)
  test_event_ids, (test_ranks, test_HR) = reclib.optim.eval_epoch(
    model=session_model,
    metrics=[reclib.metrics.normalized_rank,
             lambda scores, targets: reclib.metrics.HR(
               *utils.draw_samples(scores, targets, ns=100, generator=test_generator), k=k)],
    eval_step=eval_step(),
    dataloader=test_session_iterator,
    device=device)
  test_event_ids = test_event_ids.cpu().numpy()

  recommendations = pd.DataFrame(
    index=test_data.index,
    columns=['rank', f'HR@{k}'])
  recommendations.loc[test_event_ids, 'rank'] = test_ranks.cpu().numpy()
  recommendations.loc[test_event_ids, f'HR@{k}'] = test_HR.cpu().numpy()
  recommendations.to_csv(path.join(args.job_dir, 'recommendations.csv'))

  logging.info(f'train_rank {train_ranks.mean().item():.4f} '
               f'val_rank {val_ranks.mean().item():.4f} '
               f'test_rank {test_ranks.mean().item():.4f} '
               f'test_HR@{k} {test_HR.float().mean().item():.4f}')


class GRU4Rec(nn.Module):
  """Reference: https://arxiv.org/pdf/1511.06939.pdf"""
  def __init__(self, num_items, hidden_size, num_layers, use_embedding=False,
               output_activation=nn.Tanh, reuse_emb=True, dropout=0., input_dropout=0.):
    super().__init__()
    self.num_items = num_items
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.reuse_emb = use_embedding and reuse_emb
    if use_embedding:
      self.embedding = nn.Embedding(num_items, hidden_size)
      self.input_dropout = nn.Dropout(input_dropout)
      input_size = hidden_size
    else:
      self.embedding = OneHotEmbedding(num_items)
      self.input_dropout = nn.Identity()
      input_size = num_items
    self.gru = nn.GRU(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout)
    if self.reuse_emb:
      self.fc = nn.Linear(hidden_size, hidden_size)
    else:
      self.fc = nn.Linear(hidden_size, num_items)
    self.out_act = output_activation()

  def forward(self, input_items, pos_items, neg_items, hidden=None):
    x = self.embedding(input_items)  # (B, N, D)
    x = self.input_dropout(x)
    x, hidden = self.gru(x, hidden)  # (B, N, D), (num_layers, B, D)
    if self.reuse_emb:
      x = self.fc(x)                 # (B, N, D)
      pos_scores = self.out_act(torch.einsum(
        'bnd,bnd->bn', x, self.embedding(pos_items))).unsqueeze(2)
      neg_scores = self.out_act(torch.einsum(
        'bnd,bnkd->bnk', x, self.embedding(neg_items)))
    else:
      scores = self.out_act(self.fc(x))  # (B, N, C)
      pos_scores = torch.gather(scores, 2, pos_items.unsqueeze(2))
      neg_scores = torch.gather(scores, 2, neg_items)
    return (pos_scores, neg_scores), hidden

  @torch.no_grad()
  def recommend(self, input_items, hidden=None, selected_items=None, only_last=False):
    x = self.embedding(input_items)  # (B, N, D)
    x = self.input_dropout(x)
    x, hidden = self.gru(x, hidden)  # (B, N, D), (num_layers, B, D)
    if only_last and x.size(1) > 1:
      assert selected_items is None or selected_items.size(1) == 1  # (B, 1, ns)
      x = x[:, -1].unsqueeze(1)  # (B, 1, D)
    if self.reuse_emb:
      if selected_items is None:
        scores = self.out_act(torch.einsum(
          'bnd,kd->bnk', self.fc(x), self.embedding.weight))
      else:
        scores = self.out_act(torch.einsum(
          'bnd,bnkd->bnk', self.fc(x), self.embedding(selected_items)))
    else:
      scores = self.out_act(self.fc(x))
      if selected_items is not None:
        scores = torch.gather(scores, 2, selected_items)
    return scores, hidden


class SessionAwareWrapper(nn.Module):
  def __init__(self, model, num_users):
    super().__init__()
    self.model = model
    self.num_users = num_users
    self.num_items = model.num_items
    user_sessions = torch.zeros(num_users, model.num_layers, model.hidden_size)
    self.register_buffer('user_sessions', user_sessions, persistent=False)

  @torch.no_grad()
  def reset_sessions(self, users=None):
    self.user_sessions[users] = 0.

  def forward(self, user, input_item, pos_items, neg_items, update=True):
    sessions = self.user_sessions[user].transpose(0, 1).contiguous()  # (num_layers, B, D)
    scores, sessions = self.model(
      input_item.unsqueeze(1), pos_items.unsqueeze(1),
      neg_items.unsqueeze(1), hidden=sessions)
    if update:
      self.user_sessions[user] = sessions.detach().transpose(0, 1)
    return scores  # (B, 1, 1), (B, 1, ns)

  @torch.no_grad()
  def recommend(self, user, input_item, selected_items=None, update=True):
    if selected_items is not None:
      selected_items = selected_items.unsqueeze(1)  # (B, 1, ns)
    sessions = self.user_sessions[user].transpose(0, 1).contiguous()  # (num_layers, B, D)
    scores, sessions = self.model.recommend(
      input_item.unsqueeze(1), hidden=sessions, selected_items=selected_items)
    if update:
      self.user_sessions[user] = sessions.detach().transpose(0, 1)
    return scores  # (B, 1, C)


class OneHotEmbedding(nn.Module):
  def __init__(self, num_embeddings):
    super().__init__()
    self.num_embeddings = num_embeddings

  def forward(self, x):
    return F.one_hot(x, num_classes=self.num_embeddings).float()


class SessionDataset(Dataset):
  """Session data of every event in the dataset."""
  def __init__(self, events, session_len, user_col='user', item_col='item'):
    if session_len <= 0:
      raise ValueError('Session length must be a positive number.')
    if 0 in events[item_col].to_numpy():
      raise ValueError('Padding index 0 should not be used as item id.')
    self.events = torch.from_numpy(events.index.to_numpy()).long()  # i -> event_id, i in [0, num_events)
    self.session_len = session_len
    self.users = utils.data.lookup_table(events[user_col]).long()  # event_id -> user_id
    self.user_items = {}  # user_id -> [ item_ids ]
    user_sessions = events.groupby(user_col)
    for user, session in user_sessions:
      session_items = torch.from_numpy(session[item_col].to_numpy()).long()
      self.user_items[user] = session_items
    # position of the event in the user/item session
    self.position_in_user_sess = user_sessions.cumcount().to_numpy()  # i -> j, j in [0, num_user_events)

  def __len__(self):
    return len(self.events)

  def __getitem__(self, index):
    event = self.events[index]
    event_position = self.position_in_user_sess[index]
    user = self.users[event]
    user_items = self.user_items[user.item()]
    session_start = max(0, event_position - self.session_len)
    session_len = min(event_position - session_start + 1, self.session_len)
    if event_position < self.session_len:  # pad session
      session_items = torch.zeros(self.session_len + 1, dtype=user_items.dtype)
      session_items[-session_len:] = user_items[session_start:event_position + 1]
      input_items = session_items[:-1]
      target_items = session_items[1:]
    else:
      input_items = user_items[session_start:event_position]
      target_items = user_items[session_start + 1:event_position + 1]
    # shape:  (),  (),   (N,),        (N,),         ()
    return event, (user, input_items, target_items, session_len)


class TimeOrderedSessionIterator:
  """Iterator over session data of every event in the dataset ordered by event timestamp."""
  def __init__(self, events, batch_size=1, reset_sessions=False,
               user_col='user', item_col='item', time_col='timestamp'):
    self.batch_size = batch_size
    self.reset_sessions = reset_sessions
    self.user_sessions = {}  # user_id -> ([ event_ids ], [ item_ids ], [ timestamps ])
    for user, session in events.groupby(user_col):
      events = session.index.to_numpy()
      items = session[item_col].to_numpy()
      timestamps = session[time_col].to_numpy()
      assert utils.is_sorted(timestamps)
      self.user_sessions[user] = (events, items, timestamps)

  def __iter__(self):
    # initialize session queue (priority queue)
    session_queue = []  # (resets, timestamp, event, user, event_position)
    for user, (events, _, timestamps) in self.user_sessions.items():
      # note that timestamps are not unique, thus we need a unique event id for ordering
      heapq.heappush(session_queue, (0, timestamps[0], events[0], user, 0))
    while session_queue:
      # if sessions are reset, then the number of sessions to sample from is always the same;
      #  therefore, we can check whether there are enough sessions to form a batch
      # if sessions are not reset, the batch sizes may vary depending on the number of currently active sessions
      assert not self.reset_sessions or self.batch_size <= len(session_queue)
      actual_batch_size = min(self.batch_size, len(session_queue))
      selected_sessions = [heapq.heappop(session_queue) for _ in range(actual_batch_size)]

      batch_event = torch.zeros(actual_batch_size, dtype=torch.long)
      batch_user = torch.zeros(actual_batch_size, dtype=torch.long)
      batch_input_item = torch.zeros(actual_batch_size, dtype=torch.long)
      batch_target_item = torch.zeros(actual_batch_size, dtype=torch.long)

      for i, (resets, _, event, user, event_position) in enumerate(selected_sessions):
        events, items, timestamps = self.user_sessions[user]
        next_event_position = event_position + 1
        if next_event_position < len(events):  # if there is a next event
          heapq.heappush(session_queue, (resets, timestamps[next_event_position],
                                         events[next_event_position], user, next_event_position))
        elif self.reset_sessions:
          heapq.heappush(session_queue, (resets + 1, timestamps[0], events[0], user, 0))

        batch_event[i] = event
        batch_user[i] = user
        batch_input_item[i] = items[event_position - 1] if event_position > 0 else 0
        batch_target_item[i] = items[event_position]

      yield batch_event, (batch_user, batch_input_item, batch_target_item)


class RandomOrderSessionIterator:
  """Iterator over session data of every event in the dataset in random order."""
  def __init__(self, events, batch_size=1, reset_sessions=False,
               seed=None, user_col='user', item_col='item'):
    self.batch_size = batch_size
    self.reset_sessions = reset_sessions
    self.user_sessions = {}  # user_id -> ([ event_ids ], [ item_ids ])
    for user, session in events.groupby(user_col):
      events = session.index.to_numpy()
      items = session[item_col].to_numpy()
      self.user_sessions[user] = (events, items)
    self.rng = np.random.default_rng(seed)

  def __iter__(self):
    sessions = {user: 0 for user in self.user_sessions}  # user -> event_position
    while sessions:
      # if sessions are reset, then the number of sessions to sample from is always the same;
      #  therefore, we can check whether there are enough sessions to form a batch
      # if sessions are not reset, the batch sizes may vary depending on the number of currently active sessions
      assert not self.reset_sessions or self.batch_size <= len(sessions)
      actual_batch_size = min(self.batch_size, len(sessions))
      selected_sessions = self.rng.choice(list(sessions), actual_batch_size, replace=False)

      batch_event = torch.zeros(actual_batch_size, dtype=torch.long)
      batch_user = torch.zeros(actual_batch_size, dtype=torch.long)
      batch_input_item = torch.zeros(actual_batch_size, dtype=torch.long)
      batch_target_item = torch.zeros(actual_batch_size, dtype=torch.long)

      for i, user in enumerate(selected_sessions):
        events, items = self.user_sessions[user]
        event_position = sessions[user]
        next_event_position = event_position + 1
        if next_event_position < len(events):  # if there is a next event
          sessions[user] = next_event_position
        elif self.reset_sessions:
          sessions[user] = 0
        else:
          del sessions[user]

        batch_event[i] = events[event_position]
        batch_user[i] = user
        batch_input_item[i] = items[event_position - 1] if event_position > 0 else 0
        batch_target_item[i] = items[event_position]

      yield batch_event, (batch_user, batch_input_item, batch_target_item)


def train_step(bpr_loss, ns):
  def train_step_fn(session_model, batch_data):
    user, input_item, target_item = batch_data
    started_sessions, = torch.nonzero(input_item == 0, as_tuple=True)
    session_model.reset_sessions(user[started_sessions])
    B, = input_item.size()
    negative_samples = torch.randint(
      session_model.num_items, (B, ns), device=input_item.device)
    pos_scores, neg_scores = session_model(
      user, input_item, target_item, negative_samples)
    loss = bpr_loss(pos_scores, neg_scores)
    return loss
  return train_step_fn


def train_step_bptt(bpr_loss, ns):
  def train_step_fn(model, batch_data):
    _, input_items, target_items, _ = batch_data
    B, N = input_items.size()
    negative_samples = torch.randint(
      model.num_items, (B, N, ns), device=input_items.device)
    (pos_scores, neg_scores), _ = model(
      input_items, target_items, negative_samples)
    loss_mask = (target_items != 0).float()
    loss = bpr_loss(
      pos_scores, neg_scores,
      sample_weight=loss_mask.unsqueeze(2))
    return loss
  return train_step_fn


def eval_step(ns=None):
  def eval_step_fn(session_model, batch_data):
    user, input_item, pos_item = batch_data
    started_sessions, = torch.nonzero(input_item == 0, as_tuple=True)
    session_model.reset_sessions(user[started_sessions])
    if ns is None or ns >= session_model.num_items:
      scores = session_model.recommend(user, input_item).squeeze(1)
      targets = pos_item
    else:
      B, = input_item.size()
      neg_items = torch.randint(session_model.num_items, (B, ns), device=pos_item.device)
      selected_items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
      scores = session_model.recommend(user, input_item, selected_items=selected_items).squeeze(1)
      targets = torch.zeros(B, dtype=torch.long, device=pos_item.device)
    return scores, targets
  return eval_step_fn


if __name__ == '__main__':
  main()
