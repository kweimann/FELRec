import argparse
import functools
import logging.config
import random
from os import makedirs, path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

import reclib
from reclib import utils
from trainer import SASRec, GRU4Rec

parser = argparse.ArgumentParser()
parser.add_argument('--job-dir', required=True, metavar='PATH', help='job directory where output is stored')
parser.add_argument('--dataset', required=True, metavar='PATH', help='path to the dataset directory')
parser.add_argument('--data-split', metavar='PATH', help='path to the file containing train/val/test indices')
parser.add_argument('--checkpoint', metavar='PATH', help='path to a checkpoint of the model')
parser.add_argument('--lr', metavar='N', default=0.01, type=float, help='learning rate')
parser.add_argument('--m', metavar='N', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--wd', metavar='N', default=0, type=float, help='weight decay')
parser.add_argument('--warmup', metavar='N', default=10, type=int, help='number of warmup epochs')
parser.add_argument('--d-model', metavar='N', default=128, type=int, help='model dimensionality')
parser.add_argument('--layers', metavar='N', default=3, type=int, help='number of Transformer layers')
parser.add_argument('--heads', metavar='N', default=4, type=int, help='number of heads in MHA')
parser.add_argument('--share-mlp', action='store_true', help='share MLPs between users and items')
parser.add_argument('--reset-buffers', action='store_true',
                    help='reset embeddings after training. Otherwise keep embeddings for testing.')
parser.add_argument('--use-queue', action='store_true', help='use contrastive queues')
parser.add_argument('--queue-size', metavar='N', default=8192, type=int, help='queue size')
parser.add_argument('--dropout', metavar='N', default=0.1, type=float, help='dropout rate')
parser.add_argument('--session', metavar='N', default=64, type=int, help='session length')
parser.add_argument('--val-ns', metavar='N', default=16384, type=int,
                    help='number of negative samples during validation')
parser.add_argument('--batch-size', metavar='N', default=1024, type=int, help='batch size')
parser.add_argument('--epochs', metavar='N', default=100, type=int, help='number of epochs')
parser.add_argument('--val-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for validation')
parser.add_argument('--test-size', metavar='N', default=0.1, type=float, help='percentage of the dataset for testing')
parser.add_argument('--eval-use-mlp', action='store_true', help='use MLPs during evaluation')
parser.add_argument('--eval-drop-items', action='store_true', help='drop items that are not in the train set')
parser.add_argument('--eval-nn', metavar='N', default=None, type=int,
                    help='number of neighbors in the nearest neighbor recommendation. '
                         'This evaluation mode is used only if the value is set.')
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
  else:
    raise ValueError(f'Unknown dataset: {args.dataset}')

  logging.info(f"loading data from '{args.dataset}'")
  data = load_data(args.dataset, download=True)
  data.sort_values(by=['timestamp', 'event_id'], inplace=True)

  # split data into train/val/test
  if args.data_split is None:
    train_data, val_data, test_data = utils.data.split(
      data, val_size=args.val_size, test_size=args.test_size,
      divisible_by=args.batch_size if args.use_queue else None)
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

  user_dataset = SASRec.BidirectionalMaskedSessionDataset(GRU4Rec.SessionDataset(
    data, session_len=args.session, user_col='user', item_col='item'))
  item_dataset = SASRec.BidirectionalMaskedSessionDataset(GRU4Rec.SessionDataset(
    data, session_len=args.session, user_col='item', item_col='user'))
  dataset = SessionDataset(user_dataset, item_dataset)

  train_dataset = Subset(dataset, indices=train_data['_index'].to_numpy())
  val_dataset = Subset(dataset, indices=val_data['_index'].to_numpy())
  test_dataset = Subset(dataset, indices=test_data['_index'].to_numpy())

  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=4)
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=4)
  test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    num_workers=4)

  encoder = Encoder(
    num_layers=args.layers,
    num_heads=args.heads,
    d_model=args.d_model,
    d_ff=2 * args.d_model,
    dropout=args.dropout)

  if args.use_queue:
    model = FELRecQ(
      encoder=encoder,
      num_users=data['user'].max() + 1,
      num_items=data['item'].max() + 1,
      share_mlp=args.share_mlp,
      K=args.queue_size,
      reset_buffers=args.reset_buffers).to(device)
  else:
    model = FELRecP(
      encoder=encoder,
      num_users=data['user'].max() + 1,
      num_items=data['item'].max() + 1,
      share_mlp=args.share_mlp,
      reset_buffers=args.reset_buffers).to(device)

  if args.checkpoint is not None:
    logging.info(f"loading model checkpoint from '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=not args.reset_buffers)
    eval_descending = checkpoint['descending']
  else:
    checkpoint = None
    eval_descending = True

  if args.epochs > 0:
    optimizer = torch.optim.SGD(
      params=model.parameters(),
      lr=args.lr,
      momentum=args.m,
      weight_decay=args.wd)

    lr_schedule = reclib.optim.CosineLRSchedule(
      optimizer=optimizer,
      steps=args.epochs,
      warmup_steps=args.warmup,
      eta_min=1e-7)

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
      model.reset_users()
      model.reset_items()
      train_meters = reclib.optim.train_epoch(
        model=model,
        optimizer=optimizer,
        train_step=train_step_Q() if args.use_queue else train_step_P(),
        dataloader=train_loader,
        device=device)
      # note: `existing_items_mask` is mutated during evaluation
      existing_items_mask = create_mask(
        index=train_data['item'].unique(),
        size=model.num_items,
        device=device)
      _, (val_ranks,) = reclib.optim.eval_epoch(
        model=model,
        metrics=[reclib.metrics.normalized_rank],
        eval_step=eval_step(
          existing_items_mask=existing_items_mask,
          ns=args.val_ns,
          use_mlp=args.eval_use_mlp),
        dataloader=val_loader,
        device=device)
      mean_val_rank = val_ranks.mean().item()
      eval_descending = mean_val_rank <= 1 - mean_val_rank
      val_rank_meter.update(min(mean_val_rank, 1 - mean_val_rank))
      new_best_found = monitor.update()
      utils.save_checkpoint({'epoch': epoch + 1,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'lr_schedule': lr_schedule.state_dict(),
                             'descending': eval_descending},
                            file=checkpoint_path, best=new_best_found)
      logging.info(f'[{epoch + 1:03d}] {"(*)" if new_best_found else "   "} '
                   f'{" ".join(map(str, train_meters))} {val_rank_meter} '
                   f'[{"D" if eval_descending else "A"}]')

    checkpoint = torch.load(best_checkpoint_path)
    eval_descending = checkpoint['descending']
    model.load_state_dict(checkpoint['model'])

  print('--- evaluation ---')

  if args.reset_buffers:
    model.reset_users()
    model.reset_items()
    # note: `existing_items_mask` is mutated during evaluation
    existing_items_mask = torch.zeros(model.num_items, device=device)
    _, (train_ranks,) = reclib.optim.eval_epoch(
      model=model,
      metrics=[reclib.metrics.normalized_rank],
      eval_step=eval_step(
        existing_items_mask=existing_items_mask,
        use_mlp=args.eval_use_mlp),
      dataloader=train_loader,
      device=device)
    _, (val_ranks,) = reclib.optim.eval_epoch(
      model=model,
      metrics=[reclib.metrics.normalized_rank],
      eval_step=eval_step(
        existing_items_mask=existing_items_mask,
        use_mlp=args.eval_use_mlp),
      dataloader=val_loader,
      device=device)
    mean_train_rank = train_ranks.mean().item()
    mean_val_rank = val_ranks.mean().item()
    eval_descending = mean_val_rank <= 1 - mean_val_rank
    logging.info(f'train_rank {mean_train_rank if eval_descending else 1 - mean_train_rank:.4f} '
                 f'val_rank {mean_val_rank if eval_descending else 1 - mean_val_rank:.4f}')
  else:
    # note: `existing_items_mask` is mutated during evaluation
    existing_items_mask = create_mask(
      index=pd.concat([train_data, val_data])['item'].unique(),
      size=model.num_items,
      device=device)

  k = 10
  test_generator = torch.Generator(device=device)
  if args.seed is not None:
    test_generator.manual_seed(args.seed)

  if args.eval_nn is not None:
    test_generator_cpu = torch.Generator()
    if args.seed is not None:
      test_generator_cpu.manual_seed(args.seed)
    existing_users_mask = create_mask(
        index=pd.concat([train_data, val_data])['user'].unique(),
        size=model.num_users,
        device=device)
    test_step = eval_step_nn(
      existing_items_mask=existing_items_mask,
      existing_users_mask=existing_users_mask,
      n=args.eval_nn,
      n_sampled_users=args.val_ns,
      descending=True,
      generator=test_generator_cpu)
  else:
    test_step = eval_step(
      existing_items_mask=existing_items_mask,
      use_mlp=args.eval_use_mlp)

  # we follow the ordering that was determined during validation
  test_event_ids, (test_ranks, test_HR) = reclib.optim.eval_epoch(
    model=model,
    metrics=[functools.partial(reclib.metrics.normalized_rank,
                               descending=eval_descending),
             lambda scores, targets: reclib.metrics.HR(
               *utils.draw_samples(scores, targets, ns=100, generator=test_generator),
               k=k, descending=eval_descending)],
    eval_step=test_step,
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


class FELRecP(nn.Module):
  def __init__(self, encoder, num_items, num_users, share_mlp=False, reset_buffers=False):
    super().__init__()
    self.encoder = encoder
    self.num_users = num_users
    self.num_items = num_items
    self.register_buffer('user_embeddings', torch.zeros(num_users, encoder.d_model), persistent=not reset_buffers)
    self.register_buffer('item_embeddings', torch.zeros(num_items, encoder.d_model), persistent=not reset_buffers)
    self.user_token = nn.Parameter(torch.randn(1, 1, encoder.d_model))
    self.item_token = nn.Parameter(torch.randn(1, 1, encoder.d_model))
    self.user_projection = bottleneck_projection_net(encoder.d_model)
    self.user_prediction = bottleneck_prediction_net(encoder.d_model)
    if share_mlp:
      self.item_projection = self.user_projection
      self.item_prediction = self.user_prediction
    else:
      self.item_projection = bottleneck_projection_net(encoder.d_model)
      self.item_prediction = bottleneck_prediction_net(encoder.d_model)

  @torch.no_grad()
  def reset_users(self, users=None):
    self.user_embeddings[users] = 0.

  @torch.no_grad()
  def reset_items(self, items=None):
    self.item_embeddings[items] = 0.

  def forward(self, user_data, item_data):
    user, prev_items, prev_items_mask = user_data
    item, prev_users, prev_users_mask = item_data
    prev_items = self.item_token + self.item_embeddings[prev_items]  # (B, N, D)
    prev_users = self.user_token + self.user_embeddings[prev_users]  # (B, N, D)
    user_representation = self.encoder(prev_items, prev_items_mask)  # (B, D)
    item_representation = self.encoder(prev_users, prev_users_mask)  # (B, D)
    self.user_embeddings[user] = user_representation.detach()
    self.item_embeddings[item] = item_representation.detach()
    if self.training:
      user_projection = self.user_projection(user_representation)
      user_prediction = self.user_prediction(user_projection)
      item_projection = self.item_projection(item_representation)
      item_prediction = self.item_prediction(item_projection)
      user_outputs = user_representation, user_projection, user_prediction
      item_outputs = item_representation, item_projection, item_prediction
      return user_outputs, item_outputs
    else:
      return user_representation, item_representation

  @torch.no_grad()
  def recommend(self, user, items=None, use_mlp=False):
    user_emb = self.user_embeddings[user]
    item_emb = self.item_embeddings[items]
    if use_mlp:
      user_emb = self.user_prediction(self.user_projection(user_emb))
      item_emb = self.item_projection(item_emb)
    user_emb = F.normalize(user_emb, dim=1, p=2)
    item_emb = F.normalize(item_emb, dim=1, p=2)
    scores = torch.einsum('qd,kd->qk', user_emb, item_emb)
    return scores


class FELRecQ(nn.Module):
  def __init__(self, encoder, num_items, num_users, K=8192, T=0.07, share_mlp=False, reset_buffers=False):
    super().__init__()
    self.encoder = encoder
    self.num_users = num_users
    self.num_items = num_items
    self.K = K
    self.T = T
    self.register_buffer('user_embeddings', torch.zeros(num_users, encoder.d_model), persistent=not reset_buffers)
    self.register_buffer('item_embeddings', torch.zeros(num_items, encoder.d_model), persistent=not reset_buffers)
    self.register_buffer('user_queue', torch.randn(K, encoder.d_model), persistent=not reset_buffers)
    self.register_buffer('item_queue', torch.randn(K, encoder.d_model), persistent=not reset_buffers)
    self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long), persistent=not reset_buffers)
    self.user_queue = F.normalize(self.user_queue, dim=1, p=2)
    self.item_queue = F.normalize(self.item_queue, dim=1, p=2)
    self.user_token = nn.Parameter(torch.randn(1, 1, encoder.d_model))
    self.item_token = nn.Parameter(torch.randn(1, 1, encoder.d_model))
    self.user_projection = mlp(encoder.d_model)
    if share_mlp:
      self.item_projection = self.user_projection
    else:
      self.item_projection = mlp(encoder.d_model)

  @torch.no_grad()
  def reset_users(self, users=None):
    self.user_embeddings[users] = 0.

  @torch.no_grad()
  def reset_items(self, items=None):
    self.item_embeddings[items] = 0.

  def forward(self, user_data, item_data):
    user, prev_items, prev_items_mask = user_data
    item, prev_users, prev_users_mask = item_data
    prev_items = self.item_token + self.item_embeddings[prev_items]  # (B, N, D)
    prev_users = self.user_token + self.user_embeddings[prev_users]  # (B, N, D)
    user_representation = self.encoder(prev_items, prev_items_mask)  # (B, D)
    item_representation = self.encoder(prev_users, prev_users_mask)  # (B, D)
    self.user_embeddings[user] = user_representation.detach()
    self.item_embeddings[item] = item_representation.detach()
    if self.training:
      user_projection = F.normalize(self.user_projection(user_representation), dim=1, p=2)
      item_projection = F.normalize(self.item_projection(item_representation), dim=1, p=2)
      user_logits, user_targets = self.compute_similarity(
        user_projection, item_projection.detach(), self.item_queue.clone())
      item_logits, item_targets = self.compute_similarity(
        item_projection, user_projection.detach(), self.user_queue.clone())
      self.update_queues(user_projection, item_projection)
      user_outputs = user_logits, user_targets, user_representation
      item_outputs = item_logits, item_targets, item_representation
      return user_outputs, item_outputs
    else:
      return user_representation, item_representation

  def recommend(self, user, items=None, use_mlp=False):
    user_emb = self.user_embeddings[user]
    item_emb = self.item_embeddings[items]
    if use_mlp:
      user_emb = self.user_projection(user_emb)
      item_emb = self.item_projection(item_emb)
    user_emb = F.normalize(user_emb, dim=1, p=2)
    item_emb = F.normalize(item_emb, dim=1, p=2)
    logits = torch.einsum('qd,kd->qk', user_emb, item_emb)
    return logits

  def compute_similarity(self, q, k_pos, k_neg):
    logits_pos = torch.einsum('nd,nd->n', q, k_pos).unsqueeze(1)
    logits_neg = torch.einsum('nd,kd->nk', q, k_neg)
    logits = torch.cat([logits_pos, logits_neg], dim=1)
    logits = logits / self.T
    targets = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return logits, targets

  @torch.no_grad()
  def update_queues(self, user_projections, item_projections):
    B = user_projections.size(0)
    assert self.K % B == 0  # for simplicity
    ptr = int(self.queue_ptr)
    self.user_queue[ptr:ptr+B] = user_projections
    self.item_queue[ptr:ptr+B] = item_projections
    self.queue_ptr[0] = (ptr + B) % self.K


def mlp(input_dim, hidden_dim=None, output_dim=None):
  if hidden_dim is None:
    hidden_dim = input_dim
  if output_dim is None:
    output_dim = hidden_dim
  return nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_dim, output_dim))


def bottleneck_projection_net(d_model):
  return mlp(
    input_dim=d_model,
    hidden_dim=2 * d_model,
    output_dim=d_model // 2)


def bottleneck_prediction_net(d_model):
  return mlp(
    input_dim=d_model // 2,
    hidden_dim=2 * d_model,
    output_dim=d_model // 2)


class Encoder(SASRec.AttentionPooling):
  def __init__(self, num_layers, num_heads, d_model, d_ff,
               dropout=0.1, num_positions=512):
    super().__init__(num_layers, num_heads, d_model, d_ff,
                     dropout=dropout, num_positions=num_positions)
    self.fc = nn.Linear(self.d_model, self.d_model)

  def forward(self, x, mask=None):
    x = super().forward(x, mask=mask)  # (B, 1, D)
    x = self.fc(x).squeeze(1)
    return x


class SessionDataset(Dataset):
  def __init__(self, user_dataset, item_dataset):
    assert len(user_dataset) == len(item_dataset)
    self.user_dataset = user_dataset
    self.item_dataset = item_dataset

  def __len__(self):
    return len(self.user_dataset)

  def __getitem__(self, index):
    event, (user, input_items, _, items_mask) = self.user_dataset[index]
    _, (item, input_users, _, users_mask) = self.item_dataset[index]
    return event, ((user, input_items, items_mask), (item, input_users, users_mask))


def train_step_P():
  def train_step_fn(model, batch_data):
    user_data, item_data = batch_data
    (_, user_proj, user_pred), (_, item_proj, item_pred) = model(user_data, item_data)
    user_loss = similarity_loss(user_pred, item_proj.detach()).mean()
    item_loss = similarity_loss(item_pred, user_proj.detach()).mean()
    return user_loss + item_loss
  return train_step_fn


def similarity_loss(q, k):
  q = F.normalize(q, dim=-1, p=2)
  k = F.normalize(k, dim=-1, p=2)
  return 2 - 2 * (q * k).sum(dim=-1)


def train_step_Q():
  def train_step_fn(model, batch_data):
    user_data, item_data = batch_data
    (user_logits, user_targets, _), (item_logits, item_targets, _) = model(user_data, item_data)
    user_loss = F.cross_entropy(user_logits, user_targets)
    item_loss = F.cross_entropy(item_logits, item_targets)
    return user_loss + item_loss
  return train_step_fn


def eval_step(existing_items_mask, ns=None, use_mlp=False):
  def eval_step_fn(model, batch_data):
    user_data, item_data = batch_data
    user, _, _ = user_data
    item, _, _ = item_data
    num_items = existing_items_mask.sum()
    # update embeddings of user and item
    model(user_data, item_data)
    # get indices of existing items
    if ns is not None and ns < num_items - len(item):  # limit the number of negative samples
      existing_items_mask[item] = 0.  # hide items from this batch (to avoid sampling them)
      negative_samples = draw_samples_from_mask(existing_items_mask, ns, replace=False)
      existing_items_mask[item] = 1.  # add items from this batch to the sample pool
      existing_items = torch.cat([item, negative_samples])
    else:
      existing_items_mask[item] = 1.  # add items from this batch to the sample pool
      existing_items, = existing_items_mask.nonzero(as_tuple=True)
    # compute targets while adjusting for the existing items
    targets = torch.zeros(model.num_items, dtype=torch.long, device=existing_items_mask.device)
    targets[existing_items] = torch.arange(len(existing_items), device=existing_items_mask.device)
    targets = targets[item]
    # recommend existing items to the user
    scores = model.recommend(user, existing_items, use_mlp=use_mlp)
    return scores, targets
  return eval_step_fn


def eval_step_nn(existing_items_mask, existing_users_mask, n=1,
                 n_sampled_users=None, descending=True, generator=None):
  def eval_step_fn(model, batch_data):
    user_data, item_data = batch_data
    user, _, _ = user_data
    item, _, _ = item_data
    B = user.size(0)
    model(user_data, item_data)  # update embeddings of user and item
    existing_users_mask[user] = 0.  # hide users from this batch (to avoid sampling them)
    # sample nearest neighbors
    sampled_users = draw_samples_from_mask(
      existing_users_mask, n_sampled_users, replace=False, generator=generator)
    neighbors = nearest_neighbors(
      user, sampled_users, model.user_embeddings,
      n=n, descending=descending).flatten()
    existing_users_mask[user] = 1.  # add users from this batch to the sample pool
    # recommend items to neighbors
    existing_items_mask[item] = 1.  # add items from this batch to the sample pool
    existing_items, = existing_items_mask.nonzero(as_tuple=True)
    scores = model.recommend(neighbors, existing_items)
    scores = scores.reshape(B, n, -1).mean(dim=1)  # average the scores across neighbors
    # compute targets while adjusting for the existing items
    targets = torch.zeros(model.num_items, dtype=torch.long, device=existing_items_mask.device)
    targets[existing_items] = torch.arange(len(existing_items), device=existing_items_mask.device)
    targets = targets[item]
    return scores, targets
  return eval_step_fn


def nearest_neighbors(user, neighbors, embeddings, n=1, descending=True):
  user_emb = F.normalize(embeddings[user], dim=1, p=2)
  neighbors_emb = F.normalize(embeddings[neighbors], dim=1, p=2)
  scores = torch.einsum('qd,kd->qk', user_emb, neighbors_emb)
  order = scores.argsort(dim=1, descending=descending)
  nearest = neighbors[order[:, :n]]
  return nearest


def draw_samples_from_mask(mask, n, replace=False, generator=None):
  existing, = mask.nonzero(as_tuple=True)
  if replace:
    subset = torch.randint(
      len(existing), (n,), generator=generator, device=mask.device)
  else:
    # if `replace` is False, `generator` must be on CPU. See bug below
    subset = torch.randperm(  # avoids bug: https://github.com/pytorch/pytorch/issues/59756
      len(existing), generator=generator)[:n].to(existing.device)
  return existing[subset]


def create_mask(index, size=None, device=None):
  if not torch.is_tensor(index):
    index = torch.as_tensor(index)
  if size is None:
    size = index.max() + 1
  mask = torch.zeros(size, device=device)
  mask[index] = 1.
  return mask


if __name__ == '__main__':
  main()
