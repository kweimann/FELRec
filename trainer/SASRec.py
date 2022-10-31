import argparse
import logging.config
import random
from os import makedirs, path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

import reclib
from reclib import utils, modules
from trainer import GRU4Rec

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
parser.add_argument('--heads', metavar='N', default=4, type=int, help='number of heads in MHA')
parser.add_argument('--ns', metavar='N', default=8, type=int, help='number of negative samples')
parser.add_argument('--val-ns', metavar='N', default=16384, type=int,
                    help='number of negative samples during validation')
parser.add_argument('--dropout', metavar='N', default=0.1, type=float, help='dropout rate')
parser.add_argument('--session', metavar='N', default=64, type=int, help='session length')
parser.add_argument('--bidirectional', action='store_true', help='train a bidirectional session-based recommender')
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

  dataset = GRU4Rec.SessionDataset(data, session_len=args.session)
  if args.bidirectional:
    dataset = BidirectionalMaskedSessionDataset(dataset)
  else:
    dataset = MaskedSessionDataset(dataset)

  train_loader = DataLoader(
    dataset=Subset(dataset, indices=train_data['_index'].to_numpy()),
    batch_size=args.batch_size,
    num_workers=4,
    shuffle=True)
  val_loader = DataLoader(
    dataset=Subset(dataset, indices=val_data['_index'].to_numpy()),
    batch_size=args.batch_size,
    num_workers=4)

  if args.bidirectional:
    transformer = AttentionPooling(
      num_layers=args.layers,
      num_heads=args.heads,
      d_model=args.d_model,
      d_ff=2 * args.d_model,
      dropout=args.dropout)
  else:
    transformer = TransformerEncoder(
      num_layers=args.layers,
      num_heads=args.heads,
      d_model=args.d_model,
      d_ff=2 * args.d_model,
      dropout=args.dropout)
  model = SASRec(
    transformer=transformer,
    num_items=train_data['item'].max() + 1).to(device)

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
        steps_per_epoch=args.steps_per_epoch,
        device=device)
      _, (val_ranks,) = reclib.optim.eval_epoch(
        model=model,
        metrics=[reclib.metrics.normalized_rank],
        eval_step=eval_step(ns=args.val_ns),
        dataloader=val_loader,
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

  test_loader = DataLoader(
    dataset=Subset(dataset, indices=test_data['_index'].to_numpy()),
    batch_size=args.batch_size,
    num_workers=4)

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


class SASRec(nn.Module):
  """Reference: https://arxiv.org/pdf/1808.09781.pdf"""
  def __init__(self, transformer, num_items):
    super().__init__()
    self.num_items = num_items
    self.embedding = nn.Embedding(num_items, transformer.d_model, padding_idx=0)
    self.transformer = transformer
    self.fc = nn.Linear(transformer.d_model, transformer.d_model)

  def forward(self, input_items, mask, pos_items, neg_items):
    x = self.embedding(input_items)
    x = self.transformer(x, mask)
    x = self.fc(x)
    pos_scores = torch.einsum(
      'bnd,bnd->bn', x, self.embedding(pos_items))
    neg_scores = torch.einsum(
      'bnd,bnkd->bnk', x, self.embedding(neg_items))
    return pos_scores, neg_scores

  @torch.no_grad()
  def recommend(self, input_items, mask, selected_items=None, only_last=False):
    x = self.embedding(input_items)
    x = self.transformer(x, mask)
    if only_last and x.size(1) > 1:
      assert selected_items is None or selected_items.size(1) == 1  # (B, 1, ns)
      x = x[:, -1].unsqueeze(1)  # (B, 1, D)
    x = self.fc(x)
    if selected_items is None:
      scores = torch.einsum(
        'bnd,kd->bnk', x, self.embedding.weight)
    else:
      scores = torch.einsum(
        'bnd,bnkd->bnk', x, self.embedding(selected_items))
    return scores


class TransformerEncoder(modules.transformer.TransformerEncoder):
  def __init__(self, num_layers, num_heads, d_model, d_ff,
               dropout=0.1, num_positions=512):
    super().__init__(num_layers, num_heads, d_model, d_ff,
                     dropout=dropout, num_positions=num_positions)
    # overwrite positional encoding
    self.positional = ReversedPositionalEncoding(d_model, num_positions)


class AttentionPooling(TransformerEncoder):
  def forward(self, x, mask=None):
    assert mask is not None
    x = super().forward(x, mask=mask)  # (B, N, D)
    # average the existing encodings
    exist_mask = mask.diagonal(dim1=1, dim2=2).unsqueeze(2)  # (B, N, 1)
    num_existing = exist_mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
    x = torch.sum(exist_mask * x, dim=1, keepdim=True) / num_existing  # (B, 1, D)
    return x


class ReversedPositionalEncoding(modules.transformer.PositionalEncoding):
  def forward(self, x):
    N = x.size(1)
    return x + self.encoding[:, :N].flip(1)


class MaskedSessionDataset(Dataset):
  """Session data of every event in the dataset."""
  def __init__(self, session_dataset):
    self.dataset = session_dataset
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, index):
    event, (user, input_items, target_items, session_len) = self.dataset[index]
    mask = torch.ones(self.dataset.session_len, self.dataset.session_len).tril()
    mask[:, :-session_len] = 0.
    # shape:  (),  (),   (N,),        (N,),         (N, N)
    return event, (user, input_items, target_items, mask)


class BidirectionalMaskedSessionDataset(Dataset):
  """Session data of every event in the dataset."""
  def __init__(self, session_dataset):
    self.dataset = session_dataset
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, index):
    event, (user, input_items, target_items, session_len) = self.dataset[index]
    mask = torch.zeros(self.dataset.session_len, self.dataset.session_len)
    mask[-session_len:, -session_len:] = 1.
    target_item = target_items[-1].unsqueeze(0)
    # shape:  (),  (),   (N,),        (1,),        (N, N)
    return event, (user, input_items, target_item, mask)


def train_step(bpr_loss, ns):
  def train_step_fn(model, batch_data):
    _, input_items, pos_items, mask = batch_data
    B, N = pos_items.size()
    neg_items = torch.randint(
      model.num_items, (B, N, ns), device=pos_items.device)
    pos_scores, neg_scores = model(
      input_items, mask, pos_items, neg_items)
    loss_mask = (pos_items != 0).float()
    loss = bpr_loss(
      pos_scores.unsqueeze(2), neg_scores,
      sample_weight=loss_mask.unsqueeze(2))
    return loss
  return train_step_fn


def eval_step(ns=None):
  def eval_step_fn(model, batch_data):
    _, input_items, pos_items, mask = batch_data
    B, N = pos_items.size()
    if N > 1:
      pos_items = pos_items[:, -1].unsqueeze(1)
    if ns is None or ns >= model.num_items:
      scores = model.recommend(input_items, mask, only_last=True).squeeze(1)
      targets = pos_items.squeeze(1)
    else:
      neg_items = torch.randint(model.num_items, (B, ns), device=pos_items.device)
      selected_items = torch.cat([pos_items, neg_items], dim=1).unsqueeze(1)
      scores = model.recommend(
        input_items, mask, selected_items=selected_items, only_last=True).squeeze(1)
      targets = torch.zeros(B, dtype=torch.long, device=pos_items.device)
    return scores, targets
  return eval_step_fn


if __name__ == '__main__':
  main()
