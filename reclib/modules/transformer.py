"""Reference: https://arxiv.org/abs/1706.03762"""

__all__ = ('TransformerEncoder', 'EncoderLayer', 'SubLayer', 'MultiHeadAttention',
           'MLP', 'PositionalEncoding', 'attention', 'positional_encoding')

import math

import torch
from torch import nn
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
  def __init__(self, num_layers, num_heads, d_model, d_ff, dropout=0.1, num_positions=512):
    super().__init__()
    self.d_model = d_model
    self.positional = PositionalEncoding(d_model, num_positions)
    self.dropout = nn.Dropout(dropout)
    self.norm = nn.LayerNorm(d_model)
    self.layers = nn.ModuleList(
      [EncoderLayer(num_heads, d_model, d_ff, dropout)
       for _ in range(num_layers)])

  def forward(self, x, mask=None):
    x = x * math.sqrt(self.d_model)
    x = self.positional(x)
    x = self.dropout(x)
    x = self.norm(x)
    for layer in self.layers:
      x = layer(x, x, x, mask=mask)
    return x


class EncoderLayer(nn.Module):
  def __init__(self, num_heads, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.mha = MultiHeadAttention(num_heads, d_model)
    self.ffn = MLP(d_model, d_ff, dropout)
    self.mha_sublayer = SubLayer(d_model, dropout)
    self.ffn_sublayer = SubLayer(d_model, dropout)

  def forward(self, q, k, v, mask=None):
    x = self.mha_sublayer(q, self.mha(q, k, v, mask=mask))
    x = self.ffn_sublayer(x, self.ffn(x))
    return x


class SubLayer(nn.Module):
  def __init__(self, d_model, dropout=0.1):
    super().__init__()
    self.norm = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, outputs):
    return self.norm(inputs + self.dropout(outputs))


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, d_model):
    super().__init__()
    assert d_model % num_heads == 0
    self.num_heads = num_heads
    self.d_head = d_model // num_heads
    self.W_q = nn.Linear(d_model, d_model, bias=False)
    self.W_k = nn.Linear(d_model, d_model, bias=False)
    self.W_v = nn.Linear(d_model, d_model, bias=False)
    self.W_o = nn.Linear(d_model, d_model, bias=False)
    self.attn_weights = None

  def split_heads(self, x):
    B, N, d_model = x.size()
    return x.reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)

  # noinspection PyMethodMayBeStatic
  def concat_heads(self, x):
    B, H, N, d_head = x.size()
    d_model = H * d_head
    return x.transpose(2, 1).reshape(B, N, d_model)

  def forward(self, q, k, v, mask=None):
    q = self.split_heads(self.W_q(q))
    k = self.split_heads(self.W_k(k))
    v = self.split_heads(self.W_v(v))
    if mask is not None:
      mask = mask.unsqueeze(1)
    # mh_attn, _ = attention(q, k, v, mask)
    mh_attn, self.attn_weights = attention(q, k, v, mask)
    attn = self.W_o(self.concat_heads(mh_attn))
    return attn


class MLP(nn.Module):
  def __init__(self, d_model, d_ff=None, dropout=0.1):
    super().__init__()
    if d_ff is None:
      d_ff = d_model
    self.W_1 = nn.Linear(d_model, d_ff)
    self.W_2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.W_2(self.dropout(F.relu(self.W_1(x))))


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, length):
    super().__init__()
    encoding = positional_encoding(d_model, length).unsqueeze(0)
    self.register_buffer('encoding', encoding)

  def forward(self, x):
    N = x.size(1)
    return x + self.encoding[:, :N]


def attention(q, k, v, mask=None):
  d_model = q.size(-1)
  attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
  if mask is not None:
    attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
  attn_weights = F.softmax(attn_weights, dim=-1)
  attn = torch.matmul(attn_weights, v)
  return attn, attn_weights


def positional_encoding(d_model, length):
  assert d_model % 2 == 0  # for simplicity
  pos = torch.arange(0, length).unsqueeze(1)
  i = torch.arange(0, d_model, 2)
  angle_rads = pos * torch.exp(i * -(math.log(10000) / d_model))
  pe = torch.zeros(length, d_model)
  pe[:, 0::2] = torch.sin(angle_rads)
  pe[:, 1::2] = torch.cos(angle_rads)
  return pe
