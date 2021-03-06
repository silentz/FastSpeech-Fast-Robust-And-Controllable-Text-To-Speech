import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int,
                       n_heads: int,
                       head_dim: int,
                       dropout: float):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self._query  = nn.Linear(embedding_dim, head_dim * n_heads, bias=False)
        self._key    = nn.Linear(embedding_dim, head_dim * n_heads, bias=False)
        self._value  = nn.Linear(embedding_dim, head_dim * n_heads, bias=False)
        self._concat = nn.Linear(n_heads * head_dim, embedding_dim, bias=False)
        self._norm   = nn.LayerNorm(embedding_dim, eps=1e-6)
        self._scale  = math.sqrt(head_dim)

        self._layer_dropout = nn.Dropout(dropout)
        self._attention_dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input.shape

        # [batch, seq, dim]

        query = self._query(input)
        key   = self._key(input)
        value = self._value(input)

        # [batch, seq, n_heads * head_dim]

        query = query.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        key   =   key.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # [batch, seq, n_heads, head_dim]

        query = query.transpose(1, 2)
        key   =   key.transpose(1, 2)
        value = value.transpose(1, 2)

        # [batch, n_heads, seq, head_dim]

        key_T = key.transpose(2, 3)                             # batch, n_heads, head_dim, seq
        attention = torch.matmul(query, key_T)                  # batch, n_heads, seq, seq
        attention = F.softmax(attention / self._scale, dim=3)   # batch, n_heads, seq, seq
        attention = self._attention_dropout(attention)          # batch, n_heads, seq, seq
        mixture = torch.matmul(attention, value)                # batch, n_heads, seq, head_dim
        mixture = mixture.transpose(1, 2)                       # batch, seq, n_heads, head_dim
        mixture = mixture.reshape(batch_size, seq_len, -1)      # batch, seq, n_heads * head_dim
        mixture = self._concat(mixture)                         # batch, seq, dim
        mixture = self._layer_dropout(mixture)                  # batch, seq, dim

        result = self._norm(mixture + input)
        return result


class ConvBlock(nn.Module):

    def __init__(self, input_size: int,
                       hidden_size: int,
                       dropout: float,
                       kernel_size_1: int,
                       kernel_size_2: int):
        super().__init__()

        self._layers = nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size_1, padding='same'),
                #  nn.Linear(input_size, hidden_size, bias=True),
                nn.ReLU(inplace=True),
                #  nn.Linear(hidden_size, input_size, bias=True),
                nn.Conv1d(hidden_size, input_size, kernel_size=kernel_size_2, padding='same'),
            )

        self._norm = nn.LayerNorm(input_size)
        self._dropout = nn.Dropout(dropout)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        X = input.transpose(1, 2)
        #  X = input
        X = self._layers(X)
        X = X.transpose(1, 2)
        X = self._dropout(X)
        result = self._norm(input + X)
        return result


class FFTBlock(nn.Module):

    def __init__(self, embedding_dim: int,
                       attention_n_heads: int,
                       attention_head_dim: int,
                       dropout: float,
                       conv_hidden_size: int,
                       conv_kernel_size_1: int = 3,
                       conv_kernel_size_2: int = 3):
        super().__init__()

        self._attention = MultiHeadSelfAttention(
                embedding_dim=embedding_dim,
                n_heads=attention_n_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
            )

        #  self._attention = nn.MultiheadAttention(
        #          embed_dim=embedding_dim,
        #          num_heads=attention_n_heads,
        #          dropout=dropout,
        #      )

        self._conv = ConvBlock(
                input_size=embedding_dim,
                hidden_size=conv_hidden_size,
                dropout=dropout,
                kernel_size_1=conv_kernel_size_1,
                kernel_size_2=conv_kernel_size_2,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stage_1 = self._attention(input)
        #  stage_1, _ = self._attention(input, input, input)
        stage_2 = self._conv(stage_1)
        return stage_2


class LengthRegulator(nn.Module):

    def __init__(self, alpha: float = 1.):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor, durations: torch.LongTensor) -> torch.Tensor:
        patched_dur = torch.round(self.alpha * durations).long()
        patched_dur[patched_dur < 0] = 0
        seq = [x.repeat_interleave(y, dim=0) for x, y in zip(input, patched_dur)]
        result = pad_sequence(seq, batch_first=True, padding_value=0)

        #survive first epochs of training
        if result.shape[1] == 0:
            result = input

        return result


class Transpose(nn.Module):

    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.transpose(input, self.dim1, self.dim2)


class DurationPredictor(nn.Module):

    def __init__(self, embedding_dim: int,
                       hidden_size: int,
                       dropout: float,
                       kernel_size_1: int = 3,
                       kernel_size_2: int = 3):
        super().__init__()

        self._layers = nn.Sequential(
                Transpose(1, 2),
                nn.Conv1d(embedding_dim, hidden_size, kernel_size_1, padding='same'),
                Transpose(1, 2),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                Transpose(1, 2),
                nn.Conv1d(hidden_size, hidden_size, kernel_size_2, padding='same'),
                Transpose(1, 2),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input).squeeze(dim=2)
