import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int,
                       n_heads: int,
                       head_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self._query  = nn.Linear(embedding_dim, head_dim * n_heads)
        self._key    = nn.Linear(embedding_dim, head_dim * n_heads)
        self._value  = nn.Linear(embedding_dim, head_dim * n_heads)
        self._concat = nn.Linear(n_heads * head_dim, embedding_dim)
        self._scale  = math.sqrt(head_dim)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = input.shape

        # [batch, seq, dim]

        query = self._query(input)
        key   = self._key(input)
        value = self._value(input)

        # [batch, seq, n_heads * head_dim]

        query = query.reshape(-1, seq_len, self.n_heads, self.head_dim)
        key   = key.reshape(-1, seq_len, self.n_heads, self.head_dim)
        value = value.reshape(-1, seq_len, self.n_heads, self.head_dim)

        # [batch, seq, n_heads, head_dim]

        query = query.transpose(1, 2)
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # [batch, n_heads, seq, head_dim]

        key_T = key.transpose(2, 3)                             # batch, n_heads, head_dim, seq
        attention = torch.matmul(query, key_T)                  # batch, n_heads, seq, seq
        attention = F.softmax(attention / self._scale, dim=3)   # batch, n_heads, seq, seq
        mixture = torch.matmul(attention, value)                # batch, n_heads, seq, head_dim
        mixture = mixture.transpose(1, 2)                       # batch, seq, n_heads, head_dim
        mixture = mixture.reshape(batch_size, seq_len, -1)      # batch, seq, n_heads * head_dim
        mixture = self._concat(mixture)                         # batch, seq, dim

        return mixture
