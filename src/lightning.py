import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
import pytorch_lightning as pl

from typing import (
    Any,
    Dict,
    List,
)

from .aligner import GraphemeAligner
from .collate import LJSpeechCollator
from .dataset import LJSpeechDataset
from .layers import FFTBlock, LengthRegulator, DurationPredictor
from .mels import MelSpectrogram
from .positional import PositionalEncoding
from .vocoder import Vocoder


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset: Dataset,
                       train_batch_size: int,
                       train_num_workers: int,
                       val_dataset: Dataset,
                       val_batch_size: int,
                       val_num_workers: int,
                       test_dataset: Dataset,
                       test_batch_size: int,
                       test_num_workers: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': LJSpeechCollator(),
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
                'collate_fn': LJSpeechCollator(),
            }
        self.test_dataloader_kwargs = {
                'batch_size': test_batch_size,
                'num_workers': test_num_workers,
                'collate_fn': LJSpeechCollator(),
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)



class Module(pl.LightningModule):

    def __init__(self, encoder_layers: int,
                       decoder_layers: int,
                       embedding_dim: int,
                       attention_n_heads: int,
                       attention_head_dim: int,
                       dropout: float,
                       conv_hidden_size: int,
                       n_mels: int,
                       duration_pred_hidden_size: int,
                       criterion: nn.Module,
                       optimizer_lr: float,
                       n_examples: int):
        super().__init__()
        self.criterion = criterion
        self.optimizer_lr = optimizer_lr
        self.n_examples = n_examples

        self.embedding = nn.Embedding(
                num_embeddings=1,
                embedding_dim=embedding_dim,
                padding_idx=1,
            )
        self.pos_enc = PositionalEncoding(embedding_dim)
        self.fft_encoder = nn.Sequential(*[
                FFTBlock(
                        embedding_dim=embedding_dim,
                        attention_n_heads=attention_n_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        conv_hidden_size=conv_hidden_size,
                    )
                for _ in range(encoder_layers)
            ])
        self.len_reg = LengthRegulator()
        self.dur_pred = DurationPredictor(
                embedding_dim=embedding_dim,
                hidden_size=duration_pred_hidden_size,
                dropout=dropout,
            )
        self.fft_decoder = nn.Sequential(*[
                FFTBlock(
                        embedding_dim=embedding_dim,
                        attention_n_heads=attention_n_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        conv_hidden_size=conv_hidden_size,
                    )
                for _ in range(decoder_layers)
            ])
        self.linear = nn.Linear(embedding_dim, n_mels)

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_lr,
                betas=(0.9, 0.98),
            )
        sched = torch.optim.lr_scheduler.LambdaLR(
                optim,
                lr_lambda=lambda epoch: (0.99 ** epoch),
            )
        return {
                'optimizer': optim,
                'scheduler': sched,
            }

    def forward(self, batch) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx) -> Dict[str, Any]:
       pass

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        pass

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pass

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        pass

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        pass

