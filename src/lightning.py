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
    Tuple,
    Union,
)

from .aligner import GraphemeAligner
from .collate import LJSpeechCollator
from .dataset import LJSpeechDataset
from .layers import FFTBlock, LengthRegulator, DurationPredictor
from .mels import MelSpectrogram, MelSpectrogramConfig
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
                       duration_hidden_size: int,
                       optimizer_lr: float):
        super().__init__()
        self.optimizer_lr = optimizer_lr

        self.embedding = nn.Embedding(
                num_embeddings=51,
                embedding_dim=embedding_dim,
                padding_idx=0,
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
                hidden_size=duration_hidden_size,
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

        self.aligner = GraphemeAligner()
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())
        self.vocoder = Vocoder('./checkpoints/vocoder/waveglow_256channels_universal_v5.pt')

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

    def forward(self, input: torch.Tensor,
                      durations: Union[torch.Tensor, None] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.embedding(input)
        X = self.pos_enc(X)
        X = self.fft_encoder(X)
        my_durations = self.dur_pred(X)

        if durations is None:
            durations = my_durations
            durations = torch.exp(durations)
            durations = durations.long()

        X = self.len_reg(X, durations)
        X = self.pos_enc(X)
        X = self.fft_decoder(X)
        X = self.linear(X)
        return X, my_durations

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

