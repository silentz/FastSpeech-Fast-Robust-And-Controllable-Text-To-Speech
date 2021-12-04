import torch
import torch.nn as nn
import torch.nn.functional as F
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

from .collate import AdvancedLJSpeechCollator, Batch
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
                       val_num_workers: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': AdvancedLJSpeechCollator(),
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)



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
                       len_reg_alpha: float,
                       optimizer_lr: float):
        super().__init__()
        self.optimizer_lr = optimizer_lr

        self.embedding = nn.Embedding(
                num_embeddings=51,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )
        self.pos_enc = PositionalEncoding(embedding_dim)
        self.fft_encoder = nn.ModuleList([
                FFTBlock(
                        embedding_dim=embedding_dim,
                        attention_n_heads=attention_n_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        conv_hidden_size=conv_hidden_size,
                    )
                for _ in range(encoder_layers)
            ])
        self.len_reg = LengthRegulator(
                alpha=len_reg_alpha,
            )
        self.dur_pred = DurationPredictor(
                embedding_dim=embedding_dim,
                hidden_size=duration_hidden_size,
                dropout=dropout,
            )
        self.fft_decoder = nn.ModuleList([
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

        self.featurizer = MelSpectrogram(MelSpectrogramConfig())
        self.vocoder = Vocoder('./checkpoints/vocoder/waveglow_256channels_universal_v5.pt')

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_lr,
                betas=(0.9, 0.98),
                eps=1e-9,
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

        for enc_layer in self.fft_encoder:
            X = enc_layer(X)

        my_durations = self.dur_pred(X)

        if durations is None:
            durations = my_durations

        X = self.len_reg(X, durations)
        X = self.pos_enc(X)

        for dec_layer in self.fft_decoder:
            X = dec_layer(X)

        X = self.linear(X)
        return X, my_durations

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        target_mels = self.featurizer(batch.waveform)
        target_mels = target_mels.transpose(1, 2)

        durations = batch.durations
        durations = durations[:, :batch.tokens.shape[1]]
        durations *= target_mels.shape[1]

        out_mels, out_durations = self(batch.tokens, durations)
        max_length = max(out_mels.shape[1], target_mels.shape[1])

        def pad_to_max_length(tensor: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, n_mels = tensor.shape
            padded = torch.zeros(batch_size, max_length, n_mels, device=tensor.device)
            padded[:, :seq_len, :] = tensor
            return padded

        cut_out_mels = pad_to_max_length(out_mels)
        cut_target_mels = pad_to_max_length(target_mels)

        mels_loss = F.l1_loss(cut_out_mels, cut_target_mels)
        duration_loss = F.mse_loss(out_durations, durations)
        loss = mels_loss + duration_loss

        self.log('mel_loss', mels_loss.item())
        self.log('duration_loss', duration_loss.item())
        self.log('total_loss', loss.item())

        return {
                'loss': loss,
            }


    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        tokens, _, text = batch
        out_mels, _ = self(tokens)

        out_mels = out_mels.transpose(1, 2)
        audio = self.vocoder.inference(out_mels)

        return {
                'audio': audio[0].detach().cpu(),
                'text': text[0],
            }


    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        table_name = f'samples_{self.current_epoch}'
        table_lines = []

        for pred in outputs:
            table_lines.append([
                pred['text'],
                wandb.Audio(pred['audio'], sample_rate=22050),
            ])

        table = wandb.Table(columns=['text', 'audio'], data=table_lines)
        self.logger.experiment.log({table_name: table}, commit=True)

