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
        #  self.test_dataset = test_dataset

        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': AdvancedLJSpeechCollator(),
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
            }
        #  self.test_dataloader_kwargs = {
        #          'batch_size': test_batch_size,
        #          'num_workers': test_num_workers,
        #          'collate_fn': AdvancedLJSpeechCollator(),
        #      }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    #  def test_dataloader(self) -> DataLoader:
    #      return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)



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
        self.len_reg = LengthRegulator(
                alpha=len_reg_alpha,
            )
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

        #  self.aligner = GraphemeAligner()
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())
        self.vocoder = Vocoder('./checkpoints/vocoder/waveglow_256channels_universal_v5.pt')

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_lr,
                betas=(0.9, 0.98),
                weight_decay=1e-6,
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

        X = self.len_reg(X, durations)
        X = self.pos_enc(X)
        X = self.fft_decoder(X)
        X = self.linear(X)
        return X, my_durations

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        target_mels = self.featurizer(batch.waveform)
        target_mels = target_mels.transpose(1, 2)

        #  durations = self.aligner(batch.waveform, batch.waveform_length, batch.transcript)
        durations = batch.durations
        durations = durations[:, :batch.tokens.shape[1]]
        durations *= target_mels.shape[1]

        out_mels, out_durations = self(batch.tokens, durations)
        min_length = min(out_mels.shape[1], target_mels.shape[1])

        cut_out_mels = out_mels[:, :min_length, :]
        cut_target_mels = target_mels[:, :min_length, :]

        mels_loss = F.mse_loss(cut_out_mels, cut_target_mels)
        duration_loss = F.mse_loss(out_durations, durations.clamp(min=1e-5).log())
        loss = mels_loss + duration_loss

        self.log('train_loss', loss.item())

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
        table_lines = []

        for pred in outputs:
            table_lines.append([
                pred['text'],
                wandb.Audio(pred['audio'], sample_rate=22050),
            ])

        table = wandb.Table(columns=['text', 'audio'], data=table_lines)
        self.logger.experiment.log({'samples': table}, commit=True)

