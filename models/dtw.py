import torch
import torch.nn.functional as F
from apex.parallel.LARC import LARC
from libraries.pytorch_softdtw_cuda.soft_dtw_cuda import SoftDTW
from pytorch_lightning.core.lightning import LightningModule
from torch import nn

from models.simclr import NTXent


class DTWModule(LightningModule):
    """
    Implementation of DTW 
    """
    def __init__(self, encoder, projection, optimizer_name_ssl='adam', ssl_lr=0.001, alpha=0.1, beta=0.1, gamma=0.5, l=2, sigma=15, n_views=2, ssl_batch_size=64, temperature=0.1, **kwargs):
        super().__init__()
        # encoder class
        self.encoder = encoder
        self.projection = projection 
        # optimization
        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = ssl_lr
        # DTW-related parameters
        self.alpha = alpha
        self.l = l
        self.sigma = sigma
		# Losses
        self.dtw_loss = SoftDTW(use_cuda=True, normalize=False, gamma=gamma)
        self.ntxent_loss = NTXent(batch_size=ssl_batch_size, n_views=n_views, temperature=temperature)

        self.log_hyperparams()

    def log_hyperparams(self):
        self.hparams['in_channels'] = self.encoder.in_channels
        self.hparams['out_channels'] = self.encoder.out_channels
        self.hparams['num_head'] = self.encoder.num_head
        self.hparams['num_layers'] = self.encoder.num_layers
        self.hparams['kernel_size'] = self.encoder.kernel_size
        self.hparams['dropout'] = self.encoder.dropout
        self.hparams['embedding_size'] = self.projection.out_size
        self.save_hyperparameters(ignore=["batch_size", "n_views"])

    def _prepare_batch(self, batch):
        batch = torch.cat(batch, dim=0)
        if self.encoder.name == 'transformer':
            batch = batch.permute(0, 2, 1)
        batch = batch.float()
        return batch

    def forward(self, x):
        temporal = self.encoder(x)
        flattened = nn.Flatten()(temporal)
        projected = self.projection(flattened)
        return projected, temporal

    def training_step(self, batch, batch_idx):
		# preprocess batch
        batch = self._prepare_batch(batch)
		# pass the batch through the model
        projected, temporal = self(batch)
        first, second = temporal.split(int(temporal.shape[0]/2), dim=0)
		# normalize temporal embeddings
        first = F.normalize(first, dim=2)
        second = F.normalize(second, dim=2)
		# compute losses
        dtw_loss = self.dtw_loss(first, second).mean()
        nt_xent_loss, pos, neg = self.ntxent_loss(projected)
        loss = nt_xent_loss + self.alpha * dtw_loss
		# log losses and average similarities
        self.log("dtw_train_loss", dtw_loss)
        self.log("nt_xent_loss", nt_xent_loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        self.log("ssl_train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        projected, temporal = self(batch)

        first, second = temporal.split(int(temporal.shape[0]/2), dim=0)
        first = F.normalize(first, dim=2)
        second = F.normalize(second, dim=2)

        dtw_loss = self.dtw_loss(first, second).mean()
        nt_xent_loss, pos, neg = self.ntxent_loss(projected)
        loss = nt_xent_loss + self.alpha * dtw_loss

        self.log("dtw_val_loss", dtw_loss)
        self.log("nt_xent_val_loss", nt_xent_loss)
        self.log("ssl_val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_train_loss'
                }
            }

        elif self.optimizer_name_ssl.lower() == 'lars':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            optimizer = LARC(optimizer)
            return {
                "optimizer": optimizer
            }
