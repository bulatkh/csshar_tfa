import torch
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch import nn

from apex.parallel.LARC import LARC


class VicReg(LightningModule):
    """
    Implementation of VicReg adapted from https://github.com/facebookresearch/vicreg/
    """
    def __init__(self, encoder, projection, embedding_size, ssl_batch_size, sim_coeff = 25, std_coeff = 25, cov_coeff = 1, optimizer_name_ssl='lars', ssl_lr=0.001, **kwargs):

        super().__init__()

        self.encoder = encoder
        self.projection = projection

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = ssl_lr

        self.embedding_size = embedding_size
        self.ssl_batch_size = ssl_batch_size

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        self.log_hyperparams()

    def log_hyperparams(self):
        self.hparams['in_channels'] = self.encoder.in_channels
        self.hparams['out_channels'] = self.encoder.out_channels
        self.hparams['num_head'] = self.encoder.num_head
        self.hparams['num_layers'] = self.encoder.num_layers
        self.hparams['kernel_size'] = self.encoder.kernel_size
        self.hparams['dropout'] = self.encoder.dropout
        self.save_hyperparameters(ignore=["batch_size", "num_features"])

    def _prepare_batch(self, batch):
        x = batch[0]
        y = batch[1]

        if self.encoder.name == 'transformer':
            x = x.permute(0, 2, 1)
            y = y.permute(0, 2, 1)

        x = x.float()
        y = y.float()

        return x, y

    def _compute_vicreg_loss(self, x, y, partition):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.ssl_batch_size - 1)
        cov_y = (y.T @ y) / (self.ssl_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        self.log(f"repr_{partition}_loss", repr_loss)
        self.log(f"std_{partition}_loss", std_loss)
        self.log(f"cov_{partition}_loss", cov_loss)
        self.log(f"{partition}_loss", loss)
        return loss
    
    def forward(self, x, y):
        x = self.projection(nn.Flatten()(self.encoder(x)))
        y = self.projection(nn.Flatten()(self.encoder(y)))
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        x, y = self(x, y)
        return self._compute_vicreg_loss(x, y, 'train')

    def validation_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        x, y = self(x, y)
        return self._compute_vicreg_loss(x, y, 'val')

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

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()