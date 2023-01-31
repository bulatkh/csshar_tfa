from pandas import lreshape
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

class SupervisedModel(LightningModule):
    def __init__(self,
            encoder,
            classifier,
            fine_tuning=False,
            optimizer_name='adam',
            metric_scheduler='f1-score',
            lr=0.001):
        super().__init__()
        self.save_hyperparameters('optimizer_name', 'lr')
        self.encoder = encoder
        self.classifier = classifier
        self.fine_tuning = fine_tuning

        if self.fine_tuning:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.loss = nn.CrossEntropyLoss()
        self.metric_scheduler = metric_scheduler
        self.lr = lr
        self.optimizer_name = optimizer_name

        self.log_hyperparams()

    def log_hyperparams(self):
        self.hparams['in_channels'] = self.encoder.in_channels
        self.hparams['out_channels'] = self.encoder.out_channels
        self.hparams['kernel_size'] = self.encoder.kernel_size
        
        # log hyperparameters related to the transformer model only
        if self.encoder.name == 'transformer':
            self.hparams['num_head'] = self.encoder.num_head
            self.hparams['num_layers'] = self.encoder.num_layers
            self.hparams['dropout'] = self.encoder.dropout

        self.save_hyperparameters("optimizer_name", "lr")

    def forward(self, x):
        x = self.encoder(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x
    
    def _prepare_batch(self, batch):
        x = batch[0]
        y = batch[1].long()
        if self.encoder.name in ['cnn1d', 'transformer']:
            x = x.permute(0, 2, 1)
        x = x.float()
        return x, y
    
    def training_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = self._prepare_batch(batch)
        out = self(x)
        preds = torch.argmax(out, dim=1)

        loss = self.loss(out, y)
        self.log(f"{prefix}_loss", loss)
        return {f"{prefix}_loss": loss, "preds": preds}

    def configure_optimizers(self):
      return self._initialize_optimizer()

    def _initialize_optimizer(self):
        ### Add LR Schedulers
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": '_'.join(['val', self.metric_scheduler])
            }
        }

    
