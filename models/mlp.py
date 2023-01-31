import torch
import torch.nn as nn

from pytorch_lightning.core.lightning import LightningModule

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden=[256, 128], relu_type='relu'):
        super().__init__()
        self.name = 'MLP'
        if relu_type == 'leaky':
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Sequential(
            nn.Linear(in_size, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            self.relu
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            self.relu
        )
        self.output = nn.Linear(hidden[1], out_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.name = 'LinearClassifier'
        self.classifier = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.classifier(x)
        return x


class LinearClassifierProbing(LightningModule):
    def __init__(self, in_size, out_size, lr):
        super().__init__()
        self.name = 'LinearClassifierProbing'
        self.classifier = nn.Linear(in_size, out_size)

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _prepare_batch(self, batch):
        x = batch[0].float()
        y = batch[1].long()
        return x, y
    
    def training_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }


class ProjectionMLP(nn.Module):
    def __init__(self, in_size, fc_size, out_size):
        super().__init__()
        self.out_size = out_size
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_size, fc_size),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(fc_size, out_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MLPDropout(nn.Module):
    def __init__(self, in_size, out_size, hidden=[256, 128]):
        super(MLPDropout, self).__init__()
        self.name = 'MLP'
        self.relu = nn.ReLU()
        self.linear1 = nn.Sequential(
            nn.Linear(in_size, hidden[0]),
            # nn.BatchNorm1d(hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            # nn.BatchNorm1d(hidden[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(hidden[1], out_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x 

