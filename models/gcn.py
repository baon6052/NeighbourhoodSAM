from dataclasses import dataclass

import lightning as L
import torch
from torch import nn, optim
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from graph_sam import GraphSAM


@dataclass
class Args:
    rho: float
    radius: float
    alpha: float
    epoch_steps: int
    gamma: float


args = Args(rho=0.05, radius=0.05, alpha=0.99, epoch_steps=1, gamma=0.5)


class GCN(L.LightningModule):
    def __init__(
            self, num_features: int, num_classes: int, with_sam: bool = True,
    ):
        super().__init__()
        if with_sam:
            self.automatic_optimization = False

        self.with_sam = with_sam

        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)
        return out, h

    def configure_optimizers(self):

        base_optimizer = optim.Adam

        if self.with_sam:
            optimizer = GraphSAM(
                params=self.parameters(), arg=args, base_optimizer=base_optimizer
            )

            return optimizer
        else:
            return base_optimizer(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()

        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
        self.log("train/loss", loss, prog_bar=True)

        opt = self.optimizers()

        x = batch.x
        edge_index = batch.edge_index
        train_mask = batch.train_mask

        def closure():
            out, h = self.forward(x, edge_index)
            out = out.cpu()
            y = batch.y

            loss = self.criterion(out[train_mask], y[train_mask])
            loss.backward()
            return loss

        if self.with_sam:
            if batch_idx == 0:
                opt.optimizer.step(batch_idx, batch_idx, closure=closure, loss=loss)
            else:
                opt.optimizer.step(batch_idx, batch_idx, closure=closure)

        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage: str):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()

        node_mask = batch.val_mask
        label_mask = batch.val_mask

        if stage == "test":
            node_mask = batch.test_mask
            label_mask = batch.test_mask

        pred = out.argmax(1)

        # loss = self.criterion(out[node_mask], batch.y[label_mask])
        acc = (pred[node_mask] == batch.y[label_mask]).float().mean()
        self.log(f"{stage}_accuracy", acc, prog_bar=False)
