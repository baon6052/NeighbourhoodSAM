from dataclasses import dataclass

import torch
from graph_sam import GraphSAM
from torch.nn import Linear, Parameter
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
import lightning as L
from torch import nn, optim


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
        self, num_features: int, num_classes: int, visualise: bool = False
    ):
        super().__init__()
        self.automatic_optimization = False

        self.visualise = visualise

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

        optimizer = GraphSAM(
            params=self.parameters(), arg=args, base_optimizer=base_optimizer
        )

        return optimizer

    def training_step(self, batch, batch_idx):
        print(batch)
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()

        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])

        opt = self.optimizers()

        def closure():
            loss = self.compute_loss(batch)
            opt.zero_grad()
            self.manual_backward(loss)
            return loss

        opt.step(batch_idx, batch_idx, closure=closure, loss=loss)

        return loss
