import lightning as L
import torch
from torch import nn, optim
from torch.nn import Linear
from torch.nn.functional import elu, softmax
from torch_geometric.nn import GCNConv, global_mean_pool, GraphSAGE, SAGEConv
from torchmetrics import AUROC

from sam import SAM


class GraphSage(L.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_hidden_layers: int = 2,
        num_hidden: int = 128,
        with_sam: bool = True,
        graph_classification: bool = False,
        base_optimizer: str = "adam", # Can be "sgd" or "adam"
        seed=1234,
        lr=0.01,
    ):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.lr=lr
        if with_sam:
            self.automatic_optimization = False

        self.with_sam = with_sam
        self.graph_classification = graph_classification

        torch.manual_seed(seed)
        self.conv1 = SAGEConv(num_features, num_hidden)

        self.hidden_layers = []

        for _ in range(num_hidden_layers):
            self.hidden_layers.append(SAGEConv(num_hidden, num_hidden))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.classifier = Linear(num_hidden, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.pool = global_mean_pool

        self.roc_auc_score = AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = elu(h)

        for hidden_layer in self.hidden_layers:
            h = hidden_layer(h, edge_index)
            h = elu(h)

        if self.graph_classification:
            h = self.pool(h, batch)

        out = self.classifier(h)
        return out
    
    def compute_loss(self, out, y, batch):
        if self.graph_classification:
            loss = self.criterion(out, y)
        else:
            train_mask = batch.train_mask
            loss = self.criterion(out[train_mask], y[train_mask])
        return loss

    def configure_optimizers(self):
        if self.base_optimizer == "sgd":
            base_optimizer = optim.SGD
        elif self.base_optimizer == "adam":
            base_optimizer = optim.Adam
        else:
            raise ValueError(f"{self.base_optimizer} is an invalid base_optimizer. Must be 'sgd' or 'adam'")

        if self.with_sam:
            optimizer = SAM(
                params=self.parameters(), base_optimizer=base_optimizer, lr=self.lr
            )
            return optimizer

        return base_optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index

        out = self.forward(x, edge_index, batch.batch)

        loss = self.compute_loss(out, y, batch)

        self.log("train/loss", loss, batch_size=len(y), prog_bar=True)

        def closure():
            out = self.forward(x, edge_index, batch.batch)

            if self.graph_classification:
                loss = self.criterion(out, y)
            else:
                train_mask = batch.train_mask
                loss = self.criterion(out[train_mask], y[train_mask])

            loss.backward()
            return loss

        if self.with_sam:
            opt = self.optimizers()
            loss.backward()
            opt.optimizer.step(closure)
            opt.optimizer.zero_grad()

        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage: str):
        out = self.forward(batch.x, batch.edge_index, batch.batch)
        out = out.cpu()
        batch = batch.cpu()

        if self.graph_classification:
            pred_probs = softmax(out, dim=1).detach().numpy()
            true_labels = batch.y.detach().numpy()
            loss = self.criterion(out, batch.y)

        else:
            node_mask = batch.val_mask
            label_mask = batch.val_mask

            if stage == "test":
                node_mask = batch.test_mask
                label_mask = batch.test_mask

            pred_probs = softmax(out[node_mask], dim=1).detach().numpy()
            true_labels = batch.y[label_mask].detach().numpy()
            loss = self.criterion(out[node_mask], batch.y[label_mask])

        # Calculate accuracy
        pred_labels = pred_probs.argmax(axis=1)
        acc = (pred_labels == true_labels).mean()

        pred_probs = torch.tensor(pred_probs, dtype=torch.float32)
        true_labels = torch.tensor(true_labels,dtype=torch.long)

        auc_score = self.roc_auc_score(pred_probs, true_labels)
        self.log(f"{stage}/auc_score", auc_score, batch_size=len(true_labels), prog_bar=False)
        self.log(f"{stage}/loss", loss, batch_size=len(true_labels), prog_bar=False)
        self.log(
            f"{stage}/accuracy", acc, batch_size=len(true_labels), prog_bar=True
        )
