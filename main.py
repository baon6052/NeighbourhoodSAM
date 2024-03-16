import click
from lightning import Trainer
from datasets.dataset import get_dataset
from models.gcn import GCN
from utilities.wandb_utilities import get_callbacks
from lightning.pytorch.loggers import WandbLogger


def process_items(ctx, param, value):
    if value:
        items = value.split(",")
        items = [int(item.strip()) for item in items]
        return items


@click.command()
@click.option(
    "--dataset_type",
    type=click.Choice(
        ["node_homo", "node_hetero", "graph_homo"], case_sensitive=True
    ),
    default="node_homo",
    help="The dataset to use. Options are homo and hetero",
)
@click.option("--dataset_name", default="CORA")
@click.option("--fold_idx", type=int, default=0)
@click.option("--num_layers", type=int, default=2)
@click.option("--hidden_dim", type=int, default=64)
@click.option("--graph_classification", type=bool, default=False)
@click.option("--batch_size", type=int, default=64)
@click.option("--with_sam", type=bool, default=True)
@click.option("--seed", type=int, default=1234)
@click.option("--use_wandb", type=bool, default=True)
@click.option("--neighbour_loader/--no_neighbour_loader", default=False)
@click.option("--num_hops", callback=process_items, default=4)
@click.option("--lr", type=float, default=0.01)
@click.option("--use_early_stopping", type=bool, default=False)
@click.option(
    "--base_optimizer",
    type=click.Choice(
        ["adam", "sgd"], case_sensitive=True
    ),
    default="adam",
    help="The base optimizer to use. Options are adam and sgd. Can be used in conjunction with SAM",
)
def main(
    dataset_type: str,
    dataset_name: str,
    fold_idx: int,
    num_layers: int,
    hidden_dim: int,
    graph_classification: bool,
    batch_size: int,
    with_sam: bool,
    seed: int,
    use_wandb: bool,
    base_optimizer: str,
    neighbour_loader: bool,
    num_hops: int,
    lr: float,
    use_early_stopping: bool,
):
    datamodule = get_dataset(dataset_type, dataset_name, fold_idx, batch_size,
                             neighbour_loader, num_hops)
    model = GCN(
        num_features=datamodule.num_features,
        num_classes=datamodule.num_classes,
        num_hidden=hidden_dim,
        num_hidden_layers=num_layers,
        graph_classification=graph_classification,
        with_sam=with_sam,
        seed=seed,
        base_optimizer=base_optimizer,
        lr=lr
    )

    wandb_logger = None

    if use_wandb:
        wandb_logger = WandbLogger(project="setup_tests", entity="r252_bel", log_model=True)

    trainer = Trainer(
        accelerator='auto',
        max_epochs=500,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=get_callbacks(use_early_stopping=use_early_stopping)
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
