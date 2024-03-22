import argparse


parser = argparse.ArgumentParser(description="Dataset Loader")
parser.add_argument(
    "--dataset_type",
    type=str,
    choices=["node_homo", "node_hetero", "graph_homo"],
    default="node_homo",
    help="The type of dataset to use. Options are node_homo, node_hetero, and graph_homo.",
)
parser.add_argument("--dataset_name", type=str, default="CORA", help="The name of the dataset.")
parser.add_argument("--fold_idx", type=int, default=0, help="Index of the fold for cross-validation.")
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers.")
parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of hidden layers.")
parser.add_argument("--graph_classification", action="store_true", help="Enable graph classification.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
parser.add_argument("--with_sam", action="store_true", help="Use SAM optimization.")
parser.add_argument("--seed", type=int, default=1234, help="Random seed for initialization.")

parser.add_argument("--neighbour_loader", action="store_true")
parser.add_argument("--num_hops", type=int, default=4)

parser.add_argument("--use_wandb", action="store_true", help="Enable logging to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, help="Choose wandb project")
parser.add_argument("--checkpoint_reference_1", type=str, help="Choose wandb run 1 for loss interpolation")
parser.add_argument("--checkpoint_reference_2", type=str, help="Choose wandb run 2 for loss interpolation")

parser.add_argument(
    "--base_optimizer",
    type=str,
    choices=["adam", "sgd"],
    default="adam",
    help="The base optimizer to use. Can be used in conjunction with SAM.",
)