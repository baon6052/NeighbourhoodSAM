# gnn_sam

## Installation instructions:

## Create and activate conda evironment with python=3.11.8  

`conda create --name py311 python=3.11.8`

`conda activate py311`

## Install torch 2.1.1 from https://pytorch.org/get-started/locally/

e.g.: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Install PyG from https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
e.g.: `pip install torch_geometric`

Make sure to also install the optional dependencies:
e.g.: `pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html`

## Install final packages:
pip install lightning==2.2.1 pytorch-lightning==2.2.1 wandb dgl