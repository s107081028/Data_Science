# 111061702 DS Hw3

## Required Packages
* torch_geometric 2.3.1
  * pip install torch_geometric
* torch-spline-conv 1.2.2
  * pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+$%7BCUDA%7D.html
* dgl
  * pip install dgl -f https://data.dgl.ai/wheels/repo.html

* scipy
* networkx
* pytorch
  * torch 2.0.1+cu118

* numpy
* random

## Run code
```python
python3 train.py --es_iters 100 --epochs 1000 --use_gpu
```
* SAGE and GAT models may not be on device
* Spline should use gpu, otherwise it may take very long time.

## Dataset
* Unknown graph data
  * Label Split:
    * Train: 60, Valid: 500, Test: 1000
* Semi-supervised
  * Pseudo Labels: Train + Pseudo
    * Train: 60, Valid: 500, Test: 1000, Pseudo: 18157
    * Train with train labels and pseudo labels, validate with valid labels, and predict on test