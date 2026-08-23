# Dataset notes

SRP-GNN uses four public graph benchmarks: Amazon-Clothing, Amazon-Electronics, CoraFull, and DBLP. The preprocessed files used by this implementation are included in this repository. The manuscript cites the original sources for these datasets.

## Expected layout

```text
few_shot_data/
  Amazon_clothing_network
  Amazon_clothing_train.mat
  Amazon_clothing_test.mat
  Amazon_electronics_network
  Amazon_electronics_train.mat
  Amazon_electronics_test.mat
  dblp_network
  dblp_train.mat
  dblp_test.mat
  corafull/
    ... CoraFull files readable by DGL ...
```

The Amazon and DBLP `.mat` files contain node attributes, labels, and indices expected by `utils.load_data`. Network files are tab-separated edge lists. CoraFull is loaded through DGL's `CoraFullDataset` using the local `few_shot_data/corafull` directory.

Before running a CoraFull experiment, extract `few_shot_data/corafull/cora_full.zip` into `few_shot_data/corafull/`. The repository excludes local experiment logs, derived result files, and DGL-generated cache files.
