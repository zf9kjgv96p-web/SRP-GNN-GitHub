# SRP-GNN

Code for **SRP-GNN: Prototype Super-Resolution via Label-Free Query-Conditioned Reconstruction for Graph Few-Shot Node Classification**.

The implementation contains the degree-aware graph encoder, label-free query-conditioned prototype reconstruction (PSRM), and the decoupled class-consistent dual-level Mixup strategy described in the manuscript.

## Repository layout

```text
train.py                 In-domain episodic training and evaluation
train_cross_domain.py    Cross-domain training and evaluation
model.py, layers.py      Graph encoders and neural layers
utils.py                 Dataset loading, normalization, and episode sampling
few_shot_data/           Local dataset files (see DATA_README.md)
requirements.txt         Python dependencies
LICENSE                  MIT license for this implementation
```

The public release does not include local experiment records. Episodes are sampled at run time from the released data using the specified random seed.

## Environment

The code was developed with Python 3.10, PyTorch, DGL, NumPy, SciPy, and scikit-learn. Install the dependencies in an environment compatible with your operating system and CUDA version:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch and DGL binary wheels are platform- and CUDA-version dependent. If installation from `requirements.txt` fails for these packages, install matching wheels from their official documentation first, then install the remaining requirements.

## Data

The preprocessed benchmark files required by this implementation are included in `few_shot_data/`. See [DATA_README.md](DATA_README.md) for the expected filenames and setup instructions.

The loaders resolve paths relative to this repository. Amazon/DBLP files are already placed directly under `few_shot_data/`. Before running a CoraFull experiment, extract `few_shot_data/corafull/cora_full.zip` into `few_shot_data/corafull/`.

## In-domain experiments

The standard setting uses 20 query nodes per class and seed 42, which are fixed defaults in the scripts. Specify only the dataset, way, shot, and number of meta-training tasks:

```bash
python train.py --dataset Amazon_clothing --way 5 --shot 5 --num_tasks 5
python train.py --dataset Amazon_electronics --way 5 --shot 5 --num_tasks 5
python train.py --dataset dblp --way 5 --shot 5 --num_tasks 5
python train.py --dataset corafull --way 5 --shot 5 --num_tasks 5
```

Use `--encoder gcn` to select the GCN encoder; the default is `sgc`. Use `--wodegree` only for the degree-prior ablation.

## Cross-domain experiments

For a source-to-target experiment, specify both datasets:

```bash
python train_cross_domain.py --dataset Amazon_clothing --dataset_cr corafull --way 5 --shot 5 --num_tasks 5
python train_cross_domain.py --dataset corafull --dataset_cr Amazon_clothing --way 5 --shot 5 --num_tasks 5
```

## Model and ablations

The `USE_C2_MODULE` flag controls PSRM in both training scripts:

```python
USE_C2_MODULE = True   # complete SRP-GNN, including PSRM
USE_C2_MODULE = False  # w/o PSRM ablation: support-only prototypes
```

The checked-in default is `True` so a fresh checkout runs the complete SRP-GNN model. Set it to `False` only when reproducing the `w/o PSRM` ablation. Other switches control adaptive alpha and degree-related ablations; record their values with each experiment.

## Propagation depth

For the reported SGC-based experiments, the propagation depth is set to `K=2` for in-domain experiments and `K=1` for cross-domain experiments. Within an experiment, the encoder and scorer use the same propagation depth. These settings should be retained when reproducing the corresponding results.

## Reproducibility notes

The scripts seed Python, NumPy, and PyTorch with `--seed`. Results can still vary across hardware, CUDA, and library versions. Report the dataset split, seed, encoder, number of episodes/tasks, and all ablation switches when comparing runs. Numerical baseline values reported in the manuscript should be treated as published comparison results; rerun this code for SRP-GNN and ablation results unless the manuscript states otherwise.

## Citation

If you use this implementation, please cite the SRP-GNN manuscript and the original sources of all benchmark datasets and baseline methods listed in the manuscript.

## License

This implementation is released under the [MIT License](LICENSE).
