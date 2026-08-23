# Reproducibility checklist

Record the following for each experiment:

- dataset and, for cross-domain runs, source and target dataset;
- way, shot, and meta-training tasks (`--way`, `--shot`, `--num_tasks`);
- fixed query count (`--qry`, default 20), seed (`--seed`, default 42), and episodes;
- encoder (`sgc` or `gcn`);
- SGC propagation depth: `K=2` for in-domain experiments and `K=1` for cross-domain experiments; use the same value for the encoder and scorer within an experiment;
- `USE_C2_MODULE`, adaptive-alpha, and degree-prior switches;
- Python, PyTorch, DGL, and CUDA versions;
- output metrics and the command used to produce them.

For the complete model, keep the default `USE_C2_MODULE = True` in the relevant script. For the `w/o PSRM` ablation, set it to `False`.
