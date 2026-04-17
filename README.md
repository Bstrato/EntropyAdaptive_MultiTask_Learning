**Entropy-Adaptive Graph Neural Networks for Joint Node & Edge Classification**

This repository implements a unified GNN benchmarking framework that jointly performs node classification and link prediction across both homogeneous and heterogeneous graphs. The core contribution is an entropy-based edge routing mechanism that adaptively decides — per edge — whether to process it through a lightweight MLP or a full line graph encoder.

Key Ideas
Two model variants are compared across all datasets and backbones:

Adaptive — estimates prediction entropy for each edge and routes uncertain edges through a heavier line graph encoder, while confident edges use a fast MLP fallback. The uncertainty threshold is controlled by τ (tau).
FullLineGraph — constructs a complete line graph and processes all edges through the line graph encoder unconditionally, serving as the upper-bound baseline.

Five GNN backbones are supported:

Homogeneous: GCN, GraphSAGE, GATv2
Heterogeneous: HAN (with meta-paths), HGT

Seven benchmark datasets: Cora, CiteSeer, PubMed, Computers, Photo (homogeneous) and IMDB, DBLP (heterogeneous).

Experiment Design
Experiments sweep over 5 random seeds × 3 τ values {0.3, 0.5, 0.7} × all backbone–variant combinations, logging node and edge metrics (accuracy, AUC, precision, recall, F1) alongside granular per-stage timing breakdowns. Results are written to a unified CSV and visualized across 8 performance plots and 7 timing plots.

Repository Structure
main.py        — full experiment runner (homo + hetero pipeline)
config.py      — all hyperparameters, dataset routing, and column definitions
test.py        — smoke test: runs every combination for 20 epochs to catch errors before sbatch
models/        — backbone encoders + Adaptive / FullLineGraph wrappers
utils/         — data loading, train/eval loop, early stopping, logging
visualization/ — performance and timing plot generation

Quick Start
bash# Sanity-check all combinations before a full run
python test.py

# Full experiment sweep
python main.py
