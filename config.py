"""
config.py
All hyperparameters, dataset configurations, model settings, and experiment parameters.
"""

import torch

# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Dataset Routing
# =============================================================================
# Heterogeneous datasets → HAN / HGT backbones (native hetero, no conversion)
HETERO_DATASETS = ['IMDB', 'DBLP']

# Homogeneous datasets  → GCN / GraphSAGE / GATv2 backbones
HOMO_DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']

# Full list (order defines experiment order)
DATASETS = HOMO_DATASETS + HETERO_DATASETS

# Datasets that need manual train/val/test mask creation (homo only)
DATASETS_WITHOUT_MASKS = ['Computers', 'Photo']

# Node split ratios for datasets without predefined masks
NODE_TRAIN_RATIO = 0.6
NODE_VAL_RATIO   = 0.2
NODE_TEST_RATIO  = 0.2  # remainder

# Primary node type per heterogeneous dataset
# Used for node classification and edge prediction targets
HETERO_PRIMARY_NODE = {
    'IMDB': 'movie',
    'DBLP': 'author',
}

# Meta-paths per dataset (for HAN)
HAN_METAPATHS = {
    'IMDB': [
        ['movie', 'to', 'actor',    'to', 'movie'],
        ['movie', 'to', 'director', 'to', 'movie'],
    ],
    'DBLP': [
        ['author', 'to', 'paper', 'to', 'author'],
        ['author', 'to', 'paper', 'to', 'term',  'to', 'paper', 'to', 'author'],
    ],
}

# =============================================================================
# Experiment Settings
# =============================================================================
SEEDS      = [32, 42, 56, 61, 78]
TAU_VALUES = [0.3, 0.5, 0.7]

# =============================================================================
# Model Variants
# =============================================================================

# Heterogeneous backbones
HETERO_BACKBONES = ['HAN', 'HGT']


# Homogeneous backbones
HOMO_BACKBONES   = ['GCN', 'GraphSAGE', 'GATv2']



VARIANTS = ['Adaptive', 'FullLineGraph']

# =============================================================================
# Architecture Hyperparameters
# =============================================================================
HIDDEN_DIM      = 64
NUM_LAYERS      = 3        # GNN depth
NUM_MLP_LAYERS  = 2        # MLP depth inside GNN layers
GAT_HEADS       = 4        # Number of attention heads for GATv2
GAT_HEAD_DIM    = 16       # Per-head dim → hidden = heads * head_dim = 64
FINAL_DROPOUT   = 0.5
LEARN_EPS       = True     # Learnable epsilon in GraphSAGE / GIN-style

# HAN / HGT specific
HAN_HEADS       = 8        # Attention heads in HAN semantic-level attention
HGT_HEADS       = 4        # Attention heads per layer in HGT
HGT_NUM_LAYERS  = 3        # HGT depth

# Edge batch size for uncertain-edge processing (avoids OOM on large graphs)
EDGE_BATCH_SIZE = 50_000

# =============================================================================
# Training Hyperparameters
# =============================================================================
LR           = 0.01
WEIGHT_DECAY = 5e-4
NUM_EPOCHS   = 100
PATIENCE     = 20          # Early stopping patience (checks every 10 epochs)

# Joint loss weights
ALPHA = 0.7   # weight for node classification loss
BETA  = 0.1   # weight for auxiliary entropy estimator loss
# Edge loss weight = (1 - ALPHA)

# =============================================================================
# Link Prediction Split
# =============================================================================
VAL_RATIO          = 0.1
TEST_RATIO         = 0.1
NEG_SAMPLING_RATIO = 1.0   # 1:1 pos:neg

# =============================================================================
# Paths
# =============================================================================
DATA_ROOT    = 'data'
RESULTS_DIR  = 'results'
PLOTS_SUBDIR = 'plots'

# =============================================================================
# CSV Column Definitions
# =============================================================================
PERF_COLS = [
    'dataset', 'model', 'backbone', 'variant', 'seed', 'tau', 'duration',
    # Node metrics
    'node_accuracy', 'node_auc', 'node_precision', 'node_recall', 'node_f1',
    # Edge metrics
    'edge_accuracy', 'edge_auc', 'edge_precision', 'edge_recall', 'edge_f1',
    # Entropy stats
    'mean_entropy', 'pct_uncertain',
]

TIMING_COLS = [
    'dataset', 'model', 'backbone', 'variant', 'seed', 'tau',
    # Granular timing (seconds)
    'time_node_encoder',
    'time_edge_feat_build',
    'time_entropy_estimate',
    'time_entropy_routing',
    'time_lg_build',
    'time_lg_encoder',
    'time_fallback_mlp',
    'time_edge_total',
    'time_total_forward',
    # Counts
    'num_uncertain_edges',
    'num_certain_edges',
    'num_linegraph_edges',
    'pct_uncertain',
]

ALL_COLS = PERF_COLS + [c for c in TIMING_COLS if c not in PERF_COLS]