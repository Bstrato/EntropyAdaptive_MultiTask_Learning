"""
test_pipeline.py
Smoke test for the full pipeline before submitting sbatch.

Runs every dataset × backbone × variant combination for 20 epochs,
1 seed, 1 tau value. No plots are generated.

Purpose:
  - Catch any runtime errors across all datasets and models
  - Verify homo and hetero pipelines both execute end-to-end
  - Confirm all imports, data loading, forward passes, and metric
    computation work correctly

Expected output:
  - Each run prints epoch logs and a test result line
  - PASSED / FAILED summary at the end
  - Zero failures = safe to submit full sbatch

Usage:
    python test_pipeline.py
"""

import os
import sys
import time
import traceback

import torch

import config
from utils.data_utils        import set_seed, load_dataset, make_link_splits
from utils.hetero_data_utils import load_hetero_dataset, make_hetero_link_splits
from utils.train_utils       import train_one_epoch, evaluate_split, EarlyStopper
from utils.logging_utils     import print_header, print_test_result

from models.gcn            import GCNEncoder
from models.graphsage      import GraphSAGEEncoder
from models.gatv2          import GATv2Encoder
from models.han            import HANEncoder
from models.hgt            import HGTEncoder
from models.adaptive       import EntropyAdaptiveModel
from models.full_linegraph import FullLineGraphModel

# =============================================================================
# Test settings — reduced for speed
# =============================================================================
TEST_EPOCHS   = 20       # max epochs per run
TEST_SEED     = 42       # single seed
TEST_TAU      = 0.5      # single tau
TEST_PATIENCE = 5        # early stopping patience (checks every 5 epochs)
VAL_INTERVAL  = 5        # validate every 5 epochs

# All datasets to test
TEST_HOMO_DATASETS   = config.HOMO_DATASETS    # Cora, CiteSeer, PubMed, Computers, Photo
TEST_HETERO_DATASETS = config.HETERO_DATASETS  # IMDB, DBLP
TEST_HOMO_BACKBONES  = config.HOMO_BACKBONES   # GCN, GraphSAGE, GATv2
TEST_HETERO_BACKBONES= config.HETERO_BACKBONES # HAN, HGT
TEST_VARIANTS        = config.VARIANTS         # Adaptive, FullLineGraph

device = config.DEVICE


# =============================================================================
# Model builders (mirrors main.py exactly)
# =============================================================================
def build_homo_model(backbone_name, variant_name, num_features, num_classes, tau):
    hd = config.HIDDEN_DIM
    if backbone_name == 'GCN':
        encoder = GCNEncoder(config.NUM_LAYERS, num_features, hd,
                             num_classes, config.FINAL_DROPOUT)
    elif backbone_name == 'GraphSAGE':
        encoder = GraphSAGEEncoder(config.NUM_LAYERS, config.NUM_MLP_LAYERS,
                                   num_features, hd, num_classes,
                                   config.FINAL_DROPOUT, config.LEARN_EPS)
    elif backbone_name == 'GATv2':
        encoder = GATv2Encoder(config.NUM_LAYERS, num_features, hd,
                               num_classes, config.GAT_HEADS,
                               config.GAT_HEAD_DIM, config.FINAL_DROPOUT)
    if variant_name == 'Adaptive':
        model = EntropyAdaptiveModel(encoder, hd, num_classes, tau,
                                     config.FINAL_DROPOUT,
                                     config.EDGE_BATCH_SIZE, device)
    else:
        model = FullLineGraphModel(encoder, hd, num_classes,
                                   config.FINAL_DROPOUT, device)
    return model.to(device)


def build_hetero_model(backbone_name, variant_name, num_classes,
                       metadata, primary_node, tau):
    hd = config.HIDDEN_DIM
    if backbone_name == 'HAN':
        encoder = HANEncoder(metadata, hd, num_classes, primary_node,
                             config.HAN_HEADS, 2, config.FINAL_DROPOUT)
    elif backbone_name == 'HGT':
        encoder = HGTEncoder(metadata, hd, num_classes, primary_node,
                             config.HGT_HEADS, config.HGT_NUM_LAYERS,
                             config.FINAL_DROPOUT)
    if variant_name == 'Adaptive':
        model = EntropyAdaptiveModel(encoder, hd, num_classes, tau,
                                     config.FINAL_DROPOUT,
                                     config.EDGE_BATCH_SIZE, device)
    else:
        model = FullLineGraphModel(encoder, hd, num_classes,
                                   config.FINAL_DROPOUT, device)
    return model.to(device)


# =============================================================================
# Shared training loop
# =============================================================================
def _run_training(model, data_tr, data_va, data_te, label):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LR,
                                 weight_decay=config.WEIGHT_DECAY)
    stopper = EarlyStopper(patience=TEST_PATIENCE, alpha=config.ALPHA)

    for epoch in range(1, TEST_EPOCHS + 1):
        train_one_epoch(model, data_tr, optimizer)

        if epoch % VAL_INTERVAL == 0 or epoch == TEST_EPOCHS:
            va_node, va_edge, _, _, _ = evaluate_split(model, data_va, 'val')
            print(f'  Epoch {epoch:03d} | '
                  f'Val Node F1={va_node["f1"]:.4f} '
                  f'Edge AUC={va_edge["auc"]:.4f}')
            if stopper.update(va_node, va_edge, model):
                print(f'  Early stop at epoch {epoch}')
                break

    stopper.restore(model)
    te_node, te_edge, te_timing, _, _ = evaluate_split(model, data_te, 'test')
    print(f'  TEST | Node F1={te_node["f1"]:.4f}  '
          f'Edge AUC={te_edge["auc"]:.4f}  '
          f'Time={te_timing["time_total_forward"]:.4f}s')
    return True


# =============================================================================
# Single homo test run
# =============================================================================
def test_homo(dataset_name, backbone_name, variant_name):
    set_seed(TEST_SEED)
    ds_meta, data             = load_dataset(dataset_name, device, TEST_SEED)
    data_tr, data_va, data_te = make_link_splits(data, seed=TEST_SEED)
    model = build_homo_model(backbone_name, variant_name,
                             ds_meta.num_features, ds_meta.num_classes,
                             TEST_TAU)
    label = f'{dataset_name} | {backbone_name}-{variant_name}'
    return _run_training(model, data_tr, data_va, data_te, label)


# =============================================================================
# Single hetero test run
# =============================================================================
def test_hetero(dataset_name, backbone_name, variant_name):
    set_seed(TEST_SEED)
    ds_meta, hetero, primary  = load_hetero_dataset(dataset_name, device, TEST_SEED)
    data_tr, data_va, data_te = make_hetero_link_splits(hetero, primary,
                                                         TEST_SEED, device)
    model = build_hetero_model(backbone_name, variant_name,
                               ds_meta.num_classes,
                               hetero.metadata(), primary,
                               TEST_TAU)
    label = f'{dataset_name} | {backbone_name}-{variant_name}'
    return _run_training(model, data_tr, data_va, data_te, label)


# =============================================================================
# Main test runner
# =============================================================================
def main():
    print(f'\n{"=" * 70}')
    print(f'  PIPELINE SMOKE TEST')
    print(f'  Device  : {device}')
    print(f'  Epochs  : {TEST_EPOCHS}  |  Seed: {TEST_SEED}  |  Tau: {TEST_TAU}')
    print(f'{"=" * 70}\n')

    passed = []
    failed = []
    t_start = time.time()

    # ------------------------------------------------------------------
    # Homo tests
    # ------------------------------------------------------------------
    print(f'{"=" * 70}')
    print('  HOMOGENEOUS TESTS')
    print(f'{"=" * 70}')

    for dataset_name in TEST_HOMO_DATASETS:
        for backbone_name in TEST_HOMO_BACKBONES:
            for variant_name in TEST_VARIANTS:
                label = f'{dataset_name} | {backbone_name}-{variant_name}'
                print(f'\n--- {label} ---')
                try:
                    test_homo(dataset_name, backbone_name, variant_name)
                    passed.append(label)
                    print(f'  [PASSED]')
                except Exception:
                    failed.append((label, traceback.format_exc()))
                    print(f'  [FAILED]')
                    print(traceback.format_exc())

    # ------------------------------------------------------------------
    # Hetero tests
    # ------------------------------------------------------------------
    print(f'\n{"=" * 70}')
    print('  HETEROGENEOUS TESTS')
    print(f'{"=" * 70}')

    for dataset_name in TEST_HETERO_DATASETS:
        for backbone_name in TEST_HETERO_BACKBONES:
            for variant_name in TEST_VARIANTS:
                label = f'{dataset_name} | {backbone_name}-{variant_name}'
                print(f'\n--- {label} ---')
                try:
                    test_hetero(dataset_name, backbone_name, variant_name)
                    passed.append(label)
                    print(f'  [PASSED]')
                except Exception:
                    failed.append((label, traceback.format_exc()))
                    print(f'  [FAILED]')
                    print(traceback.format_exc())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    total   = len(passed) + len(failed)

    print(f'\n{"=" * 70}')
    print(f'  SMOKE TEST SUMMARY')
    print(f'  Total   : {total}')
    print(f'  Passed  : {len(passed)}')
    print(f'  Failed  : {len(failed)}')
    print(f'  Time    : {elapsed:.1f}s')
    print(f'{"=" * 70}')

    if failed:
        print(f'\n  FAILED RUNS:')
        for label, tb in failed:
            print(f'  - {label}')
        print()
        print('  Fix all failures before submitting sbatch.')
        sys.exit(1)
    else:
        print()
        print('  ALL PASSED — safe to submit sbatch.')
        sys.exit(0)


if __name__ == '__main__':
    main()