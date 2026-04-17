"""
main.py
Full experiment runner — unified homo + hetero pipeline.

Loop structure:
    for dataset in HOMO_DATASETS:
        for backbone in HOMO_BACKBONES (GCN, GraphSAGE, GATv2):
            for variant in VARIANTS:
                for tau in TAU_VALUES:
                    for seed in SEEDS:
                        run_homo_experiment(...)

    for dataset in HETERO_DATASETS (IMDB, DBLP):
        for backbone in HETERO_BACKBONES (HAN, HGT):
            for variant in VARIANTS:
                for tau in TAU_VALUES:
                    for seed in SEEDS:
                        run_hetero_experiment(...)

Single unified CSV output.
All 8 performance plots + 7 timing plots generated at end.
"""

import os
import time
import traceback

import torch

import config
from utils.data_utils        import set_seed, load_dataset, make_link_splits
from utils.hetero_data_utils import load_hetero_dataset, make_hetero_link_splits
from utils.train_utils       import train_one_epoch, evaluate_split, EarlyStopper
from utils.logging_utils     import (
    init_csv, append_csv, build_result_row, make_run_dir,
    print_header, print_epoch, print_test_result, print_group_summary
)
from visualization.plots        import generate_performance_plots
from visualization.timing_plots import generate_timing_plots

# Homo backbones
from models.gcn            import GCNEncoder
from models.graphsage      import GraphSAGEEncoder
from models.gatv2          import GATv2Encoder

# Hetero backbones
from models.han            import HANEncoder
from models.hgt            import HGTEncoder

# Wrappers (backbone-agnostic)
from models.adaptive       import EntropyAdaptiveModel
from models.full_linegraph import FullLineGraphModel


# =============================================================================
# Model factory — homogeneous
# =============================================================================
def build_homo_model(backbone_name: str,
                     variant_name: str,
                     num_features: int,
                     num_classes: int,
                     tau: float,
                     device: torch.device) -> torch.nn.Module:

    hd = config.HIDDEN_DIM

    if backbone_name == 'GCN':
        encoder = GCNEncoder(
            num_layers    = config.NUM_LAYERS,
            input_dim     = num_features,
            hidden_dim    = hd,
            output_dim    = num_classes,
            final_dropout = config.FINAL_DROPOUT
        )
    elif backbone_name == 'GraphSAGE':
        encoder = GraphSAGEEncoder(
            num_layers     = config.NUM_LAYERS,
            num_mlp_layers = config.NUM_MLP_LAYERS,
            input_dim      = num_features,
            hidden_dim     = hd,
            output_dim     = num_classes,
            final_dropout  = config.FINAL_DROPOUT,
            learn_eps      = config.LEARN_EPS
        )
    elif backbone_name == 'GATv2':
        encoder = GATv2Encoder(
            num_layers    = config.NUM_LAYERS,
            input_dim     = num_features,
            hidden_dim    = hd,
            output_dim    = num_classes,
            num_heads     = config.GAT_HEADS,
            head_dim      = config.GAT_HEAD_DIM,
            final_dropout = config.FINAL_DROPOUT
        )
    else:
        raise ValueError(f'Unknown homo backbone: {backbone_name}')

    return _wrap_variant(encoder, variant_name, num_classes, tau, device)


# =============================================================================
# Model factory — heterogeneous
# =============================================================================
def build_hetero_model(backbone_name: str,
                       variant_name: str,
                       num_classes: int,
                       metadata: tuple,
                       primary_node: str,
                       tau: float,
                       device: torch.device) -> torch.nn.Module:

    hd = config.HIDDEN_DIM

    if backbone_name == 'HAN':
        encoder = HANEncoder(
            metadata    = metadata,
            hidden_dim  = hd,
            output_dim  = num_classes,
            target_node = primary_node,
            num_heads   = config.HAN_HEADS,
            num_layers  = 2,
            dropout     = config.FINAL_DROPOUT
        )
    elif backbone_name == 'HGT':
        encoder = HGTEncoder(
            metadata    = metadata,
            hidden_dim  = hd,
            output_dim  = num_classes,
            target_node = primary_node,
            num_heads   = config.HGT_HEADS,
            num_layers  = config.HGT_NUM_LAYERS,
            dropout     = config.FINAL_DROPOUT
        )
    else:
        raise ValueError(f'Unknown hetero backbone: {backbone_name}')

    return _wrap_variant(encoder, variant_name, num_classes, tau, device)


# =============================================================================
# Variant wrapper (shared by homo and hetero)
# =============================================================================
def _wrap_variant(encoder, variant_name, num_classes, tau, device):
    hd = config.HIDDEN_DIM

    if variant_name == 'Adaptive':
        model = EntropyAdaptiveModel(
            node_encoder    = encoder,
            hidden_dim      = hd,
            num_node_cls    = num_classes,
            tau             = tau,
            final_dropout   = config.FINAL_DROPOUT,
            edge_batch_size = config.EDGE_BATCH_SIZE,
            device          = device
        )
    elif variant_name == 'FullLineGraph':
        model = FullLineGraphModel(
            node_encoder  = encoder,
            hidden_dim    = hd,
            num_node_cls  = num_classes,
            final_dropout = config.FINAL_DROPOUT,
            device        = device
        )
    else:
        raise ValueError(f'Unknown variant: {variant_name}')

    return model.to(device)


# =============================================================================
# Training loop (shared by homo and hetero)
# =============================================================================
def _train_and_eval(model, data_tr, data_va, data_te,
                    dataset_name, model_name):
    """
    Shared training loop with early stopping.
    Returns (te_node, te_edge, te_timing, duration).
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    stopper  = EarlyStopper(patience=config.PATIENCE, alpha=config.ALPHA)
    t_start  = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        total_loss, nloss, eloss, aloss, tr_timing = train_one_epoch(
            model, data_tr, optimizer
        )

        if epoch % 10 == 0 or epoch == config.NUM_EPOCHS:
            va_node, va_edge, va_timing, _, _ = evaluate_split(
                model, data_va, split_name='val'
            )
            print_epoch(epoch, total_loss, nloss, eloss, aloss,
                        va_node, va_edge, va_timing)

            if stopper.update(va_node, va_edge, model):
                print(f'  Early stopping at epoch {epoch}')
                break

    stopper.restore(model)

    te_node, te_edge, te_timing, _, _ = evaluate_split(
        model, data_te, split_name='test'
    )
    print_test_result(dataset_name, model_name, te_node, te_edge, te_timing)

    duration = time.time() - t_start
    return te_node, te_edge, te_timing, duration


# =============================================================================
# Single homo experiment
# =============================================================================
def run_homo_experiment(dataset_name, backbone_name, variant_name,
                        tau, seed, device):
    set_seed(seed)

    ds_meta, data       = load_dataset(dataset_name, device, seed)
    data_tr, data_va, data_te = make_link_splits(data, seed=seed)

    model_name = f'{backbone_name}-{"Adaptive" if variant_name == "Adaptive" else "FullLG"}'
    model = build_homo_model(
        backbone_name = backbone_name,
        variant_name  = variant_name,
        num_features  = ds_meta.num_features,
        num_classes   = ds_meta.num_classes,
        tau           = tau,
        device        = device
    )

    te_node, te_edge, te_timing, duration = _train_and_eval(
        model, data_tr, data_va, data_te, dataset_name, model_name
    )

    te_timing['mean_entropy']  = te_timing.get('mean_entropy',  0.0)
    te_timing['pct_uncertain'] = te_timing.get('pct_uncertain', 0.0)

    return build_result_row(
        dataset    = dataset_name,
        model_name = model_name,
        backbone   = backbone_name,
        variant    = variant_name,
        seed       = seed,
        tau        = tau,
        duration   = duration,
        node_m     = te_node,
        edge_m     = te_edge,
        timing     = te_timing
    )


# =============================================================================
# Single hetero experiment
# =============================================================================
def run_hetero_experiment(dataset_name, backbone_name, variant_name,
                          tau, seed, device):
    set_seed(seed)

    ds_meta, hetero, primary = load_hetero_dataset(dataset_name, device, seed)

    data_tr, data_va, data_te = make_hetero_link_splits(
        hetero, primary, seed, device
    )

    model_name = f'{backbone_name}-{"Adaptive" if variant_name == "Adaptive" else "FullLG"}'
    model = build_hetero_model(
        backbone_name = backbone_name,
        variant_name  = variant_name,
        num_classes   = ds_meta.num_classes,
        metadata      = hetero.metadata(),
        primary_node  = primary,
        tau           = tau,
        device        = device
    )

    te_node, te_edge, te_timing, duration = _train_and_eval(
        model, data_tr, data_va, data_te, dataset_name, model_name
    )

    te_timing['mean_entropy']  = te_timing.get('mean_entropy',  0.0)
    te_timing['pct_uncertain'] = te_timing.get('pct_uncertain', 0.0)

    return build_result_row(
        dataset    = dataset_name,
        model_name = model_name,
        backbone   = backbone_name,
        variant    = variant_name,
        seed       = seed,
        tau        = tau,
        duration   = duration,
        node_m     = te_node,
        edge_m     = te_edge,
        timing     = te_timing
    )


# =============================================================================
# Main
# =============================================================================
def main():
    device = config.DEVICE
    print(f'\n{"=" * 80}')
    print(f'  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'{"=" * 80}\n')

    run_dir, ts = make_run_dir()
    csv_path    = os.path.join(run_dir, f'results_{ts}.csv')
    plots_dir   = os.path.join(run_dir, config.PLOTS_SUBDIR)
    timing_dir  = os.path.join(run_dir, 'timing_plots')

    from utils.logging_utils import build_result_row as _brr
    _sample  = _brr('', '', '', '', 0, 0, 0, {}, {}, {})
    all_cols = list(_sample.keys())

    init_csv(csv_path, all_cols)

    all_results = []
    failed      = []

    # ------------------------------------------------------------------
    # Block 1: Heterogeneous datasets  (IMDB + DBLP — HAN + HGT)
    # ------------------------------------------------------------------
    print(f'\n{"=" * 80}')
    print('  HETEROGENEOUS DATASETS  (HAN + HGT)')
    print(f'{"=" * 80}')

    for dataset_name in config.HETERO_DATASETS:
        for backbone_name in config.HETERO_BACKBONES:
            for variant_name in config.VARIANTS:
                model_display = (
                    f'{backbone_name}-'
                    f'{"Adaptive" if variant_name == "Adaptive" else "FullLG"}'
                )
                for tau in config.TAU_VALUES:
                    group = []
                    for seed in config.SEEDS:
                        print_header(dataset_name, model_display, tau, seed)
                        try:
                            result = run_hetero_experiment(
                                dataset_name, backbone_name, variant_name,
                                tau, seed, device
                            )
                            group.append(result)
                            all_results.append(result)
                            append_csv(csv_path, result, all_cols)
                        except Exception:
                            msg = (
                                f'FAILED: {dataset_name} | {model_display} | '
                                f'tau={tau} | seed={seed}\n{traceback.format_exc()}'
                            )
                            print(msg)
                            failed.append(msg)

                    if group:
                        print_group_summary(dataset_name, model_display, tau, group)

    # ------------------------------------------------------------------
    # Block 2: Homogeneous datasets
    # ------------------------------------------------------------------
    print(f'\n{"=" * 80}')
    print('  HOMOGENEOUS DATASETS')
    print(f'{"=" * 80}')

    for dataset_name in config.HOMO_DATASETS:
        for backbone_name in config.HOMO_BACKBONES:
            for variant_name in config.VARIANTS:
                model_display = (
                    f'{backbone_name}-'
                    f'{"Adaptive" if variant_name == "Adaptive" else "FullLG"}'
                )
                for tau in config.TAU_VALUES:
                    group = []
                    for seed in config.SEEDS:
                        print_header(dataset_name, model_display, tau, seed)
                        try:
                            result = run_homo_experiment(
                                dataset_name, backbone_name, variant_name,
                                tau, seed, device
                            )
                            group.append(result)
                            all_results.append(result)
                            append_csv(csv_path, result, all_cols)
                        except Exception:
                            msg = (
                                f'FAILED: {dataset_name} | {model_display} | '
                                f'tau={tau} | seed={seed}\n{traceback.format_exc()}'
                            )
                            print(msg)
                            failed.append(msg)

                    if group:
                        print_group_summary(dataset_name, model_display, tau, group)

    # ------------------------------------------------------------------
    # Failures summary
    # ------------------------------------------------------------------
    if failed:
        print(f'\n{"=" * 60}')
        print(f'  {len(failed)} run(s) failed.')
        fail_log = os.path.join(run_dir, 'failed_runs.txt')
        with open(fail_log, 'w') as fh:
            fh.write('\n\n'.join(failed))
        print(f'  Details: {fail_log}')

    print(f'\n{"=" * 80}')
    print(f'  All experiments done.  CSV: {csv_path}')
    print(f'{"=" * 80}\n')

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    if all_results:
        try:
            generate_performance_plots(csv_path, plots_dir)
        except Exception as e:
            print(f'  Performance plots failed: {e}')
        try:
            generate_timing_plots(csv_path, timing_dir)
        except Exception as e:
            print(f'  Timing plots failed: {e}')

    print(f'\n{"=" * 80}')
    print(f'  Performance plots : {plots_dir}')
    print(f'  Timing plots      : {timing_dir}')
    print(f'{"=" * 80}\n')


if __name__ == '__main__':
    main()