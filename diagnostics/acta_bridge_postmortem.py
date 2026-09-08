"""
Step 10.1 — ACTA semantic-to-DP bridge post-mortem.

Question
--------
At the SAME source-initialized representation theta_S and for the SAME
source-reference / target feature costs, does ACTA move soft-DP edge
occupancy toward transitions preferred by the frozen source semantic teacher?

Primary diagnostic
------------------
For class c, define normalized teacher score

    q_c(i,j,m) = g_c(r_ijm) / rho_c

and soft edge occupancy

    E(i,j,m) = - d D_soft / d B(i,j,m),

because ACTA's DP uses R <- ... - B.

Semantic path quality is

    Q = sum(E * q) / sum(E).

We compare Q_ACTA and Q_UTA using exactly the same feature cost.
UTA is represented by a differentiable all-zero semantic bonus tensor;
this is numerically checked against semantic_bonus=None.

Target labels are loaded by the benchmark dataset object but are NEVER used.
The diagnostic runs at theta_S (before adaptation) to isolate the semantic
path intervention from representation drift caused by Stage-B training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from algorithms.ACTA import ACTA
from algorithms.components.acta_aligner import (
    cosine_feature_cost,
    resample_temporal_features,
    scale_feature_cost,
    transition_aware_soft_dp,
)
from algorithms.utils import fix_randomness
from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------


def parse_scenarios(text: str) -> List[Tuple[str, str]]:
    scenarios: List[Tuple[str, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid scenario '{item}'. Use source:target, e.g. 9:18."
            )
        scenarios.append((parts[0].strip(), parts[1].strip()))
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    return scenarios


def parse_seeds(text: str) -> List[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


# -----------------------------------------------------------------------------
# Edge occupancy diagnostic
# -----------------------------------------------------------------------------


@torch.enable_grad()
def edge_occupancy_from_bonus(
    path_cost: torch.Tensor,
    semantic_bonus: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return soft transition occupancy

        E = - d D_soft / d B

    and detached terminal cost.

    path_cost is treated as fixed evidence. Only B receives gradients.
    """

    if semantic_bonus.shape != (*path_cost.shape, 4):
        raise ValueError(
            "semantic_bonus must have shape [N,Ls,Lt,4] matching path_cost."
        )

    bonus = semantic_bonus.detach().clone().requires_grad_(True)
    fixed_path_cost = path_cost.detach()

    terminal = transition_aware_soft_dp(
        feature_cost=fixed_path_cost,
        semantic_bonus=bonus,
        gamma=gamma,
        return_table=False,
    )

    grad_bonus = torch.autograd.grad(
        terminal.sum(),
        bonus,
        create_graph=False,
        retain_graph=False,
    )[0]

    occupancy = (-grad_bonus).detach()

    # Each transition is used with probability in [0,1].
    tol = 1e-3
    min_value = float(occupancy.min().item())
    max_value = float(occupancy.max().item())

    if min_value < -tol or max_value > 1.0 + tol:
        raise RuntimeError(
            "Transition occupancy outside expected range: "
            f"min={min_value:.6g}, max={max_value:.6g}."
        )

    occupancy = occupancy.clamp(0.0, 1.0)
    return occupancy, terminal.detach()


def semantic_quality(
    edge_occupancy: torch.Tensor,
    normalized_scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Q = sum(E*q) / sum(E), one value per source-target pair.

    edge_occupancy: [N,G,G,4]
    normalized_scores: [N,G,G,4]
    returns: [N]
    """

    if edge_occupancy.shape != normalized_scores.shape:
        raise ValueError("Occupancy / semantic-score shape mismatch.")

    dims = (-3, -2, -1)
    mass = edge_occupancy.sum(dim=dims).clamp_min(eps)
    score = (edge_occupancy * normalized_scores).sum(dim=dims) / mass
    return score


def relative_edge_shift(
    uta_edges: torch.Tensor,
    acta_edges: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalized L1 change in transition occupancy, one value per pair.
    This is only an intervention-size diagnostic, not the primary metric.
    """

    dims = (-3, -2, -1)
    denom = uta_edges.sum(dim=dims).clamp_min(eps)
    return (acta_edges - uta_edges).abs().sum(dim=dims) / denom


# -----------------------------------------------------------------------------
# Model / data helpers
# -----------------------------------------------------------------------------


def build_algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
    # Only attributes ACTA and the benchmark loader need are provided.
    return SimpleNamespace(
        lr=args.lr,
        weight_decay=args.weight_decay,
        acta_beta=args.beta,
        acta_lambda=args.lambda_sem,
        acta_gamma=args.gamma,
        acta_ema=args.ema,
        acta_mode="ACTA",
        acta_k=args.k,
        source_model_root=args.source_model_root,
        semantic_root=args.semantic_root,
        bs=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
    )


def make_full_train_loader(train_dataset, batch_size: int) -> DataLoader:
    """
    Deterministic, complete traversal of target TRAIN data.
    Unlike adaptation training, diagnostic coverage does not drop the tail.
    """

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


@torch.no_grad()
def build_pair_features(
    algorithm: ACTA,
    target_temporal_features: torch.Tensor,
    class_id: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reproduce ACTA's B x K source-reference / target pairing exactly,
    except the encoder is deliberately in eval mode for deterministic
    source-initialized diagnostics.
    """

    reference_x, reference_y, reference_indices = algorithm.sample_class_references(
        class_id=class_id,
        k=k,
        return_indices=True,
    )

    if not torch.all(reference_y == int(class_id)):
        raise RuntimeError("Wrong-class source reference in diagnostic.")

    reference_features = algorithm.feature_extractor.forward_features(reference_x)

    batch_size = int(target_temporal_features.shape[0])
    _, feature_dim, source_length = reference_features.shape
    target_length = int(target_temporal_features.shape[-1])

    source_pairs = (
        reference_features[None, :, :, :]
        .expand(batch_size, k, feature_dim, source_length)
        .reshape(batch_size * k, feature_dim, source_length)
    )

    target_pairs = (
        target_temporal_features[:, None, :, :]
        .expand(batch_size, k, feature_dim, target_length)
        .reshape(batch_size * k, feature_dim, target_length)
    )

    return source_pairs, target_pairs, reference_indices


# -----------------------------------------------------------------------------
# One class / one target batch
# -----------------------------------------------------------------------------


def diagnose_class_batch(
    algorithm: ACTA,
    target_temporal_features: torch.Tensor,
    class_id: int,
    k: int,
    gamma: float,
    lambda_sem: float,
    run_uta_none_sanity: bool,
) -> Dict[str, torch.Tensor]:

    source_pairs, target_pairs, reference_indices = build_pair_features(
        algorithm=algorithm,
        target_temporal_features=target_temporal_features,
        class_id=class_id,
        k=k,
    )

    semantic_length = int(algorithm.semantic_bank.semantic_temporal_length)

    with torch.no_grad():
        source_grid = resample_temporal_features(source_pairs, semantic_length)
        target_grid = resample_temporal_features(target_pairs, semantic_length)
        feature_cost = cosine_feature_cost(source_grid, target_grid)
        scaled_feature_cost, _ = scale_feature_cost(feature_cost)

        q_single = algorithm.semantic_bank.normalized_transition_scores(
            class_id=class_id,
            source_length=semantic_length,
            target_length=semantic_length,
            device=feature_cost.device,
            dtype=feature_cost.dtype,
        )

        q = q_single.unsqueeze(0).expand(
            scaled_feature_cost.shape[0], -1, -1, -1
        )

        acta_bonus_single = algorithm.semantic_bank.semantic_bonus(
            class_id=class_id,
            reliability_class_id=class_id,
            source_length=semantic_length,
            target_length=semantic_length,
            lambda_sem=lambda_sem,
            device=feature_cost.device,
            dtype=feature_cost.dtype,
        )

        acta_bonus = acta_bonus_single.unsqueeze(0).expand(
            scaled_feature_cost.shape[0], -1, -1, -1
        )

        zero_bonus = torch.zeros_like(acta_bonus)

    uta_edges, uta_terminal_zero = edge_occupancy_from_bonus(
        path_cost=scaled_feature_cost,
        semantic_bonus=zero_bonus,
        gamma=gamma,
    )

    acta_edges, acta_terminal = edge_occupancy_from_bonus(
        path_cost=scaled_feature_cost,
        semantic_bonus=acta_bonus,
        gamma=gamma,
    )

    if run_uta_none_sanity:
        with torch.no_grad():
            uta_terminal_none = transition_aware_soft_dp(
                feature_cost=scaled_feature_cost.detach(),
                semantic_bonus=None,
                gamma=gamma,
                return_table=False,
            )

        max_delta = float(
            (uta_terminal_zero - uta_terminal_none).abs().max().item()
        )
        if max_delta > 1e-6:
            raise RuntimeError(
                "Zero semantic tensor is not numerically equivalent to UTA: "
                f"max terminal delta={max_delta:.8g}."
            )

    q_uta_pair = semantic_quality(uta_edges, q)
    q_acta_pair = semantic_quality(acta_edges, q)
    shift_pair = relative_edge_shift(uta_edges, acta_edges)

    batch_size = int(target_temporal_features.shape[0])

    # [B*K] -> [B,K] -> [B]
    q_uta = q_uta_pair.reshape(batch_size, k).mean(dim=1)
    q_acta = q_acta_pair.reshape(batch_size, k).mean(dim=1)
    edge_shift = shift_pair.reshape(batch_size, k).mean(dim=1)

    return {
        "q_uta": q_uta.detach(),
        "q_acta": q_acta.detach(),
        "delta_q": (q_acta - q_uta).detach(),
        "edge_shift": edge_shift.detach(),
        "reference_indices": reference_indices.detach().cpu(),
        "acta_terminal": acta_terminal.detach(),
    }


# -----------------------------------------------------------------------------
# One source-target seed
# -----------------------------------------------------------------------------


def run_seed(
    args: argparse.Namespace,
    source_id: str,
    target_id: str,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:

    fix_randomness(seed)

    device = torch.device(args.device)
    configs = get_dataset_class(args.dataset)()
    model_args = build_algorithm_args(args)

    data_path = str(Path(args.data_root) / args.dataset)

    src_train_dl, _ = data_generator(data_path, str(source_id), model_args)
    trg_train_dl, _ = data_generator(data_path, str(target_id), model_args)

    algorithm = ACTA(configs, device, model_args)
    algorithm.to(device)
    algorithm.configure_source_context(
        dataset_name=args.dataset,
        source_id=str(source_id),
        seed=int(seed),
        source_model_root=args.source_model_root,
        semantic_root=args.semantic_root,
    )
    algorithm.attach_source_reference_pool(
        source_train_dataset=src_train_dl.dataset,
        reference_seed=int(seed),
    )

    # Deliberately diagnose the common source checkpoint theta_S.
    # Eval mode removes dropout stochasticity from the correspondence test.
    algorithm.eval()
    algorithm.feature_extractor.eval()
    algorithm.classifier.eval()
    algorithm.ema_feature_extractor.eval()
    algorithm.ema_classifier.eval()
    algorithm.semantic_bank.eval()

    target_loader = make_full_train_loader(
        trg_train_dl.dataset,
        batch_size=args.bs,
    )

    weighted_q_uta_all: List[torch.Tensor] = []
    weighted_q_acta_all: List[torch.Tensor] = []
    weighted_delta_all: List[torch.Tensor] = []
    weighted_shift_all: List[torch.Tensor] = []

    # Accumulators for class-level reporting.
    class_q_uta: Dict[int, List[torch.Tensor]] = {
        c: [] for c in range(configs.num_classes)
    }
    class_q_acta: Dict[int, List[torch.Tensor]] = {
        c: [] for c in range(configs.num_classes)
    }
    class_delta: Dict[int, List[torch.Tensor]] = {
        c: [] for c in range(configs.num_classes)
    }
    class_shift: Dict[int, List[torch.Tensor]] = {
        c: [] for c in range(configs.num_classes)
    }

    sanity_pending = True
    processed_batches = 0

    for batch_index, (trg_x, _trg_y_unused) in enumerate(target_loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        trg_x = trg_x.float().to(device)

        with torch.no_grad():
            target_probabilities = algorithm.ema_probabilities(trg_x)
            target_temporal_features = algorithm.feature_extractor.forward_features(trg_x)

        q_uta_by_class: List[torch.Tensor] = []
        q_acta_by_class: List[torch.Tensor] = []
        delta_by_class: List[torch.Tensor] = []
        shift_by_class: List[torch.Tensor] = []

        for class_id in range(configs.num_classes):
            result = diagnose_class_batch(
                algorithm=algorithm,
                target_temporal_features=target_temporal_features,
                class_id=class_id,
                k=args.k,
                gamma=args.gamma,
                lambda_sem=args.lambda_sem,
                run_uta_none_sanity=sanity_pending,
            )
            sanity_pending = False

            q_uta_by_class.append(result["q_uta"])
            q_acta_by_class.append(result["q_acta"])
            delta_by_class.append(result["delta_q"])
            shift_by_class.append(result["edge_shift"])

            class_q_uta[class_id].append(result["q_uta"].cpu())
            class_q_acta[class_id].append(result["q_acta"].cpu())
            class_delta[class_id].append(result["delta_q"].cpu())
            class_shift[class_id].append(result["edge_shift"].cpu())

        # [C tensors of B] -> [B,C]
        q_uta_matrix = torch.stack(q_uta_by_class, dim=1)
        q_acta_matrix = torch.stack(q_acta_by_class, dim=1)
        delta_matrix = torch.stack(delta_by_class, dim=1)
        shift_matrix = torch.stack(shift_by_class, dim=1)

        probabilities = target_probabilities.detach()

        weighted_q_uta = (probabilities * q_uta_matrix).sum(dim=1)
        weighted_q_acta = (probabilities * q_acta_matrix).sum(dim=1)
        weighted_delta = (probabilities * delta_matrix).sum(dim=1)
        weighted_shift = (probabilities * shift_matrix).sum(dim=1)

        weighted_q_uta_all.append(weighted_q_uta.cpu())
        weighted_q_acta_all.append(weighted_q_acta.cpu())
        weighted_delta_all.append(weighted_delta.cpu())
        weighted_shift_all.append(weighted_shift.cpu())

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("No target batches were processed.")

    weighted_q_uta = torch.cat(weighted_q_uta_all)
    weighted_q_acta = torch.cat(weighted_q_acta_all)
    weighted_delta = torch.cat(weighted_delta_all)
    weighted_shift = torch.cat(weighted_shift_all)

    scenario_row: Dict[str, float] = {
        "scenario": f"{source_id}->{target_id}",
        "source": str(source_id),
        "target": str(target_id),
        "seed": int(seed),
        "n_target": int(weighted_delta.numel()),
        "weighted_q_uta": float(weighted_q_uta.mean().item()),
        "weighted_q_acta": float(weighted_q_acta.mean().item()),
        "weighted_delta_q": float(weighted_delta.mean().item()),
        "delta_q_positive_fraction": float((weighted_delta > 0).float().mean().item()),
        "weighted_edge_shift": float(weighted_shift.mean().item()),
    }

    class_rows: List[Dict[str, float]] = []
    for class_id in range(configs.num_classes):
        q_u = torch.cat(class_q_uta[class_id])
        q_a = torch.cat(class_q_acta[class_id])
        dq = torch.cat(class_delta[class_id])
        sh = torch.cat(class_shift[class_id])

        class_rows.append(
            {
                "scenario": f"{source_id}->{target_id}",
                "source": str(source_id),
                "target": str(target_id),
                "seed": int(seed),
                "class_id": int(class_id),
                "kappa": float(algorithm.semantic_bank.class_kappa(class_id).item()),
                "q_uta": float(q_u.mean().item()),
                "q_acta": float(q_a.mean().item()),
                "delta_q": float(dq.mean().item()),
                "delta_q_positive_fraction": float((dq > 0).float().mean().item()),
                "edge_shift": float(sh.mean().item()),
            }
        )

    return scenario_row, class_rows


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[Dict[str, float]]) -> None:
    print("\n" + "=" * 78)
    print("STEP 10.1 SUMMARY — semantic edge quality at common theta_S")
    print("=" * 78)

    by_scenario: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    all_deltas: List[float] = []

    for scenario, scenario_rows in by_scenario.items():
        deltas = torch.tensor(
            [float(r["weighted_delta_q"]) for r in scenario_rows],
            dtype=torch.float64,
        )
        positive_fractions = torch.tensor(
            [float(r["delta_q_positive_fraction"]) for r in scenario_rows],
            dtype=torch.float64,
        )
        shifts = torch.tensor(
            [float(r["weighted_edge_shift"]) for r in scenario_rows],
            dtype=torch.float64,
        )

        all_deltas.extend(deltas.tolist())

        print(
            f"{scenario:>10s} | "
            f"mean ΔQ={deltas.mean().item():+.6f} | "
            f"seed-positive={int((deltas > 0).sum().item())}/{len(deltas)} | "
            f"sample-positive≈{positive_fractions.mean().item():.3f} | "
            f"edge-shift={shifts.mean().item():.4f}"
        )

    overall = torch.tensor(all_deltas, dtype=torch.float64)
    print("-" * 78)
    print(
        f"OVERALL    | mean ΔQ={overall.mean().item():+.6f} | "
        f"seed-positive={int((overall > 0).sum().item())}/{len(overall)}"
    )
    print("=" * 78)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ACTA Step 10.1 semantic-to-DP bridge post-mortem"
    )

    parser.add_argument("--dataset", default="UCIHAR")
    parser.add_argument(
        "--scenarios",
        default="9:18,12:16,23:13",
        help="Comma-separated source:target pairs.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--data_root", default="data")
    parser.add_argument("--source_model_root", default="source_models")
    parser.add_argument("--semantic_root", default="semantic_packages")
    parser.add_argument(
        "--output_dir",
        default="diagnostics/results/step10_1",
    )

    # Locked pilot values.
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lambda_sem", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the real diagnostic.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    scenarios = parse_scenarios(args.scenarios)
    seeds = parse_seeds(args.seeds)

    if args.k <= 0:
        raise ValueError("k must be positive.")
    if args.gamma <= 0:
        raise ValueError("gamma must be positive.")

    scenario_rows: List[Dict[str, float]] = []
    class_rows: List[Dict[str, float]] = []

    for source_id, target_id in scenarios:
        for seed in seeds:
            print(
                f"\n[Step10.1] dataset={args.dataset} "
                f"scenario={source_id}->{target_id} seed={seed}"
            )

            scenario_row, seed_class_rows = run_seed(
                args=args,
                source_id=source_id,
                target_id=target_id,
                seed=seed,
            )

            scenario_rows.append(scenario_row)
            class_rows.extend(seed_class_rows)

            print(
                "  weighted "
                f"Q_UTA={scenario_row['weighted_q_uta']:+.6f}  "
                f"Q_ACTA={scenario_row['weighted_q_acta']:+.6f}  "
                f"ΔQ={scenario_row['weighted_delta_q']:+.6f}  "
                f"positive={scenario_row['delta_q_positive_fraction']:.3f}  "
                f"edge_shift={scenario_row['weighted_edge_shift']:.4f}"
            )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "seed_summary.csv", scenario_rows)
    write_csv(output_dir / "class_summary.csv", class_rows)

    print_summary(scenario_rows)

    print(f"\nSaved: {output_dir / 'seed_summary.csv'}")
    print(f"Saved: {output_dir / 'class_summary.csv'}")


if __name__ == "__main__":
    main()