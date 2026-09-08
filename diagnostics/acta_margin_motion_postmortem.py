"""
Step 12 — Source-Task Margin-Admissible Motion Diagnostic for ACTA.

Question
--------
Steps 10.1–10.3 showed:

    ACTA makes temporal paths more semantically admissible,
    but the resulting alignment gradient is not reliably more target-useful.

Step 11 showed that a same-class/cross-class covariance subspace is NOT a
valid definition of safe representation motion.

This diagnostic tests a stricter, task-defined notion:

    A representation update is admissible only if, to first order,
    it does not decrease the SOURCE classifier margin of the selected class
    against any competing class.

For class c and competitor k:

    m_{c,k}(H) = z_c(H) - z_k(H)

A descent/update direction d is margin-admissible when:

    <∇_H m_{c,k}, d> >= 0    for every k != c.

Given the raw ACTA descent direction d0 = -g_ACTA, we solve the exact
Euclidean projection:

    d_safe = argmin_d 0.5 ||d-d0||^2
             s.t. A_c d >= 0.

The same projection is also applied to UTA as a mechanistic control.

IMPORTANT
---------
- The margin geometry comes only from the frozen SOURCE checkpoint.
- Target labels are used ONLY to choose the oracle class in this diagnostic
  and to construct the oracle target gradient used for evaluation.
- No target label is used in ACTA training or semantic preparation.
- No hyperparameter is tuned here: the constraint is strict first-order
  non-decrease of every source-task class margin.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from algorithms.ACTA import ACTA
from algorithms.utils import fix_randomness
from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator


# -----------------------------------------------------------------------------
# CLI / setup
# -----------------------------------------------------------------------------


def parse_scenarios(text: str) -> List[Tuple[str, str]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid scenario '{item}'. Use source:target.")
        out.append((parts[0].strip(), parts[1].strip()))
    if not out:
        raise ValueError("At least one scenario is required.")
    return out


def parse_seeds(text: str) -> List[int]:
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def build_algorithm_args(args: argparse.Namespace) -> SimpleNamespace:
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


def make_full_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


# -----------------------------------------------------------------------------
# Matched oracle-class UTA / ACTA losses
# -----------------------------------------------------------------------------


def matched_class_loss_matrices(
    algorithm: ACTA,
    target_temporal_features: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        UTA  [B,C]
        ACTA [B,C]

    For each class, UTA and ACTA reuse exactly the same source references.
    """

    batch_size = int(target_temporal_features.shape[0])
    num_classes = int(algorithm.configs.num_classes)

    uta_by_class = []
    acta_by_class = []

    for class_id in range(num_classes):

        reference_x, reference_y, _ = algorithm.sample_class_references(
            class_id=class_id,
            k=k,
            return_indices=True,
        )

        if not torch.all(reference_y == class_id):
            raise RuntimeError("Wrong-class source reference.")

        with torch.no_grad():
            reference_features = algorithm.feature_extractor.forward_features(
                reference_x
            )

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

        uta_details = algorithm.alignment_core(
            source_features=source_pairs,
            target_features=target_pairs,
            class_id=class_id,
            use_semantics=False,
            semantic_class_id=class_id,
            reliability_class_id=class_id,
            return_details=True,
        )

        acta_details = algorithm.alignment_core(
            source_features=source_pairs,
            target_features=target_pairs,
            class_id=class_id,
            use_semantics=True,
            semantic_class_id=class_id,
            reliability_class_id=class_id,
            return_details=True,
        )

        uta_c = uta_details["pair_loss"].reshape(batch_size, k).mean(dim=1)
        acta_c = acta_details["pair_loss"].reshape(batch_size, k).mean(dim=1)

        uta_by_class.append(uta_c)
        acta_by_class.append(acta_c)

    return (
        torch.stack(uta_by_class, dim=1),
        torch.stack(acta_by_class, dim=1),
    )


# -----------------------------------------------------------------------------
# SOURCE-task margin geometry
# -----------------------------------------------------------------------------


def build_margin_constraint_bank(
    algorithm: ACTA,
    feature_shape: Tuple[int, int],
) -> Dict[int, torch.Tensor]:
    """
    Build exact source-classifier margin gradients in temporal feature space.

    Returns:
        bank[c] : [C-1, D*L]

    Each row is:
        ∇_H (z_c - z_k)

    The mapping H -> adaptive_pool -> classifier is deterministic and frozen,
    so these gradients are computed once per source checkpoint.
    """

    d, l = int(feature_shape[0]), int(feature_shape[1])
    device = algorithm.device_runtime
    num_classes = int(algorithm.configs.num_classes)

    bank: Dict[int, torch.Tensor] = {}

    # One symbolic/dummy temporal representation is sufficient because
    # the current benchmark path from H to logits is adaptive pooling
    # followed by a linear classifier. Autograd keeps this implementation
    # exact without hard-coding that algebra.
    h = torch.zeros(
        1,
        d,
        l,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    pooled = algorithm.feature_extractor.adaptive_pool(h)
    pooled = pooled.reshape(pooled.shape[0], -1)
    logits = algorithm.classifier(pooled)[0]

    for class_id in range(num_classes):

        rows = []

        for competitor in range(num_classes):

            if competitor == class_id:
                continue

            margin = logits[class_id] - logits[competitor]

            grad = torch.autograd.grad(
                margin,
                h,
                retain_graph=True,
                create_graph=False,
            )[0]

            rows.append(
                grad.detach().reshape(-1)
            )

        a = torch.stack(rows, dim=0)

        # Row normalization changes neither the homogeneous half-space
        # nor the exact projection solution, but improves numerical stability.
        a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)

        bank[class_id] = a.detach()

    return bank


# -----------------------------------------------------------------------------
# Exact projection onto homogeneous half-space intersection
# -----------------------------------------------------------------------------


def _project_one_exact(
    direction: np.ndarray,
    constraints: np.ndarray,
    feasibility_tol: float = 1e-8,
    lambda_tol: float = 1e-8,
) -> Tuple[np.ndarray, bool]:
    """
    Exact active-set enumeration for:

        min_d 0.5 ||d-d0||^2
        s.t. A d >= 0

    K = C-1 is small (5 for UCIHAR), so enumerating all active subsets
    is cheap and avoids an optimization dependency.
    """

    d0 = np.asarray(direction, dtype=np.float64)
    a = np.asarray(constraints, dtype=np.float64)

    k = int(a.shape[0])

    raw_values = a @ d0
    raw_feasible = bool(np.all(raw_values >= -feasibility_tol))

    if raw_feasible:
        return d0.copy(), False

    gram = a @ a.T
    best = None
    best_obj = np.inf

    indices = list(range(k))

    # Empty active set is infeasible here because raw direction violated
    # at least one constraint, so start at size 1.
    for size in range(1, k + 1):

        for subset in itertools.combinations(indices, size):

            s = np.asarray(subset, dtype=np.int64)

            g_ss = gram[np.ix_(s, s)]
            rhs = -raw_values[s]

            try:
                lam = np.linalg.solve(g_ss, rhs)
            except np.linalg.LinAlgError:
                lam = np.linalg.lstsq(g_ss, rhs, rcond=None)[0]

            # KKT multiplier condition.
            if np.any(lam < -lambda_tol):
                continue

            candidate = d0 + a[s].T @ lam

            values = a @ candidate

            if np.any(values < -feasibility_tol):
                continue

            diff = candidate - d0
            obj = 0.5 * float(diff @ diff)

            if obj < best_obj:
                best_obj = obj
                best = candidate

    if best is None:
        # The zero vector is always feasible for a homogeneous cone.
        # Reaching this branch would indicate numerical degeneracy.
        zero = np.zeros_like(d0)

        if np.any(a @ zero < -feasibility_tol):
            raise RuntimeError("Homogeneous margin cone unexpectedly infeasible.")

        return zero, True

    return best, True


def project_batch_by_oracle_class(
    descent_direction: torch.Tensor,
    labels: torch.Tensor,
    constraint_bank: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project each sample's descent/update direction onto its oracle-class
    SOURCE margin-admissible cone.

    Returns:
        projected [B,D,L]
        changed   [B] bool
        norm_retention [B]
    """

    device = descent_direction.device
    dtype = descent_direction.dtype

    flat = descent_direction.detach().reshape(
        descent_direction.shape[0],
        -1,
    )

    projected_rows = []
    changed_rows = []
    retention_rows = []

    for b in range(flat.shape[0]):

        class_id = int(labels[b].item())

        d0 = (
            flat[b]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        a = (
            constraint_bank[class_id]
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        projected, changed = _project_one_exact(
            direction=d0,
            constraints=a,
        )

        d0_norm = float(np.linalg.norm(d0))
        projected_norm = float(np.linalg.norm(projected))

        retention = (
            projected_norm / max(d0_norm, 1e-12)
        )

        projected_rows.append(
            torch.from_numpy(projected)
        )
        changed_rows.append(changed)
        retention_rows.append(retention)

    projected = (
        torch.stack(projected_rows, dim=0)
        .to(device=device, dtype=dtype)
        .reshape_as(descent_direction)
    )

    changed = torch.tensor(
        changed_rows,
        device=device,
        dtype=torch.bool,
    )

    retention = torch.tensor(
        retention_rows,
        device=device,
        dtype=dtype,
    )

    return projected, changed, retention


def min_margin_derivative(
    direction: torch.Tensor,
    labels: torch.Tensor,
    constraint_bank: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    Per sample minimum normalized source-margin directional derivative.
    Feasible directions should have min >= ~0.
    """

    flat = direction.reshape(direction.shape[0], -1)

    values = []

    for b in range(flat.shape[0]):

        class_id = int(labels[b].item())
        a = constraint_bank[class_id].to(
            device=flat.device,
            dtype=flat.dtype,
        )

        values.append(
            (a @ flat[b]).min()
        )

    return torch.stack(values)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def cosine_per_sample(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:

    a = a.flatten(1)
    b = b.flatten(1)

    an = a.norm(dim=1)
    bn = b.norm(dim=1)

    cosine = (
        (a * b).sum(dim=1)
        /
        (an * bn).clamp_min(eps)
    )

    zero = (an <= eps) | (bn <= eps)

    return torch.where(
        zero,
        torch.zeros_like(cosine),
        cosine,
    )


# -----------------------------------------------------------------------------
# One scenario/seed
# -----------------------------------------------------------------------------


def run_seed(
    args: argparse.Namespace,
    source_id: str,
    target_id: str,
    seed: int,
) -> Dict[str, float]:

    fix_randomness(seed)

    device = torch.device(args.device)
    configs = get_dataset_class(args.dataset)()
    model_args = build_algorithm_args(args)

    data_path = str(Path(args.data_root) / args.dataset)

    src_train_dl, _ = data_generator(
        data_path,
        str(source_id),
        model_args,
    )

    trg_train_dl, _ = data_generator(
        data_path,
        str(target_id),
        model_args,
    )

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

    algorithm.eval()
    algorithm.feature_extractor.eval()
    algorithm.classifier.eval()
    algorithm.ema_feature_extractor.eval()
    algorithm.ema_classifier.eval()
    algorithm.semantic_bank.eval()

    target_loader = make_full_loader(
        trg_train_dl.dataset,
        batch_size=args.bs,
    )

    constraint_bank = None

    cos_uta_all = []
    cos_acta_all = []
    cos_safe_uta_all = []
    cos_safe_acta_all = []

    changed_uta_all = []
    changed_acta_all = []

    retention_uta_all = []
    retention_acta_all = []

    min_raw_acta_all = []
    min_safe_acta_all = []

    processed_batches = 0

    for batch_index, (trg_x, trg_y) in enumerate(target_loader):

        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        trg_x = trg_x.float().to(device)
        trg_y = trg_y.long().to(device)

        with torch.no_grad():
            h0 = algorithm.feature_extractor.forward_features(trg_x)

        if constraint_bank is None:
            constraint_bank = build_margin_constraint_bank(
                algorithm=algorithm,
                feature_shape=(
                    int(h0.shape[1]),
                    int(h0.shape[2]),
                ),
            )

        h = h0.detach().requires_grad_(True)

        uta_matrix, acta_matrix = matched_class_loss_matrices(
            algorithm=algorithm,
            target_temporal_features=h,
            k=args.k,
        )

        gather_index = trg_y[:, None]

        uta_oracle = uta_matrix.gather(1, gather_index).squeeze(1)
        acta_oracle = acta_matrix.gather(1, gather_index).squeeze(1)

        pooled = algorithm.feature_extractor.adaptive_pool(h)
        pooled = pooled.reshape(pooled.shape[0], -1)
        target_logits = algorithm.classifier(pooled)

        target_ce = F.cross_entropy(
            target_logits,
            trg_y,
            reduction="none",
        )

        g_target = torch.autograd.grad(
            target_ce.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_uta = torch.autograd.grad(
            uta_oracle.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_acta = torch.autograd.grad(
            acta_oracle.sum(),
            h,
            retain_graph=False,
            create_graph=False,
        )[0]

        # Descent/update directions.
        d_target = -g_target
        d_uta = -g_uta
        d_acta = -g_acta

        d_safe_uta, changed_uta, retention_uta = (
            project_batch_by_oracle_class(
                descent_direction=d_uta,
                labels=trg_y,
                constraint_bank=constraint_bank,
            )
        )

        d_safe_acta, changed_acta, retention_acta = (
            project_batch_by_oracle_class(
                descent_direction=d_acta,
                labels=trg_y,
                constraint_bank=constraint_bank,
            )
        )

        cos_uta_all.append(
            cosine_per_sample(d_uta, d_target).detach().cpu()
        )
        cos_acta_all.append(
            cosine_per_sample(d_acta, d_target).detach().cpu()
        )
        cos_safe_uta_all.append(
            cosine_per_sample(d_safe_uta, d_target).detach().cpu()
        )
        cos_safe_acta_all.append(
            cosine_per_sample(d_safe_acta, d_target).detach().cpu()
        )

        changed_uta_all.append(changed_uta.detach().cpu())
        changed_acta_all.append(changed_acta.detach().cpu())

        retention_uta_all.append(retention_uta.detach().cpu())
        retention_acta_all.append(retention_acta.detach().cpu())

        min_raw_acta_all.append(
            min_margin_derivative(
                d_acta,
                trg_y,
                constraint_bank,
            ).detach().cpu()
        )

        min_safe_acta_all.append(
            min_margin_derivative(
                d_safe_acta,
                trg_y,
                constraint_bank,
            ).detach().cpu()
        )

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("No target batches were processed.")

    cos_uta = torch.cat(cos_uta_all)
    cos_acta = torch.cat(cos_acta_all)
    cos_safe_uta = torch.cat(cos_safe_uta_all)
    cos_safe_acta = torch.cat(cos_safe_acta_all)

    changed_uta = torch.cat(changed_uta_all)
    changed_acta = torch.cat(changed_acta_all)

    retention_uta = torch.cat(retention_uta_all)
    retention_acta = torch.cat(retention_acta_all)

    min_raw_acta = torch.cat(min_raw_acta_all)
    min_safe_acta = torch.cat(min_safe_acta_all)

    delta_safe_vs_acta = cos_safe_acta - cos_acta
    delta_safe_vs_uta = cos_safe_acta - cos_uta
    delta_safe_acta_vs_safe_uta = cos_safe_acta - cos_safe_uta

    row = {
        "scenario": f"{source_id}->{target_id}",
        "source": str(source_id),
        "target": str(target_id),
        "seed": int(seed),
        "n_target": int(cos_uta.numel()),

        "cos_uta": float(cos_uta.mean().item()),
        "cos_acta": float(cos_acta.mean().item()),
        "cos_margin_uta": float(cos_safe_uta.mean().item()),
        "cos_margin_acta": float(cos_safe_acta.mean().item()),

        "delta_margin_acta_vs_acta": float(
            delta_safe_vs_acta.mean().item()
        ),
        "delta_margin_acta_vs_uta": float(
            delta_safe_vs_uta.mean().item()
        ),
        "delta_margin_acta_vs_margin_uta": float(
            delta_safe_acta_vs_safe_uta.mean().item()
        ),

        "fraction_margin_acta_better_than_acta": float(
            (delta_safe_vs_acta > 0).float().mean().item()
        ),
        "fraction_margin_acta_better_than_uta": float(
            (delta_safe_vs_uta > 0).float().mean().item()
        ),

        "fraction_raw_acta_constraint_violating": float(
            changed_acta.float().mean().item()
        ),
        "fraction_raw_uta_constraint_violating": float(
            changed_uta.float().mean().item()
        ),

        "mean_acta_norm_retention": float(
            retention_acta.mean().item()
        ),
        "mean_uta_norm_retention": float(
            retention_uta.mean().item()
        ),

        "mean_min_raw_acta_margin_derivative": float(
            min_raw_acta.mean().item()
        ),
        "min_safe_acta_margin_derivative": float(
            min_safe_acta.min().item()
        ),
    }

    print(
        f"\n[Step12] dataset={args.dataset} "
        f"scenario={source_id}->{target_id} seed={seed}"
    )
    print(
        "  "
        f"cos UTA={row['cos_uta']:+.6f}  "
        f"ACTA={row['cos_acta']:+.6f}  "
        f"M-UTA={row['cos_margin_uta']:+.6f}  "
        f"M-ACTA={row['cos_margin_acta']:+.6f}"
    )
    print(
        "  "
        f"M-ACTA−ACTA={row['delta_margin_acta_vs_acta']:+.6f}  "
        f"M-ACTA−UTA={row['delta_margin_acta_vs_uta']:+.6f}  "
        f"M-ACTA−M-UTA={row['delta_margin_acta_vs_margin_uta']:+.6f}"
    )
    print(
        "  "
        f"raw ACTA violating="
        f"{row['fraction_raw_acta_constraint_violating']:.3f}  "
        f"norm retention={row['mean_acta_norm_retention']:.3f}  "
        f"sample better vs UTA="
        f"{row['fraction_margin_acta_better_than_uta']:.3f}"
    )

    # Exact projection sanity check.
    if row["min_safe_acta_margin_derivative"] < -1e-5:
        raise RuntimeError(
            "Projected ACTA direction violates source-margin constraints: "
            f"min={row['min_safe_acta_margin_derivative']}"
        )

    return row


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to save.")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs))


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="UCIHAR")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--source_model_root", default="source_models")
    parser.add_argument("--semantic_root", default="semantic_packages")

    parser.add_argument(
        "--scenarios",
        default="9:18,12:16,23:13",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_sem", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--ema", type=float, default=0.99)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--max_batches", type=int, default=None)

    parser.add_argument(
        "--output_dir",
        default="diagnostics/results/step12",
    )

    args = parser.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    seeds = parse_seeds(args.seeds)

    rows = []

    for source_id, target_id in scenarios:
        for seed in seeds:
            rows.append(
                run_seed(
                    args=args,
                    source_id=source_id,
                    target_id=target_id,
                    seed=seed,
                )
            )

    print("\n" + "=" * 102)
    print("STEP 12 SUMMARY — source-task margin-admissible ACTA motion")
    print("=" * 102)

    for source_id, target_id in scenarios:

        scenario = f"{source_id}->{target_id}"
        subset = [r for r in rows if r["scenario"] == scenario]

        d_vs_acta = [
            float(r["delta_margin_acta_vs_acta"])
            for r in subset
        ]
        d_vs_uta = [
            float(r["delta_margin_acta_vs_uta"])
            for r in subset
        ]
        d_vs_safe_uta = [
            float(r["delta_margin_acta_vs_margin_uta"])
            for r in subset
        ]

        print(
            f"{scenario:>10} | "
            f"M-ACTA−ACTA={mean(d_vs_acta):+.6f} | "
            f"M-ACTA−UTA={mean(d_vs_uta):+.6f} | "
            f"M-ACTA−M-UTA={mean(d_vs_safe_uta):+.6f} | "
            f"seed-positive-vs-UTA="
            f"{sum(x > 0 for x in d_vs_uta)}/{len(d_vs_uta)}"
        )

    overall_vs_acta = [
        float(r["delta_margin_acta_vs_acta"])
        for r in rows
    ]
    overall_vs_uta = [
        float(r["delta_margin_acta_vs_uta"])
        for r in rows
    ]
    overall_vs_safe_uta = [
        float(r["delta_margin_acta_vs_margin_uta"])
        for r in rows
    ]

    print("-" * 102)
    print(
        f"{'OVERALL':>10} | "
        f"M-ACTA−ACTA={mean(overall_vs_acta):+.6f} | "
        f"M-ACTA−UTA={mean(overall_vs_uta):+.6f} | "
        f"M-ACTA−M-UTA={mean(overall_vs_safe_uta):+.6f} | "
        f"seed-positive-vs-UTA="
        f"{sum(x > 0 for x in overall_vs_uta)}/{len(overall_vs_uta)}"
    )
    print("=" * 102)

    output_path = Path(args.output_dir) / "seed_summary.csv"
    write_csv(output_path, rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
