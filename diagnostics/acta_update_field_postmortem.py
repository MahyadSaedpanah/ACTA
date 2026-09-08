"""
Step 11 — Source-Semantic Update Subspace post-mortem for ACTA.

Builds a SOURCE-ONLY class-conditioned latent motion subspace from
same-class vs cross-class DTW-aligned source residuals, then tests whether
projecting ACTA's oracle-class alignment gradient into that subspace improves
agreement with the oracle supervised target gradient.

Target labels are used ONLY for the final diagnostic comparison.
"""

from __future__ import annotations

import argparse
import csv
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
from algorithms.components.acta_aligner import resample_temporal_features
from algorithms.utils import fix_randomness
from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator

from semantic_preparation.dtw_utils import (
    prepare_dtw_sequence,
    dtw_path_multivariate,
)
from semantic_preparation.global_teacher import (
    sample_same_pairs,
    sample_cross_pairs,
    class_pair_seed,
)


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


def collect_dataset_tensors(dataset) -> Tuple[torch.Tensor, np.ndarray]:
    xs = []
    ys = []

    for i in range(len(dataset)):
        item = dataset[i]

        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise RuntimeError("Expected dataset items to contain (x, y).")

        x, y = item[0], item[1]

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        if torch.is_tensor(y):
            y = int(y.detach().cpu().item())
        else:
            y = int(y)

        xs.append(x.detach().cpu().float())
        ys.append(y)

    return torch.stack(xs, dim=0), np.asarray(ys, dtype=np.int64)


@torch.no_grad()
def extract_source_latent_grid(
    algorithm: ACTA,
    x_cpu: torch.Tensor,
    semantic_length: int,
    batch_size: int,
) -> torch.Tensor:

    outputs = []

    algorithm.feature_extractor.eval()

    for start in range(0, len(x_cpu), batch_size):
        xb = x_cpu[start:start + batch_size].to(algorithm.device_runtime)
        h = algorithm.feature_extractor.forward_features(xb)
        h = resample_temporal_features(h, semantic_length)
        outputs.append(h.detach().cpu().float())

    return torch.cat(outputs, dim=0)


def build_prepared_sequence_cache(
    x_cpu: torch.Tensor,
    downsample: int,
) -> List[np.ndarray]:

    return [
        prepare_dtw_sequence(
            x_cpu[i],
            downsample=downsample,
        )
        for i in range(len(x_cpu))
    ]


def pair_directional_covariance(
    pair: Tuple[int, int],
    sequence_cache: List[np.ndarray],
    latent_grid: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Equal-weight covariance contribution from ONE source pair.
    Each residual is L2-normalized, so magnitude cannot define the field.
    """

    i, j = int(pair[0]), int(pair[1])

    a = sequence_cache[i]
    b = sequence_cache[j]

    pi, pj, _, _ = dtw_path_multivariate(a, b)

    ha = latent_grid[i]  # [D,G]
    hb = latent_grid[j]  # [D,G]
    g = int(ha.shape[-1])

    if a.shape[0] != g or b.shape[0] != g:
        raise RuntimeError(
            "DTW semantic resolution must match latent semantic grid. "
            f"Got {a.shape[0]},{b.shape[0]} and G={g}."
        )

    pi_t = torch.as_tensor(pi, dtype=torch.long)
    pj_t = torch.as_tensor(pj, dtype=torch.long)

    residual = (
        hb[:, pj_t] - ha[:, pi_t]
    ).transpose(0, 1).double()  # [P,D]

    norms = residual.norm(dim=1, keepdim=True)
    valid = norms.squeeze(1) > eps

    if not bool(valid.any()):
        d = int(ha.shape[0])
        return torch.zeros((d, d), dtype=torch.float64)

    residual = residual[valid]
    residual = residual / residual.norm(dim=1, keepdim=True).clamp_min(eps)

    return (residual.T @ residual) / float(residual.shape[0])


def build_class_subspace(
    class_id: int,
    labels: np.ndarray,
    sequence_cache: List[np.ndarray],
    latent_grid: torch.Tensor,
    dataset_name: str,
    source_id: str,
    n_pairs: int,
    ridge_factor: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Source-only discriminative subspace.

    C+ : same-class DTW-aligned residual directions
    C- : cross-class DTW-aligned residual directions

    Whitening by C+ + C- gives generalized eigenvalues in [0,1].
    Keep directions > 0.5: more same-class than cross-class energy.
    """

    class_id = int(class_id)
    num_classes = int(labels.max()) + 1
    d = int(latent_grid.shape[1])

    class_indices = {
        c: np.where(labels == c)[0].astype(np.int64)
        for c in range(num_classes)
    }

    pos_idx = class_indices[class_id]
    other_idx = np.concatenate(
        [class_indices[c] for c in range(num_classes) if c != class_id],
        axis=0,
    )

    rng = np.random.default_rng(
        class_pair_seed(
            dataset_name,
            source_id,
            class_id,
            "update_field_step11",
        )
    )

    positive_pairs = sample_same_pairs(pos_idx, n_pairs, rng)
    negative_pairs = sample_cross_pairs(pos_idx, other_idx, n_pairs, rng)

    c_pos = torch.zeros((d, d), dtype=torch.float64)
    c_neg = torch.zeros((d, d), dtype=torch.float64)

    for pair in positive_pairs:
        c_pos += pair_directional_covariance(
            pair,
            sequence_cache,
            latent_grid,
        )

    for pair in negative_pairs:
        c_neg += pair_directional_covariance(
            pair,
            sequence_cache,
            latent_grid,
        )

    c_pos /= float(len(positive_pairs))
    c_neg /= float(len(negative_pairs))

    c_pos = 0.5 * (c_pos + c_pos.T)
    c_neg = 0.5 * (c_neg + c_neg.T)

    c_total = c_pos + c_neg

    trace_scale = float(torch.trace(c_total).item()) / float(d)
    ridge = max(float(ridge_factor) * max(trace_scale, 1e-12), 1e-10)

    identity = torch.eye(d, dtype=torch.float64)
    c_reg = c_total + ridge * identity

    eval_total, evec_total = torch.linalg.eigh(c_reg)
    eval_total = eval_total.clamp_min(1e-12)

    inv_sqrt = (
        evec_total
        @ torch.diag(eval_total.rsqrt())
        @ evec_total.T
    )

    whitened_pos = inv_sqrt @ c_pos @ inv_sqrt
    whitened_pos = 0.5 * (whitened_pos + whitened_pos.T)

    eigvals, eigvecs = torch.linalg.eigh(whitened_pos)

    keep = eigvals > 0.5

    if bool(keep.any()):
        mapped = inv_sqrt @ eigvecs[:, keep]
        q, _ = torch.linalg.qr(mapped, mode="reduced")
        basis = q.float()
    else:
        basis = torch.empty((d, 0), dtype=torch.float32)

    stats = {
        "class_id": class_id,
        "rank": int(basis.shape[1]),
        "num_positive_pairs": int(len(positive_pairs)),
        "num_negative_pairs": int(len(negative_pairs)),
        "ridge": float(ridge),
        "max_discriminative_eigenvalue": float(eigvals.max().item()),
        "mean_discriminative_eigenvalue": float(eigvals.mean().item()),
    }

    return basis, stats


def build_update_field(
    algorithm: ACTA,
    source_dataset,
    args: argparse.Namespace,
    source_id: str,
) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:

    x_cpu, labels = collect_dataset_tensors(source_dataset)

    semantic_length = int(
        algorithm.semantic_bank.semantic_temporal_length
    )

    raw_length = int(x_cpu.shape[-1])

    if raw_length % semantic_length != 0:
        raise RuntimeError(
            f"Raw length {raw_length} is not divisible by semantic length "
            f"{semantic_length}."
        )

    downsample = raw_length // semantic_length

    sequence_cache = build_prepared_sequence_cache(
        x_cpu,
        downsample=downsample,
    )

    latent_grid = extract_source_latent_grid(
        algorithm=algorithm,
        x_cpu=x_cpu,
        semantic_length=semantic_length,
        batch_size=args.bs,
    )

    bases = []
    stats = []

    for class_id in range(int(algorithm.configs.num_classes)):

        basis, class_stats = build_class_subspace(
            class_id=class_id,
            labels=labels,
            sequence_cache=sequence_cache,
            latent_grid=latent_grid,
            dataset_name=args.dataset,
            source_id=source_id,
            n_pairs=args.field_pairs,
            ridge_factor=args.field_ridge,
        )

        bases.append(
            basis.to(
                device=algorithm.device_runtime,
                dtype=torch.float32,
            )
        )
        stats.append(class_stats)

    return bases, stats


def matched_class_loss_matrices(
    algorithm: ACTA,
    target_temporal_features: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

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


def project_gradient_by_oracle_class(
    gradient: torch.Tensor,
    labels: torch.Tensor,
    bases: List[torch.Tensor],
) -> torch.Tensor:

    projected = torch.zeros_like(gradient)

    for class_id, basis in enumerate(bases):

        mask = labels == int(class_id)

        if not bool(mask.any()):
            continue

        g = gradient[mask]

        if basis.shape[1] == 0:
            projected[mask] = 0.0
            continue

        coeff = torch.einsum("dr,ndl->nrl", basis, g)
        p_g = torch.einsum("dr,nrl->ndl", basis, coeff)

        projected[mask] = p_g

    return projected


def cosine_per_sample(
    g_a: torch.Tensor,
    g_b: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:

    a = g_a.flatten(1)
    b = g_b.flatten(1)

    a_norm = a.norm(dim=1)
    b_norm = b.norm(dim=1)

    dot = (a * b).sum(dim=1)
    denom = (a_norm * b_norm).clamp_min(eps)

    cosine = dot / denom

    zero = (a_norm <= eps) | (b_norm <= eps)
    cosine = torch.where(zero, torch.zeros_like(cosine), cosine)

    return cosine


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

    algorithm.eval()
    algorithm.feature_extractor.eval()
    algorithm.classifier.eval()
    algorithm.ema_feature_extractor.eval()
    algorithm.ema_classifier.eval()
    algorithm.semantic_bank.eval()

    bases, field_stats = build_update_field(
        algorithm=algorithm,
        source_dataset=src_train_dl.dataset,
        args=args,
        source_id=str(source_id),
    )

    algorithm.attach_source_reference_pool(
        source_train_dataset=src_train_dl.dataset,
        reference_seed=int(seed),
    )

    target_loader = make_full_loader(
        trg_train_dl.dataset,
        batch_size=args.bs,
    )

    cos_uta_all = []
    cos_acta_all = []
    cos_proj_uta_all = []
    cos_proj_acta_all = []

    processed_batches = 0

    for batch_index, (trg_x, trg_y) in enumerate(target_loader):

        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        trg_x = trg_x.float().to(device)
        trg_y = trg_y.long().to(device)

        with torch.no_grad():
            h0 = algorithm.feature_extractor.forward_features(trg_x)

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
        logits = algorithm.classifier(pooled)

        target_ce = F.cross_entropy(
            logits,
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

        g_proj_uta = project_gradient_by_oracle_class(
            g_uta,
            trg_y,
            bases,
        )

        g_proj_acta = project_gradient_by_oracle_class(
            g_acta,
            trg_y,
            bases,
        )

        cos_uta_all.append(
            cosine_per_sample(g_uta, g_target).detach().cpu()
        )
        cos_acta_all.append(
            cosine_per_sample(g_acta, g_target).detach().cpu()
        )
        cos_proj_uta_all.append(
            cosine_per_sample(g_proj_uta, g_target).detach().cpu()
        )
        cos_proj_acta_all.append(
            cosine_per_sample(g_proj_acta, g_target).detach().cpu()
        )

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("No target batches were processed.")

    cos_uta = torch.cat(cos_uta_all)
    cos_acta = torch.cat(cos_acta_all)
    cos_proj_uta = torch.cat(cos_proj_uta_all)
    cos_proj_acta = torch.cat(cos_proj_acta_all)

    delta_acta_vs_uta = cos_acta - cos_uta
    delta_proj_acta_vs_acta = cos_proj_acta - cos_acta
    delta_proj_acta_vs_uta = cos_proj_acta - cos_uta
    delta_proj_acta_vs_proj_uta = cos_proj_acta - cos_proj_uta

    row = {
        "scenario": f"{source_id}->{target_id}",
        "source": str(source_id),
        "target": str(target_id),
        "seed": int(seed),
        "n_target": int(cos_uta.numel()),

        "cos_uta": float(cos_uta.mean().item()),
        "cos_acta": float(cos_acta.mean().item()),
        "cos_projected_uta": float(cos_proj_uta.mean().item()),
        "cos_projected_acta": float(cos_proj_acta.mean().item()),

        "delta_acta_vs_uta": float(delta_acta_vs_uta.mean().item()),
        "delta_projected_acta_vs_acta": float(
            delta_proj_acta_vs_acta.mean().item()
        ),
        "delta_projected_acta_vs_uta": float(
            delta_proj_acta_vs_uta.mean().item()
        ),
        "delta_projected_acta_vs_projected_uta": float(
            delta_proj_acta_vs_proj_uta.mean().item()
        ),

        "fraction_projected_acta_better_than_acta": float(
            (delta_proj_acta_vs_acta > 0).float().mean().item()
        ),
        "fraction_projected_acta_better_than_uta": float(
            (delta_proj_acta_vs_uta > 0).float().mean().item()
        ),

        "mean_field_rank": float(
            np.mean([s["rank"] for s in field_stats])
        ),
    }

    print(
        f"\n[Step11] dataset={args.dataset} "
        f"scenario={source_id}->{target_id} seed={seed}"
    )
    print(
        "  "
        f"cos UTA={row['cos_uta']:+.6f}  "
        f"ACTA={row['cos_acta']:+.6f}  "
        f"P-UTA={row['cos_projected_uta']:+.6f}  "
        f"P-ACTA={row['cos_projected_acta']:+.6f}"
    )
    print(
        "  "
        f"P-ACTA - ACTA={row['delta_projected_acta_vs_acta']:+.6f}  "
        f"P-ACTA - UTA={row['delta_projected_acta_vs_uta']:+.6f}  "
        f"P-ACTA - P-UTA="
        f"{row['delta_projected_acta_vs_projected_uta']:+.6f}"
    )
    print(
        "  "
        f"sample better vs ACTA="
        f"{row['fraction_projected_acta_better_than_acta']:.3f}  "
        f"vs UTA={row['fraction_projected_acta_better_than_uta']:.3f}  "
        f"mean field rank={row['mean_field_rank']:.2f}"
    )

    for stats in field_stats:
        stats["scenario"] = f"{source_id}->{target_id}"
        stats["source"] = str(source_id)
        stats["target"] = str(target_id)
        stats["seed"] = int(seed)

    return row, field_stats


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to save.")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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

    parser.add_argument("--field_pairs", type=int, default=80)
    parser.add_argument("--field_ridge", type=float, default=1e-4)

    parser.add_argument("--max_batches", type=int, default=None)

    parser.add_argument(
        "--output_dir",
        default="diagnostics/results/step11",
    )

    args = parser.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    seeds = parse_seeds(args.seeds)

    rows = []
    all_field_stats = []

    for source_id, target_id in scenarios:
        for seed in seeds:

            row, field_stats = run_seed(
                args=args,
                source_id=source_id,
                target_id=target_id,
                seed=seed,
            )

            rows.append(row)
            all_field_stats.extend(field_stats)

    print("\n" + "=" * 98)
    print("STEP 11 SUMMARY — source-semantic projected ACTA gradient")
    print("=" * 98)

    for source_id, target_id in scenarios:

        scenario = f"{source_id}->{target_id}"
        subset = [r for r in rows if r["scenario"] == scenario]

        d_vs_acta = [
            float(r["delta_projected_acta_vs_acta"])
            for r in subset
        ]
        d_vs_uta = [
            float(r["delta_projected_acta_vs_uta"])
            for r in subset
        ]
        d_synergy = [
            float(r["delta_projected_acta_vs_projected_uta"])
            for r in subset
        ]

        print(
            f"{scenario:>10} | "
            f"P-ACTA−ACTA={mean(d_vs_acta):+.6f} | "
            f"P-ACTA−UTA={mean(d_vs_uta):+.6f} | "
            f"P-ACTA−P-UTA={mean(d_synergy):+.6f} | "
            f"seed-positive-vs-UTA="
            f"{sum(x > 0 for x in d_vs_uta)}/{len(d_vs_uta)}"
        )

    overall_vs_acta = [
        float(r["delta_projected_acta_vs_acta"])
        for r in rows
    ]
    overall_vs_uta = [
        float(r["delta_projected_acta_vs_uta"])
        for r in rows
    ]
    overall_synergy = [
        float(r["delta_projected_acta_vs_projected_uta"])
        for r in rows
    ]

    print("-" * 98)
    print(
        f"{'OVERALL':>10} | "
        f"P-ACTA−ACTA={mean(overall_vs_acta):+.6f} | "
        f"P-ACTA−UTA={mean(overall_vs_uta):+.6f} | "
        f"P-ACTA−P-UTA={mean(overall_synergy):+.6f} | "
        f"seed-positive-vs-UTA="
        f"{sum(x > 0 for x in overall_vs_uta)}/{len(overall_vs_uta)}"
    )
    print("=" * 98)

    output_dir = Path(args.output_dir)

    seed_path = output_dir / "seed_summary.csv"
    field_path = output_dir / "field_summary.csv"

    write_csv(seed_path, rows)
    write_csv(field_path, all_field_stats)

    print(f"\nSaved: {seed_path}")
    print(f"Saved: {field_path}")


if __name__ == "__main__":
    main()