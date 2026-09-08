"""
Step 10.2 — ACTA semantic-path -> representation-gradient post-mortem.

Question
--------
At the SAME source-initialized representation theta_S, with the SAME
target samples, SAME EMA class probabilities, and SAME source references,
does ACTA's semantic path intervention produce a representation gradient
that is more consistent with the oracle supervised target gradient than UTA?

Target labels are used ONLY for this post-hoc diagnostic.
They are never used to train ACTA, choose hyperparameters, or alter semantics.

Primary metrics
---------------
For each target sample b, let

    g_T    = d L_target_supervised / d H_t
    g_UTA  = d L_align_UTA        / d H_t
    g_ACTA = d L_align_ACTA       / d H_t

where H_t is the temporal feature map at theta_S.

Directional agreement:

    cos(g_align, g_T)

First-order target utility of an alignment descent step:

    u = <g_align, g_T> / (||g_T||^2 + eps)

because for H <- H - eta*g_align,

    Delta L_T ~= -eta <g_T, g_align>.

We report ACTA - UTA deltas for both metrics.

A supplementary gradient-shift metric checks whether semantics changes the
actual representation gradient:

    ||g_ACTA - g_UTA|| / (||g_UTA|| + eps)
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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from algorithms.ACTA import ACTA
from algorithms.utils import fix_randomness
from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator


# -----------------------------------------------------------------------------
# Parsing / setup
# -----------------------------------------------------------------------------


def parse_scenarios(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid scenario '{item}'. Use source:target, e.g. 9:18."
            )
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


def make_full_train_loader(train_dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


# -----------------------------------------------------------------------------
# Matched UTA / ACTA alignment losses
# -----------------------------------------------------------------------------


def matched_class_losses(
    algorithm: ACTA,
    target_temporal_features: torch.Tensor,
    class_id: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return per-target UTA and ACTA class losses [B], using EXACTLY the same
    sampled same-class source references for both methods.
    """

    class_id = int(class_id)
    batch_size = int(target_temporal_features.shape[0])

    reference_x, reference_y, _reference_indices = (
        algorithm.sample_class_references(
            class_id=class_id,
            k=k,
            return_indices=True,
        )
    )

    if not torch.all(reference_y == class_id):
        raise RuntimeError("Wrong-class source reference.")

    with torch.no_grad():
        reference_features = algorithm.feature_extractor.forward_features(
            reference_x
        )

    _, feature_dim, source_length = reference_features.shape
    target_length = int(target_temporal_features.shape[-1])

    if int(target_temporal_features.shape[1]) != int(feature_dim):
        raise RuntimeError("Reference/target feature dimension mismatch.")

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

    # Same source_pairs, target_pairs, feature encoder state, and references.
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

    uta = uta_details["pair_loss"].reshape(batch_size, k).mean(dim=1)
    acta = acta_details["pair_loss"].reshape(batch_size, k).mean(dim=1)

    return uta, acta


def matched_alignment_losses(
    algorithm: ACTA,
    target_temporal_features: torch.Tensor,
    target_probabilities: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft class-conditioned per-target losses [B] for matched UTA / ACTA.
    """

    num_classes = int(algorithm.configs.num_classes)

    uta_by_class: List[torch.Tensor] = []
    acta_by_class: List[torch.Tensor] = []

    for class_id in range(num_classes):
        uta_c, acta_c = matched_class_losses(
            algorithm=algorithm,
            target_temporal_features=target_temporal_features,
            class_id=class_id,
            k=k,
        )
        uta_by_class.append(uta_c)
        acta_by_class.append(acta_c)

    uta_matrix = torch.stack(uta_by_class, dim=1)    # [B,C]
    acta_matrix = torch.stack(acta_by_class, dim=1)  # [B,C]

    probabilities = target_probabilities.detach()

    uta = (probabilities * uta_matrix).sum(dim=1)
    acta = (probabilities * acta_matrix).sum(dim=1)

    return uta, acta


# -----------------------------------------------------------------------------
# Gradient metrics
# -----------------------------------------------------------------------------


def per_sample_gradient_metrics(
    g_align: torch.Tensor,
    g_target: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        cosine agreement [B]
        normalized first-order utility [B]
    """

    a = g_align.flatten(1)
    t = g_target.flatten(1)

    dot = (a * t).sum(dim=1)

    a_norm = a.norm(dim=1)
    t_norm = t.norm(dim=1)

    cosine = dot / (a_norm * t_norm).clamp_min(eps)
    utility = dot / t.pow(2).sum(dim=1).clamp_min(eps)

    return cosine, utility


def relative_gradient_shift(
    g_uta: torch.Tensor,
    g_acta: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    u = g_uta.flatten(1)
    a = g_acta.flatten(1)
    return (a - u).norm(dim=1) / u.norm(dim=1).clamp_min(eps)


# -----------------------------------------------------------------------------
# One seed
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

    # Diagnose common theta_S, not an adapted checkpoint.
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

    cos_uta_all: List[torch.Tensor] = []
    cos_acta_all: List[torch.Tensor] = []
    delta_cos_all: List[torch.Tensor] = []

    util_uta_all: List[torch.Tensor] = []
    util_acta_all: List[torch.Tensor] = []
    delta_util_all: List[torch.Tensor] = []

    grad_shift_all: List[torch.Tensor] = []

    processed_batches = 0

    for batch_index, (trg_x, trg_y) in enumerate(target_loader):

        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        trg_x = trg_x.float().to(device)
        trg_y = trg_y.long().to(device)

        with torch.no_grad():
            target_probabilities = algorithm.ema_probabilities(trg_x)
            h0 = algorithm.feature_extractor.forward_features(trg_x)

        # Common target representation variable for all three gradients.
        h = h0.detach().requires_grad_(True)

        uta_per_target, acta_per_target = matched_alignment_losses(
            algorithm=algorithm,
            target_temporal_features=h,
            target_probabilities=target_probabilities,
            k=args.k,
        )

        # Oracle supervised target loss from the exact benchmark classifier
        # path beginning at the temporal feature map H_t.
        pooled = algorithm.feature_extractor.adaptive_pool(h)
        pooled = pooled.reshape(pooled.shape[0], -1)
        target_logits = algorithm.classifier(pooled)

        target_ce_per_target = F.cross_entropy(
            target_logits,
            trg_y,
            reduction="none",
        )

        # Losses are sample-separable, so gradient of the sum gives one
        # independent per-sample gradient in h[b].
        g_target = torch.autograd.grad(
            target_ce_per_target.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_uta = torch.autograd.grad(
            uta_per_target.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_acta = torch.autograd.grad(
            acta_per_target.sum(),
            h,
            retain_graph=False,
            create_graph=False,
        )[0]

        cos_uta, util_uta = per_sample_gradient_metrics(
            g_align=g_uta,
            g_target=g_target,
        )

        cos_acta, util_acta = per_sample_gradient_metrics(
            g_align=g_acta,
            g_target=g_target,
        )

        delta_cos = cos_acta - cos_uta
        delta_util = util_acta - util_uta
        grad_shift = relative_gradient_shift(g_uta, g_acta)

        cos_uta_all.append(cos_uta.detach().cpu())
        cos_acta_all.append(cos_acta.detach().cpu())
        delta_cos_all.append(delta_cos.detach().cpu())

        util_uta_all.append(util_uta.detach().cpu())
        util_acta_all.append(util_acta.detach().cpu())
        delta_util_all.append(delta_util.detach().cpu())

        grad_shift_all.append(grad_shift.detach().cpu())

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("No target batches were processed.")

    cos_uta = torch.cat(cos_uta_all)
    cos_acta = torch.cat(cos_acta_all)
    delta_cos = torch.cat(delta_cos_all)

    util_uta = torch.cat(util_uta_all)
    util_acta = torch.cat(util_acta_all)
    delta_util = torch.cat(delta_util_all)

    grad_shift = torch.cat(grad_shift_all)

    row: Dict[str, float] = {
        "scenario": f"{source_id}->{target_id}",
        "source": str(source_id),
        "target": str(target_id),
        "seed": int(seed),
        "n_target": int(delta_cos.numel()),

        "cos_uta": float(cos_uta.mean().item()),
        "cos_acta": float(cos_acta.mean().item()),
        "delta_cos": float(delta_cos.mean().item()),
        "fraction_acta_cos_better": float((delta_cos > 0).float().mean().item()),
        "fraction_uta_cos_positive": float((cos_uta > 0).float().mean().item()),
        "fraction_acta_cos_positive": float((cos_acta > 0).float().mean().item()),

        "utility_uta": float(util_uta.mean().item()),
        "utility_acta": float(util_acta.mean().item()),
        "delta_utility": float(delta_util.mean().item()),
        "fraction_acta_utility_better": float(
            (delta_util > 0).float().mean().item()
        ),

        "gradient_shift": float(grad_shift.mean().item()),
    }

    print(
        f"\n[Step10.2] dataset={args.dataset} "
        f"scenario={source_id}->{target_id} seed={seed}"
    )
    print(
        "  "
        f"cos UTA={row['cos_uta']:+.6f}  "
        f"ACTA={row['cos_acta']:+.6f}  "
        f"Δcos={row['delta_cos']:+.6f}  "
        f"ACTA-better={row['fraction_acta_cos_better']:.3f}"
    )
    print(
        "  "
        f"FO utility UTA={row['utility_uta']:+.6f}  "
        f"ACTA={row['utility_acta']:+.6f}  "
        f"Δu={row['delta_utility']:+.6f}  "
        f"grad_shift={row['gradient_shift']:.4f}"
    )

    return row


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to save.")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values))


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
        default="diagnostics/results/step10_2",
    )

    args = parser.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    seeds = parse_seeds(args.seeds)

    rows: List[Dict[str, float]] = []

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

    print("\n" + "=" * 86)
    print("STEP 10.2 SUMMARY — alignment-gradient agreement with oracle target gradient")
    print("=" * 86)

    for source_id, target_id in scenarios:
        scenario = f"{source_id}->{target_id}"
        subset = [r for r in rows if r["scenario"] == scenario]

        dcos = [float(r["delta_cos"]) for r in subset]
        dutil = [float(r["delta_utility"]) for r in subset]
        shift = [float(r["gradient_shift"]) for r in subset]

        print(
            f"{scenario:>10} | "
            f"mean Δcos={mean(dcos):+.6f} | "
            f"seed-positive={sum(x > 0 for x in dcos)}/{len(dcos)} | "
            f"mean Δutility={mean(dutil):+.6f} | "
            f"grad-shift={mean(shift):.4f}"
        )

    all_dcos = [float(r["delta_cos"]) for r in rows]
    all_dutil = [float(r["delta_utility"]) for r in rows]

    print("-" * 86)
    print(
        f"{'OVERALL':>10} | "
        f"mean Δcos={mean(all_dcos):+.6f} | "
        f"seed-positive={sum(x > 0 for x in all_dcos)}/{len(all_dcos)} | "
        f"mean Δutility={mean(all_dutil):+.6f}"
    )
    print("=" * 86)

    output_path = Path(args.output_dir) / "seed_summary.csv"
    write_csv(output_path, rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()