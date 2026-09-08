"""
Step 13 — Relative / Discriminative Semantic Transport Diagnostic for ACTA.

Motivation
----------
Steps 10–12 established:

1) TSA makes ACTA paths more semantically admissible.
2) Absolute same-class attraction along those paths does not reliably produce
   a better target-task gradient than UTA.
3) Source-margin projection repairs some harmful update components, but it
   helps UTA more than ACTA. Therefore update safety alone does not explain
   the remaining TSA-vs-UTA gap.

This diagnostic tests a different objective:

    do NOT only attract the target toward the selected source class;
    improve its transport relation to the selected class RELATIVE to
    competing source classes.

For an oracle target class y and class-specific transport costs l_c(x):

    L_rel(x,y)
        = l_y(x)
          - mean_{k != y} l_k(x)

This is parameter-free, bounded because l_c is built from cosine distances,
and directly discriminative:

    minimize positive-class transport cost
    while increasing competing-class transport costs.

We evaluate four gradients at the same source-initialized theta_S:

    absolute UTA
    absolute ACTA
    relative UTA
    relative ACTA

Target labels are used ONLY for this post-hoc oracle diagnostic.
No target labels enter semantic preparation or ACTA training.
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


def matched_class_loss_matrices(
    algorithm: ACTA,
    target_temporal_features: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return matched class-transport costs:
        UTA  [B,C]
        ACTA [B,C]

    For every class c, source references are sampled once and reused exactly
    between UTA and ACTA.
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


def oracle_absolute_loss(
    class_loss_matrix: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return class_loss_matrix.gather(
        1,
        labels[:, None],
    ).squeeze(1)


def oracle_relative_loss(
    class_loss_matrix: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    L_rel = positive transport - mean competing transport.
    """

    num_classes = int(class_loss_matrix.shape[1])

    if num_classes < 2:
        raise RuntimeError("Relative transport requires at least two classes.")

    positive = oracle_absolute_loss(
        class_loss_matrix,
        labels,
    )

    competing_mean = (
        class_loss_matrix.sum(dim=1) - positive
    ) / float(num_classes - 1)

    return positive - competing_mean


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

    cos_abs_uta_all = []
    cos_abs_acta_all = []
    cos_rel_uta_all = []
    cos_rel_acta_all = []

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

        abs_uta = oracle_absolute_loss(
            uta_matrix,
            trg_y,
        )
        abs_acta = oracle_absolute_loss(
            acta_matrix,
            trg_y,
        )

        rel_uta = oracle_relative_loss(
            uta_matrix,
            trg_y,
        )
        rel_acta = oracle_relative_loss(
            acta_matrix,
            trg_y,
        )

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

        g_abs_uta = torch.autograd.grad(
            abs_uta.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_abs_acta = torch.autograd.grad(
            abs_acta.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_rel_uta = torch.autograd.grad(
            rel_uta.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_rel_acta = torch.autograd.grad(
            rel_acta.sum(),
            h,
            retain_graph=False,
            create_graph=False,
        )[0]

        # Gradients and descent directions have the same cosine after both
        # signs are flipped, so compare gradients directly.
        cos_abs_uta_all.append(
            cosine_per_sample(g_abs_uta, g_target).detach().cpu()
        )
        cos_abs_acta_all.append(
            cosine_per_sample(g_abs_acta, g_target).detach().cpu()
        )
        cos_rel_uta_all.append(
            cosine_per_sample(g_rel_uta, g_target).detach().cpu()
        )
        cos_rel_acta_all.append(
            cosine_per_sample(g_rel_acta, g_target).detach().cpu()
        )

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("No target batches were processed.")

    cos_abs_uta = torch.cat(cos_abs_uta_all)
    cos_abs_acta = torch.cat(cos_abs_acta_all)
    cos_rel_uta = torch.cat(cos_rel_uta_all)
    cos_rel_acta = torch.cat(cos_rel_acta_all)

    d_rel_acta_vs_rel_uta = cos_rel_acta - cos_rel_uta
    d_rel_acta_vs_abs_uta = cos_rel_acta - cos_abs_uta
    d_rel_acta_vs_abs_acta = cos_rel_acta - cos_abs_acta
    d_rel_uta_vs_abs_uta = cos_rel_uta - cos_abs_uta

    row = {
        "scenario": f"{source_id}->{target_id}",
        "source": str(source_id),
        "target": str(target_id),
        "seed": int(seed),
        "n_target": int(cos_abs_uta.numel()),

        "cos_absolute_uta": float(cos_abs_uta.mean().item()),
        "cos_absolute_acta": float(cos_abs_acta.mean().item()),
        "cos_relative_uta": float(cos_rel_uta.mean().item()),
        "cos_relative_acta": float(cos_rel_acta.mean().item()),

        "delta_relative_acta_vs_relative_uta": float(
            d_rel_acta_vs_rel_uta.mean().item()
        ),
        "delta_relative_acta_vs_absolute_uta": float(
            d_rel_acta_vs_abs_uta.mean().item()
        ),
        "delta_relative_acta_vs_absolute_acta": float(
            d_rel_acta_vs_abs_acta.mean().item()
        ),
        "delta_relative_uta_vs_absolute_uta": float(
            d_rel_uta_vs_abs_uta.mean().item()
        ),

        "fraction_relative_acta_better_than_relative_uta": float(
            (d_rel_acta_vs_rel_uta > 0).float().mean().item()
        ),
        "fraction_relative_acta_better_than_absolute_uta": float(
            (d_rel_acta_vs_abs_uta > 0).float().mean().item()
        ),
    }

    print(
        f"\n[Step13] dataset={args.dataset} "
        f"scenario={source_id}->{target_id} seed={seed}"
    )
    print(
        "  "
        f"Abs UTA={row['cos_absolute_uta']:+.6f}  "
        f"Abs ACTA={row['cos_absolute_acta']:+.6f}  "
        f"Rel UTA={row['cos_relative_uta']:+.6f}  "
        f"Rel ACTA={row['cos_relative_acta']:+.6f}"
    )
    print(
        "  "
        f"Rel-ACTA−Rel-UTA="
        f"{row['delta_relative_acta_vs_relative_uta']:+.6f}  "
        f"Rel-ACTA−Abs-UTA="
        f"{row['delta_relative_acta_vs_absolute_uta']:+.6f}  "
        f"Rel-UTA−Abs-UTA="
        f"{row['delta_relative_uta_vs_absolute_uta']:+.6f}"
    )
    print(
        "  "
        f"sample Rel-ACTA>Rel-UTA="
        f"{row['fraction_relative_acta_better_than_relative_uta']:.3f}  "
        f"Rel-ACTA>Abs-UTA="
        f"{row['fraction_relative_acta_better_than_absolute_uta']:.3f}"
    )

    return row


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
        default="diagnostics/results/step13",
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

    print("\n" + "=" * 106)
    print("STEP 13 SUMMARY — relative / discriminative semantic transport")
    print("=" * 106)

    for source_id, target_id in scenarios:

        scenario = f"{source_id}->{target_id}"
        subset = [r for r in rows if r["scenario"] == scenario]

        d_sem = [
            float(r["delta_relative_acta_vs_relative_uta"])
            for r in subset
        ]
        d_vs_abs = [
            float(r["delta_relative_acta_vs_absolute_uta"])
            for r in subset
        ]
        d_rel = [
            float(r["delta_relative_uta_vs_absolute_uta"])
            for r in subset
        ]

        print(
            f"{scenario:>10} | "
            f"Rel-ACTA−Rel-UTA={mean(d_sem):+.6f} | "
            f"Rel-ACTA−Abs-UTA={mean(d_vs_abs):+.6f} | "
            f"Rel-UTA−Abs-UTA={mean(d_rel):+.6f} | "
            f"semantic seed-positive="
            f"{sum(x > 0 for x in d_sem)}/{len(d_sem)}"
        )

    overall_sem = [
        float(r["delta_relative_acta_vs_relative_uta"])
        for r in rows
    ]
    overall_vs_abs = [
        float(r["delta_relative_acta_vs_absolute_uta"])
        for r in rows
    ]
    overall_rel = [
        float(r["delta_relative_uta_vs_absolute_uta"])
        for r in rows
    ]

    print("-" * 106)
    print(
        f"{'OVERALL':>10} | "
        f"Rel-ACTA−Rel-UTA={mean(overall_sem):+.6f} | "
        f"Rel-ACTA−Abs-UTA={mean(overall_vs_abs):+.6f} | "
        f"Rel-UTA−Abs-UTA={mean(overall_rel):+.6f} | "
        f"semantic seed-positive="
        f"{sum(x > 0 for x in overall_sem)}/{len(overall_sem)}"
    )
    print("=" * 106)

    output_path = Path(args.output_dir) / "seed_summary.csv"
    write_csv(output_path, rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()