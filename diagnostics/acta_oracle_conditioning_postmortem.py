"""
Step 10.3 — Oracle-class conditioning post-mortem for ACTA.

Purpose
-------
Step 10.1 established that ACTA makes temporal paths more semantically
admissible. Step 10.2 showed that this does not reliably improve the
representation gradient relative to the oracle supervised target gradient.

This diagnostic isolates the remaining ambiguity:

    Is the failure caused mainly by soft EMA class conditioning,
    or does it remain even when the correct target class is supplied?

Target labels are used ONLY as a post-hoc oracle diagnostic.

For each target batch at the common source-initialized theta_S, we construct
the same per-class UTA and ACTA alignment losses l_c(x_t), with exactly matched
same-class source references between UTA and ACTA.

We then compare two conditioning rules:

    EMA:
        L(b) = sum_c p_ema(c|x_b) l_c(x_b)

    ORACLE:
        L(b) = l_{y_b}(x_b)

For each rule we measure cosine agreement with the oracle supervised target
gradient g_T = d CE_target / d H_t.

Interpretation
--------------
If ACTA improves strongly under ORACLE but not EMA:
    class conditioning is the main bottleneck.

If ACTA still fails under ORACLE:
    the deeper problem is the representation/alignment objective itself;
    semantically legal correspondence is not sufficient for a useful update.
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


def make_full_train_loader(train_dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        train_dataset,
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
    Returns:
        uta_matrix  [B,C]
        acta_matrix [B,C]

    For each class c, source references are sampled ONCE and reused exactly
    for UTA and ACTA.
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

    uta_matrix = torch.stack(uta_by_class, dim=1)
    acta_matrix = torch.stack(acta_by_class, dim=1)

    return uta_matrix, acta_matrix


def cosine_per_sample(
    g_align: torch.Tensor,
    g_target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    a = g_align.flatten(1)
    t = g_target.flatten(1)

    dot = (a * t).sum(dim=1)
    denom = (a.norm(dim=1) * t.norm(dim=1)).clamp_min(eps)

    return dot / denom


def relative_gradient_shift(
    g_uta: torch.Tensor,
    g_acta: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    u = g_uta.flatten(1)
    a = g_acta.flatten(1)
    return (a - u).norm(dim=1) / u.norm(dim=1).clamp_min(eps)


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

    # Common theta_S.
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

    delta_ema_all = []
    delta_oracle_all = []

    cos_uta_ema_all = []
    cos_acta_ema_all = []
    cos_uta_oracle_all = []
    cos_acta_oracle_all = []

    shift_ema_all = []
    shift_oracle_all = []

    ema_correct_all = []
    ema_conf_all = []

    processed_batches = 0

    for batch_index, (trg_x, trg_y) in enumerate(target_loader):

        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        trg_x = trg_x.float().to(device)
        trg_y = trg_y.long().to(device)

        with torch.no_grad():
            target_probabilities = algorithm.ema_probabilities(trg_x)
            h0 = algorithm.feature_extractor.forward_features(trg_x)

            ema_pred = target_probabilities.argmax(dim=1)
            ema_correct_all.append((ema_pred == trg_y).float().cpu())
            ema_conf_all.append(target_probabilities.max(dim=1).values.cpu())

        h = h0.detach().requires_grad_(True)

        uta_matrix, acta_matrix = matched_class_loss_matrices(
            algorithm=algorithm,
            target_temporal_features=h,
            k=args.k,
        )

        # EMA conditioning
        probs = target_probabilities.detach()

        uta_ema = (probs * uta_matrix).sum(dim=1)
        acta_ema = (probs * acta_matrix).sum(dim=1)

        # Oracle conditioning
        gather_index = trg_y[:, None]

        uta_oracle = uta_matrix.gather(1, gather_index).squeeze(1)
        acta_oracle = acta_matrix.gather(1, gather_index).squeeze(1)

        # Oracle supervised target gradient
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

        g_uta_ema = torch.autograd.grad(
            uta_ema.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_acta_ema = torch.autograd.grad(
            acta_ema.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_uta_oracle = torch.autograd.grad(
            uta_oracle.sum(),
            h,
            retain_graph=True,
            create_graph=False,
        )[0]

        g_acta_oracle = torch.autograd.grad(
            acta_oracle.sum(),
            h,
            retain_graph=False,
            create_graph=False,
        )[0]

        cos_uta_ema = cosine_per_sample(g_uta_ema, g_target)
        cos_acta_ema = cosine_per_sample(g_acta_ema, g_target)

        cos_uta_oracle = cosine_per_sample(g_uta_oracle, g_target)
        cos_acta_oracle = cosine_per_sample(g_acta_oracle, g_target)

        delta_ema = cos_acta_ema - cos_uta_ema
        delta_oracle = cos_acta_oracle - cos_uta_oracle

        shift_ema = relative_gradient_shift(g_uta_ema, g_acta_ema)
        shift_oracle = relative_gradient_shift(g_uta_oracle, g_acta_oracle)

        cos_uta_ema_all.append(cos_uta_ema.detach().cpu())
        cos_acta_ema_all.append(cos_acta_ema.detach().cpu())
        cos_uta_oracle_all.append(cos_uta_oracle.detach().cpu())
        cos_acta_oracle_all.append(cos_acta_oracle.detach().cpu())

        delta_ema_all.append(delta_ema.detach().cpu())
        delta_oracle_all.append(delta_oracle.detach().cpu())

        shift_ema_all.append(shift_ema.detach().cpu())
        shift_oracle_all.append(shift_oracle.detach().cpu())

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError("No target batches were processed.")

    def cat(xs):
        return torch.cat(xs)

    cos_uta_ema = cat(cos_uta_ema_all)
    cos_acta_ema = cat(cos_acta_ema_all)
    cos_uta_oracle = cat(cos_uta_oracle_all)
    cos_acta_oracle = cat(cos_acta_oracle_all)

    delta_ema = cat(delta_ema_all)
    delta_oracle = cat(delta_oracle_all)

    shift_ema = cat(shift_ema_all)
    shift_oracle = cat(shift_oracle_all)

    ema_correct = cat(ema_correct_all)
    ema_conf = cat(ema_conf_all)

    row: Dict[str, float] = {
        "scenario": f"{source_id}->{target_id}",
        "source": str(source_id),
        "target": str(target_id),
        "seed": int(seed),
        "n_target": int(delta_ema.numel()),

        "ema_target_accuracy": float(ema_correct.mean().item()),
        "ema_mean_confidence": float(ema_conf.mean().item()),

        "cos_uta_ema": float(cos_uta_ema.mean().item()),
        "cos_acta_ema": float(cos_acta_ema.mean().item()),
        "delta_cos_ema": float(delta_ema.mean().item()),
        "fraction_acta_better_ema": float(
            (delta_ema > 0).float().mean().item()
        ),

        "cos_uta_oracle": float(cos_uta_oracle.mean().item()),
        "cos_acta_oracle": float(cos_acta_oracle.mean().item()),
        "delta_cos_oracle": float(delta_oracle.mean().item()),
        "fraction_acta_better_oracle": float(
            (delta_oracle > 0).float().mean().item()
        ),

        "oracle_minus_ema_delta_gain": float(
            (delta_oracle.mean() - delta_ema.mean()).item()
        ),

        "gradient_shift_ema": float(shift_ema.mean().item()),
        "gradient_shift_oracle": float(shift_oracle.mean().item()),
    }

    print(
        f"\n[Step10.3] dataset={args.dataset} "
        f"scenario={source_id}->{target_id} seed={seed}"
    )
    print(
        "  "
        f"EMA acc={row['ema_target_accuracy']:.3f}  "
        f"conf={row['ema_mean_confidence']:.3f}"
    )
    print(
        "  "
        f"EMA    Δcos={row['delta_cos_ema']:+.6f}  "
        f"ACTA-better={row['fraction_acta_better_ema']:.3f}  "
        f"shift={row['gradient_shift_ema']:.4f}"
    )
    print(
        "  "
        f"ORACLE Δcos={row['delta_cos_oracle']:+.6f}  "
        f"ACTA-better={row['fraction_acta_better_oracle']:.3f}  "
        f"shift={row['gradient_shift_oracle']:.4f}  "
        f"oracle-gain={row['oracle_minus_ema_delta_gain']:+.6f}"
    )

    return row


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        default="diagnostics/results/step10_3",
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

    print("\n" + "=" * 90)
    print("STEP 10.3 SUMMARY — EMA vs oracle class conditioning")
    print("=" * 90)

    for source_id, target_id in scenarios:
        scenario = f"{source_id}->{target_id}"
        subset = [r for r in rows if r["scenario"] == scenario]

        ema_delta = [float(r["delta_cos_ema"]) for r in subset]
        oracle_delta = [float(r["delta_cos_oracle"]) for r in subset]
        oracle_gain = [
            float(r["oracle_minus_ema_delta_gain"])
            for r in subset
        ]

        print(
            f"{scenario:>10} | "
            f"EMA Δcos={mean(ema_delta):+.6f} | "
            f"ORACLE Δcos={mean(oracle_delta):+.6f} | "
            f"oracle-gain={mean(oracle_gain):+.6f} | "
            f"oracle seed-positive="
            f"{sum(x > 0 for x in oracle_delta)}/{len(oracle_delta)}"
        )

    all_ema = [float(r["delta_cos_ema"]) for r in rows]
    all_oracle = [float(r["delta_cos_oracle"]) for r in rows]
    all_gain = [float(r["oracle_minus_ema_delta_gain"]) for r in rows]

    print("-" * 90)
    print(
        f"{'OVERALL':>10} | "
        f"EMA Δcos={mean(all_ema):+.6f} | "
        f"ORACLE Δcos={mean(all_oracle):+.6f} | "
        f"oracle-gain={mean(all_gain):+.6f} | "
        f"oracle seed-positive="
        f"{sum(x > 0 for x in all_oracle)}/{len(all_oracle)}"
    )
    print("=" * 90)

    output_path = Path(args.output_dir) / "seed_summary.csv"
    write_csv(output_path, rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()