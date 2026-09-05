"""
Build ACTA's source-only semantic package.

For every source class c:

    A -> global temporal-semantic teacher
    B -> local additive student distillation
    C -> source-side reliability calibration

Saved per class:

    g_c          local semantic model
    kappa_c      source identifiability / reliability
    rho_c        local-score RMS scale

No target domain is accepted by this script.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score


# ============================================================
# Repository root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from configs.data_model_configs import get_dataset_class

from semantic_preparation.dtw_utils import (
    DEFAULT_DOWNSAMPLE,
    prepare_dtw_sequence,
)

from semantic_preparation.global_teacher import (
    PATH_GRID,
    source_split_seed,
    class_pair_seed,
    three_way_class_split,
    build_class_pair_records,
    fit_global_teacher,
    global_teacher_logits,
    records_to_global_xy,
)

from semantic_preparation.local_distillation import (
    fit_local_student,
    local_student_logits,
    spearman_correlation,
)


SEMANTIC_PACKAGE_VERSION = "ACTA_SEMANTIC_V1"


# ============================================================
# Utilities
# ============================================================

def cpu_state_dict(module):
    return {
        key: value.detach().cpu()
        for key, value
        in module.state_dict().items()
    }


def canonicalize_samples(
    x,
    input_channels,
):
    """
    Convert source samples to [N, C, T].
    """

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.float()

    if x.ndim == 2:
        x = x.unsqueeze(1)

    if x.ndim != 3:
        raise ValueError(
            "Expected source samples with shape "
            "[N,C,T] or [N,T,C]."
        )

    if x.shape[1] == input_channels:
        return x.contiguous()

    if x.shape[2] == input_channels:
        return (
            x.permute(0, 2, 1)
            .contiguous()
        )

    raise ValueError(
        "Could not infer channel dimension. "
        f"Shape={tuple(x.shape)}, "
        f"expected channels={input_channels}."
    )


@torch.no_grad()
def local_score_rms(
    model,
    records,
    eps=1e-6,
):
    """
    Source-only scale:

        rho_c = sqrt(E[g_c(e)^2]) + eps

    We use split-B distillation paths.

    Note:
        path_bias is intentionally excluded.
        ACTA DP uses local edge scores g_c(e).
    """

    model.eval()

    sum_sq = 0.0
    count = 0

    for record in records:

        local = torch.as_tensor(
            record["local"],
            dtype=torch.float32,
        )

        score = model.local_score(
            local
        ).double()

        sum_sq += float(
            (score * score)
            .sum()
            .item()
        )

        count += int(
            score.numel()
        )

    if count == 0:
        raise RuntimeError(
            "Cannot compute semantic score scale "
            "from zero edges."
        )

    rms = math.sqrt(
        sum_sq / count
    )

    return float(
        rms + eps
    )


# ============================================================
# One class
# ============================================================

def build_one_class(
    dataset_name,
    source_id,
    class_id,
    num_classes,
    cache,
    split_a,
    split_b,
    split_c,
    n_pairs,
    student_epochs,
    student_lr,
    student_weight_decay,
    device,
):

    print(
        "\n"
        + "=" * 68
    )

    print(
        f"Class {class_id}"
    )

    print(
        "=" * 68
    )

    # --------------------------------------------------------
    # A: global teacher
    # --------------------------------------------------------

    rng_a = np.random.default_rng(
        class_pair_seed(
            dataset_name,
            source_id,
            class_id,
            "teacher",
        )
    )

    records_a = build_class_pair_records(
        cache=cache,
        class_indices=split_a,
        class_id=class_id,
        num_classes=num_classes,
        rng=rng_a,
        n_pairs=n_pairs,
    )

    global_teacher = fit_global_teacher(
        records_a
    )

    _, labels_a = records_to_global_xy(
        records_a
    )

    logits_a = global_teacher_logits(
        global_teacher,
        records_a
    )

    teacher_train_auc = roc_auc_score(
        labels_a,
        logits_a
    )

    # --------------------------------------------------------
    # B: local distillation
    # --------------------------------------------------------

    rng_b = np.random.default_rng(
        class_pair_seed(
            dataset_name,
            source_id,
            class_id,
            "student",
        )
    )

    records_b = build_class_pair_records(
        cache=cache,
        class_indices=split_b,
        class_id=class_id,
        num_classes=num_classes,
        rng=rng_b,
        n_pairs=n_pairs,
    )

    teacher_b = global_teacher_logits(
        global_teacher,
        records_b
    )

    student_seed = class_pair_seed(
        dataset_name,
        source_id,
        class_id,
        "local_student",
    )

    student, history = fit_local_student(
        records=records_b,
        teacher_logits=teacher_b,
        seed=student_seed,
        epochs=student_epochs,
        lr=student_lr,
        weight_decay=student_weight_decay,
        device=device,
    )

    student_b = local_student_logits(
        student,
        records_b,
        device=device,
    )

    rho_b = spearman_correlation(
        teacher_b,
        student_b,
    )

    # --------------------------------------------------------
    # C: source reliability calibration
    # --------------------------------------------------------

    rng_c = np.random.default_rng(
        class_pair_seed(
            dataset_name,
            source_id,
            class_id,
            "eval",
        )
    )

    records_c = build_class_pair_records(
        cache=cache,
        class_indices=split_c,
        class_id=class_id,
        num_classes=num_classes,
        rng=rng_c,
        n_pairs=n_pairs,
    )

    _, labels_c = records_to_global_xy(
        records_c
    )

    teacher_c = global_teacher_logits(
        global_teacher,
        records_c
    )

    student_c = local_student_logits(
        student,
        records_c,
        device=device,
    )

    global_auc_c = float(
        roc_auc_score(
            labels_c,
            teacher_c
        )
    )

    local_auc_c = float(
        roc_auc_score(
            labels_c,
            student_c
        )
    )

    rho_c = float(
        spearman_correlation(
            teacher_c,
            student_c,
        )
    )

    # --------------------------------------------------------
    # ACTA source reliability
    # --------------------------------------------------------

    kappa = float(
        np.clip(
            2.0 * local_auc_c - 1.0,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Semantic local-score scale
    # --------------------------------------------------------

    score_scale = local_score_rms(
        student,
        records_b,
    )

    print(
        f"A records:       {len(records_a)}"
    )

    print(
        f"B records:       {len(records_b)}"
    )

    print(
        f"C records:       {len(records_c)}"
    )

    print(
        f"teacher AUC(C):  {global_auc_c:.6f}"
    )

    print(
        f"local AUC(C):    {local_auc_c:.6f}"
    )

    print(
        f"teacher/local ρ: {rho_c:.6f}"
    )

    print(
        f"kappa:           {kappa:.6f}"
    )

    print(
        f"score scale:     {score_scale:.6f}"
    )

    print(
        f"distill loss:    "
        f"{history[0]:.6f}"
        f" -> "
        f"{history[-1]:.6f}"
    )

    return {
        "state_dict":
            cpu_state_dict(student),

        "source_global_auc":
            global_auc_c,

        "source_local_auc":
            local_auc_c,

        "source_teacher_student_rho":
            rho_c,

        "kappa":
            kappa,

        "score_scale":
            score_scale,

        "distillation": {
            "train_teacher_auc":
                float(
                    teacher_train_auc
                ),

            "student_rho_B":
                float(
                    rho_b
                ),

            "initial_loss":
                float(
                    history[0]
                ),

            "final_loss":
                float(
                    history[-1]
                ),
        },

        "counts": {
            "A_samples":
                int(
                    len(
                        split_a[class_id]
                    )
                ),

            "B_samples":
                int(
                    len(
                        split_b[class_id]
                    )
                ),

            "C_samples":
                int(
                    len(
                        split_c[class_id]
                    )
                ),

            "A_pairs":
                int(
                    len(records_a)
                ),

            "B_pairs":
                int(
                    len(records_b)
                ),

            "C_pairs":
                int(
                    len(records_c)
                ),
        },
    }


# ============================================================
# Full source package
# ============================================================

def build_semantic_package(
    dataset_name,
    source_id,
    data_root,
    output_root,
    n_pairs,
    downsample,
    student_epochs,
    student_lr,
    student_weight_decay,
    device,
):

    dataset_class = get_dataset_class(
        dataset_name
    )

    configs = dataset_class()

    source_path = (
        Path(data_root)
        /
        dataset_name
        /
        f"train_{source_id}.pt"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            source_path
        )

    obj = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )

    samples = canonicalize_samples(
        obj["samples"],
        configs.input_channels,
    )

    labels = obj["labels"]

    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(
            labels
        )

    labels = (
        labels.long()
        .view(-1)
    )

    if len(samples) != len(labels):
        raise ValueError(
            "Sample/label count mismatch."
        )

    print(
        "\nACTA Semantic Preparation"
    )

    print(
        "Dataset:   ",
        dataset_name
    )

    print(
        "Source:    ",
        source_id
    )

    print(
        "Samples:   ",
        len(samples)
    )

    print(
        "Classes:   ",
        configs.num_classes
    )

    print(
        "Downsample:",
        downsample
    )

    # --------------------------------------------------------
    # Target-independent source split
    # --------------------------------------------------------

    split_seed = source_split_seed(
        dataset_name,
        source_id,
    )

    split_a, split_b, split_c = (
        three_way_class_split(
            labels.numpy(),
            num_classes=
                configs.num_classes,
            seed=split_seed,
        )
    )

    # --------------------------------------------------------
    # Prepare each source trajectory exactly once
    # --------------------------------------------------------

    print(
        "\nPreparing DTW cache..."
    )

    cache = [
        prepare_dtw_sequence(
            sample,
            downsample=downsample,
        )
        for sample in samples
    ]

    print(
        "DTW temporal length:",
        cache[0].shape[0]
    )

    print(
        "DTW channels:       ",
        cache[0].shape[1]
    )

    # --------------------------------------------------------
    # Build all class-specific semantics
    # --------------------------------------------------------

    classes = {}

    for class_id in range(
        configs.num_classes
    ):

        classes[int(class_id)] = (
            build_one_class(
                dataset_name=
                    dataset_name,

                source_id=
                    str(source_id),

                class_id=
                    class_id,

                num_classes=
                    configs.num_classes,

                cache=
                    cache,

                split_a=
                    split_a,

                split_b=
                    split_b,

                split_c=
                    split_c,

                n_pairs=
                    n_pairs,

                student_epochs=
                    student_epochs,

                student_lr=
                    student_lr,

                student_weight_decay=
                    student_weight_decay,

                device=
                    device,
            )
        )

    # --------------------------------------------------------
    # Package
    # --------------------------------------------------------

    package = {
        "version":
            SEMANTIC_PACKAGE_VERSION,

        "dataset":
            dataset_name,

        "source_domain":
            str(source_id),

        "num_classes":
            int(
                configs.num_classes
            ),

        "input_channels":
            int(
                configs.input_channels
            ),

        "classes":
            classes,

        "semantic_geometry": {
            "local_feature_dim":
                8,

            "local_feature_order": [
                "u",
                "v",
                "v-u",
                "|v-u|",
                "start",
                "diagonal",
                "vertical",
                "horizontal",
            ],

            "move_convention": {
                "diagonal":
                    [1, 1],

                "vertical":
                    [1, 0],

                "horizontal":
                    [0, 1],
            },

            "global_path_grid":
                int(PATH_GRID),

            "dtw_downsample":
                int(downsample),

            "semantic_temporal_length":
                int(cache[0].shape[0]),

            "semantic_channels":
                int(cache[0].shape[1]),

            "dtw_normalization":
                "per-sample per-channel z-normalization",

            "dtw_local_cost":
                "squared Euclidean",
        },

        "training": {
            "source_split_seed":
                int(split_seed),

            "pairs_per_type":
                int(n_pairs),

            "student_epochs":
                int(student_epochs),

            "student_lr":
                float(student_lr),

            "student_weight_decay":
                float(
                    student_weight_decay
                ),

            "target_information_used":
                False,
        },
    }

    output_path = (
        Path(output_root)
        /
        dataset_name
        /
        f"source_{source_id}"
        /
        "semantic_package.pt"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        package,
        output_path,
    )

    print(
        "\n"
        + "=" * 68
    )

    print(
        "SEMANTIC PACKAGE SAVED"
    )

    print(
        output_path
    )

    print(
        "=" * 68
    )

    print(
        "\nClass summary:"
    )

    for class_id in range(
        configs.num_classes
    ):

        info = classes[class_id]

        print(
            f"class {class_id}: "
            f"AUC={info['source_local_auc']:.4f} | "
            f"kappa={info['kappa']:.4f} | "
            f"scale={info['score_scale']:.4f}"
        )

    return output_path


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build source-only ACTA "
            "temporal semantic package."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="UCIHAR",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="./semantic_packages",
    )

    parser.add_argument(
        "--pairs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
    )

    parser.add_argument(
        "--student_epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--student_lr",
        type=float,
        default=3e-3,
    )

    parser.add_argument(
        "--student_weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if (
        args.device.startswith("cuda")
        and
        not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    build_semantic_package(
        dataset_name=args.dataset,
        source_id=args.source,
        data_root=args.data_root,
        output_root=args.output_root,
        n_pairs=args.pairs,
        downsample=args.downsample,
        student_epochs=args.student_epochs,
        student_lr=args.student_lr,
        student_weight_decay=
            args.student_weight_decay,
        device=args.device,
    )


if __name__ == "__main__":
    main()