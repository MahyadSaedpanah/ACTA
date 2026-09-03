"""
Distillation utilities for ACTA's local semantic model.

The global source-only teacher provides one logit per temporal path.

The local additive student learns to reconstruct that logit using:

    S_c(pi)
        =
        b_c
        +
        (1 / L_s)
        * sum_{e in pi} g_c(e)

No target data or target labels are used here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from semantic_preparation.local_admissibility import (
    LocalSemanticEnergy,
)


# ============================================================
# PAD VARIABLE-LENGTH LOCAL PATHS
# ============================================================

def records_to_local_batch(
    records,
    device="cpu",
):
    """
    Convert variable-length local path records into
    a padded tensor batch.

    Returns
    -------
    local_x:
        [N, L_max, 8]

    mask:
        [N, L_max]

    source_length:
        [N]

    labels:
        [N]
    """

    if len(records) == 0:
        raise ValueError(
            "records cannot be empty."
        )

    n = len(records)

    path_lengths = [
        int(
            r["local"].shape[0]
        )
        for r in records
    ]

    max_length = max(
        path_lengths
    )

    feature_dim = int(
        records[0]["local"].shape[1]
    )

    local_x = torch.zeros(
        n,
        max_length,
        feature_dim,
        dtype=torch.float32,
        device=device,
    )

    mask = torch.zeros(
        n,
        max_length,
        dtype=torch.float32,
        device=device,
    )

    source_length = torch.empty(
        n,
        dtype=torch.float32,
        device=device,
    )

    labels = torch.empty(
        n,
        dtype=torch.long,
        device=device,
    )

    for i, record in enumerate(
        records
    ):

        local = torch.as_tensor(
            record["local"],
            dtype=torch.float32,
            device=device,
        )

        length = local.shape[0]

        local_x[
            i,
            :length,
        ] = local

        mask[
            i,
            :length,
        ] = 1.0

        source_length[i] = float(
            record["source_length"]
        )

        labels[i] = int(
            record["label"]
        )

    return (
        local_x,
        mask,
        source_length,
        labels,
    )


# ============================================================
# LOCAL STUDENT TRAINING
# ============================================================

def fit_local_student(
    records,
    teacher_logits,
    seed,
    epochs=300,
    lr=3e-3,
    weight_decay=1e-4,
    device="cpu",
):
    """
    Full-batch source-only distillation.

    Objective:

        SmoothL1(
            local_student_path_logit,
            global_teacher_path_logit
        )

    No validation-set checkpoint selection is used.
    The fixed final epoch is returned.
    """

    teacher_logits = np.asarray(
        teacher_logits,
        dtype=np.float32,
    ).reshape(-1)

    if len(teacher_logits) != len(records):
        raise ValueError(
            "teacher_logits and records "
            "must have equal length."
        )

    torch.manual_seed(
        int(seed)
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed)
        )

    model = LocalSemanticEnergy().to(
        device
    )

    (
        local_x,
        mask,
        source_length,
        _,
    ) = records_to_local_batch(
        records,
        device=device,
    )

    teacher_target = torch.as_tensor(
        teacher_logits,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(
            weight_decay
        ),
    )

    history = []

    for epoch in range(
        1,
        int(epochs) + 1
    ):

        model.train()

        prediction = model(
            local_x,
            mask,
            source_length,
        )

        loss = F.smooth_l1_loss(
            prediction,
            teacher_target,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        history.append(
            float(
                loss.detach()
                .cpu()
                .item()
            )
        )

    return (
        model,
        history,
    )


# ============================================================
# LOCAL STUDENT INFERENCE
# ============================================================

@torch.no_grad()
def local_student_logits(
    model,
    records,
    device="cpu",
):
    """
    Evaluate local additive path logits.
    """

    model.eval()

    (
        local_x,
        mask,
        source_length,
        _,
    ) = records_to_local_batch(
        records,
        device=device,
    )

    logits = model(
        local_x,
        mask,
        source_length,
    )

    return (
        logits.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


# ============================================================
# SMALL DEPENDENCY-FREE SPEARMAN IMPLEMENTATION
# ============================================================

def _average_ranks(x):
    """
    Average ranks with tie handling.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    order = np.argsort(
        x,
        kind="mergesort",
    )

    sorted_x = x[
        order
    ]

    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    i = 0

    while i < len(x):

        j = i + 1

        while (
            j < len(x)
            and
            sorted_x[j]
            ==
            sorted_x[i]
        ):
            j += 1

        # 1-based average rank.
        avg_rank = (
            (
                i + 1
            )
            +
            j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    return ranks


def spearman_correlation(
    x,
    y,
):
    """
    Rank correlation without requiring scipy.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    if len(x) != len(y):
        raise ValueError(
            "x and y must have equal length."
        )

    if len(x) < 2:
        return float("nan")

    rx = _average_ranks(x)
    ry = _average_ranks(y)

    rx = (
        rx - rx.mean()
    )

    ry = (
        ry - ry.mean()
    )

    denom = np.sqrt(
        np.sum(
            rx * rx
        )
        *
        np.sum(
            ry * ry
        )
    )

    if denom <= 0:
        return float("nan")

    return float(
        np.sum(
            rx * ry
        )
        /
        denom
    )