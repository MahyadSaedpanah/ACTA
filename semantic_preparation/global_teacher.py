"""
Global source-only temporal semantic admissibility teacher for ACTA.

This module implements the validated global path-geometry teacher.

Important:
    - SOURCE data only.
    - No target ID/data/labels are needed.
    - Samples are split BEFORE temporal pairs are constructed.
    - DTW cost is NOT an input to the semantic teacher.
    - The teacher uses only symmetric temporal path geometry.
"""

from __future__ import annotations

import hashlib

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from semantic_preparation.dtw_utils import (
    dtw_path_multivariate,
    local_edge_features,
)


# ============================================================
# CONSTANTS
# ============================================================

PATH_GRID = 16

DEFAULT_PAIRS_PER_TYPE = 80

GLOBAL_TEACHER_RANDOM_STATE = 20260828


# ============================================================
# DETERMINISTIC SOURCE-ONLY RNG
# ============================================================

def stable_seed(*parts):
    """
    Stable process-independent seed.

    Unlike Python hash(), this gives the same result across
    different Python processes/machines.

    IMPORTANT:
    ACTA semantic preparation should pass only
    dataset/source/class/stage information here.
    """

    text = "||".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def source_split_seed(
    dataset_name,
    source_id,
):
    """
    One deterministic A/B/C split per source domain.
    """

    return stable_seed(
        "ACTA_SEMANTIC_V1",
        "source_split",
        dataset_name,
        source_id,
    )


def class_pair_seed(
    dataset_name,
    source_id,
    class_id,
    stage,
):
    """
    Deterministic class-specific pair-sampling seed.

    stage examples:
        teacher
        student
        eval
    """

    return stable_seed(
        "ACTA_SEMANTIC_V1",
        "pair_sampling",
        dataset_name,
        source_id,
        class_id,
        stage,
    )


# ============================================================
# THREE-WAY SOURCE SPLIT
# ============================================================

def three_way_class_split(
    labels,
    num_classes,
    seed,
):
    """
    Split SOURCE samples class-wise:

        A = 50%  global teacher
        B = 25%  local-student distillation
        C = rest  untouched source evaluation

    Samples are split BEFORE any temporal pairs are formed.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    ).reshape(-1)

    rng = np.random.default_rng(
        int(seed)
    )

    split_a = {}
    split_b = {}
    split_c = {}

    for class_id in range(
        int(num_classes)
    ):

        idx = np.where(
            labels == class_id
        )[0]

        if len(idx) < 8:
            raise RuntimeError(
                f"Too few samples for class "
                f"{class_id}: {len(idx)}. "
                "Need at least 8 for A/B/C splitting."
            )

        idx = rng.permutation(idx)

        n = len(idx)

        n_a = int(
            round(
                0.50 * n
            )
        )

        n_b = int(
            round(
                0.25 * n
            )
        )

        # Keep sufficient samples in all three subsets.
        n_a = max(
            3,
            min(
                n_a,
                n - 5,
            ),
        )

        n_b = max(
            2,
            min(
                n_b,
                n - n_a - 2,
            ),
        )

        split_a[class_id] = (
            idx[:n_a]
            .astype(np.int64)
        )

        split_b[class_id] = (
            idx[n_a:n_a + n_b]
            .astype(np.int64)
        )

        split_c[class_id] = (
            idx[n_a + n_b:]
            .astype(np.int64)
        )

    return (
        split_a,
        split_b,
        split_c,
    )


# ============================================================
# GLOBAL PATH DESCRIPTOR
# ============================================================

def directed_path_descriptor(
    pi,
    pj,
    n,
    m,
    grid_size=PATH_GRID,
):
    """
    Convert a monotone DTW path into a fixed-size
    directed temporal-warp descriptor.

    Output:

        lag          : grid_size
        distortion   : grid_size - 1

    Therefore with PATH_GRID=16:

        dimension = 16 + 15 = 31
    """

    pi = np.asarray(
        pi,
        dtype=np.int64,
    )

    pj = np.asarray(
        pj,
        dtype=np.int64,
    )

    n = int(n)
    m = int(m)
    grid_size = int(grid_size)

    if n <= 0 or m <= 0:
        raise ValueError(
            "Sequence lengths must be positive."
        )

    if grid_size < 2:
        raise ValueError(
            "grid_size must be >= 2."
        )

    if len(pi) != len(pj):
        raise ValueError(
            "pi and pj must have equal length."
        )

    sums = np.zeros(
        n,
        dtype=np.float64,
    )

    counts = np.zeros(
        n,
        dtype=np.float64,
    )

    for i, j in zip(
        pi,
        pj,
    ):

        sums[int(i)] += float(j)

        counts[int(i)] += 1.0

    valid = (
        counts > 0
    )

    if not valid.any():
        raise RuntimeError(
            "Temporal path contains no valid cells."
        )

    mapping = np.zeros(
        n,
        dtype=np.float64,
    )

    mapping[valid] = (
        sums[valid]
        /
        counts[valid]
    )

    idx = np.arange(
        n,
        dtype=np.float64,
    )

    if not np.all(valid):

        mapping = np.interp(
            idx,
            idx[valid],
            mapping[valid],
        )

    u = (
        idx
        /
        max(
            n - 1,
            1,
        )
    )

    v = (
        mapping
        /
        max(
            m - 1,
            1,
        )
    )

    grid = np.linspace(
        0.0,
        1.0,
        grid_size,
    )

    vg = np.interp(
        grid,
        u,
        v,
    )

    # --------------------------------------------------------
    # Temporal displacement
    # --------------------------------------------------------

    lag = np.abs(
        vg - grid
    )

    # --------------------------------------------------------
    # Local compression / dilation
    # --------------------------------------------------------

    du = (
        1.0
        /
        (
            grid_size - 1
        )
    )

    slope = (
        np.diff(vg)
        /
        du
    )

    slope = np.clip(
        slope,
        1e-3,
        1e3,
    )

    distortion = np.abs(
        np.log(slope)
    )

    descriptor = np.concatenate(
        [
            lag,
            distortion,
        ],
        axis=0,
    )

    return descriptor.astype(
        np.float64
    )


def symmetric_path_descriptor(
    pi,
    pj,
    n,
    m,
    grid_size=PATH_GRID,
):
    """
    Pair ordering should not define semantic admissibility.

    descriptor(a,b)
        =
        0.5 *
        (
            directed(a,b)
            +
            directed(b,a)
        )
    """

    forward = directed_path_descriptor(
        pi,
        pj,
        n,
        m,
        grid_size=grid_size,
    )

    inverse = directed_path_descriptor(
        pj,
        pi,
        m,
        n,
        grid_size=grid_size,
    )

    return (
        0.5
        *
        (
            forward
            +
            inverse
        )
    )


# ============================================================
# PAIR SAMPLING
# ============================================================

def sample_same_pairs(
    indices,
    n_pairs,
    rng,
):
    """
    Uniform sample without replacement from all
    unordered same-class pairs.
    """

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    pairs = []

    for i in range(
        len(indices)
    ):

        for j in range(
            i + 1,
            len(indices),
        ):

            pairs.append(
                (
                    int(indices[i]),
                    int(indices[j]),
                )
            )

    if len(pairs) == 0:
        raise RuntimeError(
            "No same-class pairs available."
        )

    n_select = min(
        int(n_pairs),
        len(pairs),
    )

    selected = rng.choice(
        len(pairs),
        size=n_select,
        replace=False,
    )

    return [
        pairs[int(k)]
        for k in selected
    ]


def sample_cross_pairs(
    anchor_indices,
    other_indices,
    n_pairs,
    rng,
):
    """
    Sample ordered anchor-vs-other-class pairs
    without replacement.
    """

    anchor_indices = np.asarray(
        anchor_indices,
        dtype=np.int64,
    )

    other_indices = np.asarray(
        other_indices,
        dtype=np.int64,
    )

    n_other = len(
        other_indices
    )

    total = (
        len(anchor_indices)
        *
        n_other
    )

    if total == 0:
        raise RuntimeError(
            "No cross-class pairs available."
        )

    n_select = min(
        int(n_pairs),
        total,
    )

    selected = rng.choice(
        total,
        size=n_select,
        replace=False,
    )

    pairs = []

    for flat in selected:

        flat = int(flat)

        a = (
            flat
            //
            n_other
        )

        b = (
            flat
            %
            n_other
        )

        pairs.append(
            (
                int(
                    anchor_indices[a]
                ),
                int(
                    other_indices[b]
                ),
            )
        )

    return pairs


# ============================================================
# PAIR RECORD CONSTRUCTION
# ============================================================

def _build_pair_record(
    cache,
    pair,
    label,
):
    """
    Run DTW once and retain BOTH:

        global descriptor  -> global teacher
        local edge geometry -> local student

    This avoids recomputing the same path later.
    """

    i, j = pair

    a = cache[int(i)]
    b = cache[int(j)]

    (
        pi,
        pj,
        path_length,
        total_cost,
    ) = dtw_path_multivariate(
        a,
        b,
    )

    descriptor = (
        symmetric_path_descriptor(
            pi,
            pj,
            a.shape[0],
            b.shape[0],
        )
    )

    local = local_edge_features(
        pi,
        pj,
        a.shape[0],
        b.shape[0],
    )

    return {
        "pair": (
            int(i),
            int(j),
        ),

        "label":
            int(label),

        "descriptor":
            descriptor.astype(
                np.float32
            ),

        "local":
            local.astype(
                np.float32
            ),

        # Critical normalization used by LocalSemanticEnergy
        "source_length":
            int(
                a.shape[0]
            ),

        "path_length":
            int(
                path_length
            ),

        # Diagnostic only.
        # NOT an input to the semantic teacher.
        "dtw_cost":
            float(
                total_cost
            ),
    }


def build_class_pair_records(
    cache,
    class_indices,
    class_id,
    num_classes,
    rng,
    n_pairs=DEFAULT_PAIRS_PER_TYPE,
):
    """
    Build positive and negative temporal-pair records
    for one semantic class.

    positive:
        class_id vs same class

    negative:
        class_id vs every other class
    """

    class_id = int(
        class_id
    )

    positive_idx = np.asarray(
        class_indices[class_id],
        dtype=np.int64,
    )

    other_parts = []

    for c in range(
        int(num_classes)
    ):

        if c == class_id:
            continue

        other_parts.append(
            np.asarray(
                class_indices[c],
                dtype=np.int64,
            )
        )

    if len(other_parts) == 0:
        raise RuntimeError(
            "Need at least two classes."
        )

    other_idx = np.concatenate(
        other_parts,
        axis=0,
    )

    positive_pairs = sample_same_pairs(
        positive_idx,
        n_pairs,
        rng,
    )

    negative_pairs = sample_cross_pairs(
        positive_idx,
        other_idx,
        n_pairs,
        rng,
    )

    records = []

    for pair in positive_pairs:

        records.append(
            _build_pair_record(
                cache,
                pair,
                label=1,
            )
        )

    for pair in negative_pairs:

        records.append(
            _build_pair_record(
                cache,
                pair,
                label=0,
            )
        )

    return records


# ============================================================
# GLOBAL TEACHER
# ============================================================

def records_to_global_xy(
    records,
):
    """
    Extract only global path descriptors + labels.

    DTW costs are intentionally excluded.
    """

    if len(records) == 0:
        raise ValueError(
            "records cannot be empty."
        )

    x = np.stack(
        [
            r["descriptor"]
            for r in records
        ],
        axis=0,
    ).astype(
        np.float64
    )

    y = np.asarray(
        [
            r["label"]
            for r in records
        ],
        dtype=np.int64,
    )

    return x, y


def make_global_teacher():
    """
    Exact simple teacher used by the validated
    temporal-admissibility experiment.
    """

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),

            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=
                        GLOBAL_TEACHER_RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_global_teacher(
    records,
):
    """
    Fit teacher using global path geometry only.
    """

    x, y = records_to_global_xy(
        records
    )

    if len(
        np.unique(y)
    ) != 2:

        raise RuntimeError(
            "Global teacher requires both "
            "positive and negative pairs."
        )

    teacher = make_global_teacher()

    teacher.fit(
        x,
        y,
    )

    return teacher


def global_teacher_logits(
    teacher,
    records,
):
    """
    Return pre-sigmoid teacher logits.

    These logits, not probabilities, are what the
    local additive student will distill.
    """

    x, _ = records_to_global_xy(
        records
    )

    logits = teacher.decision_function(
        x
    )

    return np.asarray(
        logits,
        dtype=np.float32,
    ).reshape(-1)