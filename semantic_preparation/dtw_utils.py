"""
DTW and temporal-path utilities for ACTA.

This module defines the path convention that is shared by:

1. source-side temporal semantic preparation,
2. the local admissibility teacher,
3. the ACTA semantic alignment dynamic program.

Path convention
---------------
For consecutive path cells

    previous = (i_prev, j_prev)
    current  = (i, j)

we define:

    diagonal   : (i-i_prev, j-j_prev) = (1, 1)
    vertical   : (1, 0)
    horizontal : (0, 1)

The first cell is assigned the special START move.

IMPORTANT:
Do not change this convention independently in later ACTA modules.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit


# ============================================================
# ACTA PATH-MOVE CONVENTION
#
# Feature ordering is intentionally fixed:
#
# [start, diagonal, vertical, horizontal]
# ============================================================

MOVE_START = 0
MOVE_DIAGONAL = 1
MOVE_VERTICAL = 2
MOVE_HORIZONTAL = 3

NUM_MOVE_TYPES = 4

MOVE_NAMES = {
    MOVE_START: "start",
    MOVE_DIAGONAL: "diagonal",
    MOVE_VERTICAL: "vertical",
    MOVE_HORIZONTAL: "horizontal",
}


# Discovery protocol used 128 -> 64 for UCIHAR.
# This is a downsampling FACTOR, not a fixed sequence length.
DEFAULT_DOWNSAMPLE = 2


# ============================================================
# 1. RAW SEQUENCE PREPARATION
# ============================================================

def prepare_dtw_sequence(
    x,
    downsample=DEFAULT_DOWNSAMPLE,
    eps=1e-6,
):
    """
    Prepare one multivariate time-series sample for DTW.

    Parameters
    ----------
    x:
        Tensor/array with shape [C, T].
        A one-dimensional [T] input is also accepted.

    downsample:
        Average-pooling factor in the raw temporal axis.
        A factor of 1 disables downsampling.

    Returns
    -------
    np.ndarray
        Contiguous float64 array with shape [T_dtw, C].

    Notes
    -----
    The transformation matches the ACTA discovery protocol:

        [C, T]
            -> [T, C]
            -> average downsampling
            -> per-sample, per-channel z-normalization
    """

    downsample = int(downsample)

    if downsample < 1:
        raise ValueError(
            "downsample must be >= 1."
        )

    if torch.is_tensor(x):
        x = (
            x.detach()
            .cpu()
            .numpy()
        )
    else:
        x = np.asarray(x)

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.ndim == 1:
        x = x[None, :]

    if x.ndim != 2:
        raise ValueError(
            "Expected one sample with shape [C, T], "
            f"got {x.shape}."
        )

    # [C, T] -> [T, C]
    x = x.T

    if downsample > 1:

        usable_length = (
            x.shape[0]
            //
            downsample
            *
            downsample
        )

        if usable_length == 0:
            raise ValueError(
                "Sequence is shorter than the requested "
                f"downsampling factor ({downsample})."
            )

        x = x[:usable_length]

        x = (
            x.reshape(
                usable_length // downsample,
                downsample,
                x.shape[1],
            )
            .mean(axis=1)
        )

    # Per-sample, per-channel normalization.
    mean = x.mean(
        axis=0,
        keepdims=True,
    )

    std = x.std(
        axis=0,
        keepdims=True,
    )

    x = (
        x - mean
    ) / (
        std + eps
    )

    return np.ascontiguousarray(
        x,
        dtype=np.float64,
    )


# ============================================================
# 2. EXACT MULTIVARIATE DTW
#
# This preserves the predecessor ordering used in the
# discovery notebook:
#
#     diagonal > up > left
#
# when costs are tied.
# ============================================================

@njit(cache=True)
def _dtw_path_multivariate_numba(
    x,
    y,
):
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    inf = 1e300

    cumulative = np.empty(
        (n + 1, m + 1),
        dtype=np.float64,
    )

    cumulative[:, :] = inf
    cumulative[0, 0] = 0.0

    # predecessor:
    #
    # 0 = diagonal
    # 1 = up
    # 2 = left
    predecessor = np.zeros(
        (n + 1, m + 1),
        dtype=np.int8,
    )

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            local_cost = 0.0

            for k in range(d):

                diff = (
                    x[i - 1, k]
                    -
                    y[j - 1, k]
                )

                local_cost += (
                    diff * diff
                )

            diagonal = cumulative[
                i - 1,
                j - 1,
            ]

            up = cumulative[
                i - 1,
                j,
            ]

            left = cumulative[
                i,
                j - 1,
            ]

            # Exact tie-breaking used during discovery.
            if (
                diagonal <= up
                and
                diagonal <= left
            ):

                best = diagonal
                predecessor[i, j] = 0

            elif up <= left:

                best = up
                predecessor[i, j] = 1

            else:

                best = left
                predecessor[i, j] = 2

            cumulative[i, j] = (
                local_cost
                +
                best
            )

    # Maximum possible monotone path length.
    max_length = (
        n + m + 2
    )

    pi_reverse = np.empty(
        max_length,
        dtype=np.int32,
    )

    pj_reverse = np.empty(
        max_length,
        dtype=np.int32,
    )

    i = n
    j = m
    length = 0

    while (
        i > 0
        and
        j > 0
    ):

        pi_reverse[length] = (
            i - 1
        )

        pj_reverse[length] = (
            j - 1
        )

        length += 1

        move = predecessor[
            i,
            j,
        ]

        if move == 0:
            i -= 1
            j -= 1

        elif move == 1:
            i -= 1

        else:
            j -= 1

    pi = np.empty(
        length,
        dtype=np.int32,
    )

    pj = np.empty(
        length,
        dtype=np.int32,
    )

    for k in range(length):

        reverse_k = (
            length - 1 - k
        )

        pi[k] = pi_reverse[
            reverse_k
        ]

        pj[k] = pj_reverse[
            reverse_k
        ]

    return (
        pi,
        pj,
        length,
        cumulative[n, m],
    )


def dtw_path_multivariate(
    x,
    y,
):
    """
    Exact multivariate DTW path.

    Inputs must have shape:

        x: [T_x, C]
        y: [T_y, C]

    Local cost is squared Euclidean distance across channels.

    Returns
    -------
    pi, pj:
        Monotone path coordinates.

    path_length:
        Number of path cells.

    total_cost:
        Total accumulated DTW cost.
    """

    x = np.ascontiguousarray(
        x,
        dtype=np.float64,
    )

    y = np.ascontiguousarray(
        y,
        dtype=np.float64,
    )

    if (
        x.ndim != 2
        or
        y.ndim != 2
    ):
        raise ValueError(
            "DTW inputs must have shape [T, C]."
        )

    if (
        x.shape[0] == 0
        or
        y.shape[0] == 0
    ):
        raise ValueError(
            "DTW inputs cannot be empty."
        )

    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "DTW inputs must have the same "
            "number of channels."
        )

    (
        pi,
        pj,
        path_length,
        total_cost,
    ) = _dtw_path_multivariate_numba(
        x,
        y,
    )

    validate_path(
        pi,
        pj,
        x.shape[0],
        y.shape[0],
    )

    return (
        pi,
        pj,
        int(path_length),
        float(total_cost),
    )


# ============================================================
# 3. PATH VALIDATION
# ============================================================

def validate_path(
    pi,
    pj,
    n,
    m,
):
    """
    Validate ACTA's monotone DTW path convention.
    """

    pi = np.asarray(pi)
    pj = np.asarray(pj)

    if (
        pi.ndim != 1
        or
        pj.ndim != 1
    ):
        raise ValueError(
            "Path coordinates must be 1-D."
        )

    if len(pi) != len(pj):
        raise ValueError(
            "pi and pj must have equal length."
        )

    if len(pi) == 0:
        raise ValueError(
            "Path cannot be empty."
        )

    if (
        pi[0] != 0
        or
        pj[0] != 0
    ):
        raise ValueError(
            "Path must start at (0, 0)."
        )

    if (
        pi[-1] != n - 1
        or
        pj[-1] != m - 1
    ):
        raise ValueError(
            "Path must end at "
            f"({n - 1}, {m - 1})."
        )

    if (
        np.any(pi < 0)
        or
        np.any(pi >= n)
        or
        np.any(pj < 0)
        or
        np.any(pj >= m)
    ):
        raise ValueError(
            "Path contains out-of-range indices."
        )

    if len(pi) > 1:

        di = np.diff(pi)
        dj = np.diff(pj)

        valid = (
            ((di == 1) & (dj == 1))
            |
            ((di == 1) & (dj == 0))
            |
            ((di == 0) & (dj == 1))
        )

        if not np.all(valid):
            bad = np.where(
                ~valid
            )[0][0]

            raise ValueError(
                "Invalid path transition at "
                f"{bad}->{bad + 1}: "
                f"delta=({di[bad]}, {dj[bad]})."
            )

    return True


# ============================================================
# 4. PATH -> MOVE TYPES
# ============================================================

def path_move_ids(
    pi,
    pj,
):
    """
    Convert path coordinates to ACTA move IDs.

    Returned array has the same length as the path.

    First position:
        START

    Later positions:
        DIAGONAL / VERTICAL / HORIZONTAL
    """

    pi = np.asarray(
        pi,
        dtype=np.int64,
    )

    pj = np.asarray(
        pj,
        dtype=np.int64,
    )

    if (
        pi.ndim != 1
        or
        pj.ndim != 1
        or
        len(pi) != len(pj)
        or
        len(pi) == 0
    ):
        raise ValueError(
            "Invalid path arrays."
        )

    moves = np.empty(
        len(pi),
        dtype=np.int64,
    )

    moves[0] = MOVE_START

    for k in range(
        1,
        len(pi),
    ):

        di = (
            int(pi[k])
            -
            int(pi[k - 1])
        )

        dj = (
            int(pj[k])
            -
            int(pj[k - 1])
        )

        if (
            di == 1
            and
            dj == 1
        ):

            moves[k] = (
                MOVE_DIAGONAL
            )

        elif (
            di == 1
            and
            dj == 0
        ):

            # source/time-i advances
            # while target/time-j stays fixed
            moves[k] = (
                MOVE_VERTICAL
            )

        elif (
            di == 0
            and
            dj == 1
        ):

            # target/time-j advances
            # while source/time-i stays fixed
            moves[k] = (
                MOVE_HORIZONTAL
            )

        else:

            raise ValueError(
                "Invalid monotone path move "
                f"at index {k}: "
                f"delta=({di}, {dj})."
            )

    return moves


def path_move_names(
    pi,
    pj,
):
    """
    Human-readable move sequence, useful for debugging.
    """

    moves = path_move_ids(
        pi,
        pj,
    )

    return [
        MOVE_NAMES[int(move)]
        for move in moves
    ]


# ============================================================
# 5. LOCAL ACTA EDGE FEATURES
#
# Exact feature ordering:
#
# [
#   u,
#   v,
#   v-u,
#   |v-u|,
#   start,
#   diagonal,
#   vertical,
#   horizontal
# ]
# ============================================================

def local_edge_features(
    pi,
    pj,
    n,
    m,
):
    """
    Build local geometric features along one temporal path.

    The coordinates are normalized to [0, 1], making the
    representation independent of absolute sequence length.
    """

    validate_path(
        pi,
        pj,
        n,
        m,
    )

    pi = np.asarray(
        pi,
        dtype=np.int64,
    )

    pj = np.asarray(
        pj,
        dtype=np.int64,
    )

    moves = path_move_ids(
        pi,
        pj,
    )

    denom_i = max(
        int(n) - 1,
        1,
    )

    denom_j = max(
        int(m) - 1,
        1,
    )

    u = (
        pi.astype(np.float64)
        /
        denom_i
    )

    v = (
        pj.astype(np.float64)
        /
        denom_j
    )

    features = np.zeros(
        (len(pi), 8),
        dtype=np.float64,
    )

    features[:, 0] = u
    features[:, 1] = v
    features[:, 2] = (
        v - u
    )

    features[:, 3] = np.abs(
        v - u
    )

    for k, move in enumerate(
        moves
    ):
        features[
            k,
            4 + int(move),
        ] = 1.0

    return features