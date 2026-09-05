"""
Transition-aware soft dynamic programming for ACTA.

This module implements only the structured alignment primitive.
It does NOT know about CNNs, classes, semantic packages, EMA,
or source reference sampling.

Semantic bonus channel convention
---------------------------------
0 : start
1 : diagonal   (1, 1)
2 : vertical   (1, 0)
3 : horizontal (0, 1)

The convention exactly matches semantic_preparation/dtw_utils.py.
"""

from __future__ import annotations

import torch


SEM_START = 0
SEM_DIAGONAL = 1
SEM_VERTICAL = 2
SEM_HORIZONTAL = 3

NUM_SEMANTIC_TRANSITIONS = 4


def softmin(
    values,
    gamma,
):
    """
    Numerically stable differentiable minimum.

        softmin_gamma(a)
            =
            -gamma * logsumexp(-a / gamma)

    Parameters
    ----------
    values:
        Tensor whose last dimension contains candidates.

    gamma:
        Positive smoothing temperature.
    """

    gamma = float(gamma)

    if gamma <= 0.0:
        raise ValueError(
            "gamma must be strictly positive."
        )

    return (
        -gamma
        *
        torch.logsumexp(
            -values / gamma,
            dim=-1,
        )
    )


def _validate_inputs(
    feature_cost,
    semantic_bonus,
):
    if not torch.is_tensor(feature_cost):
        raise TypeError(
            "feature_cost must be a torch.Tensor."
        )

    if feature_cost.ndim < 2:
        raise ValueError(
            "feature_cost must have shape "
            "[..., Ls, Lt]."
        )

    ls = int(feature_cost.shape[-2])
    lt = int(feature_cost.shape[-1])

    if ls < 1 or lt < 1:
        raise ValueError(
            "Temporal dimensions must be non-empty."
        )

    if semantic_bonus is not None:

        if not torch.is_tensor(
            semantic_bonus
        ):
            raise TypeError(
                "semantic_bonus must be a torch.Tensor."
            )

        expected_shape = (
            tuple(feature_cost.shape)
            +
            (NUM_SEMANTIC_TRANSITIONS,)
        )

        if (
            tuple(semantic_bonus.shape)
            !=
            expected_shape
        ):
            raise ValueError(
                "semantic_bonus must have shape "
                "[..., Ls, Lt, 4]. "
                f"Expected {expected_shape}, "
                f"got {tuple(semantic_bonus.shape)}."
            )

        if (
            semantic_bonus.device
            !=
            feature_cost.device
        ):
            raise ValueError(
                "feature_cost and semantic_bonus "
                "must be on the same device."
            )

    return ls, lt


def transition_aware_soft_dp(
    feature_cost,
    semantic_bonus=None,
    gamma=0.1,
    return_table=False,
):
    """
    ACTA transition-aware soft dynamic program.

    Parameters
    ----------
    feature_cost:
        Tensor [..., Ls, Lt].

        Local feature dissimilarity d_ij.

    semantic_bonus:
        Optional tensor [..., Ls, Lt, 4].

        This argument is already assumed to contain the
        complete scaled semantic contribution, e.g.

            lambda * kappa_c * g_c(edge) / rho_c

        Higher semantic_bonus makes a transition cheaper.

        If None, the method becomes feature-only UTA.

    gamma:
        Soft-min temperature.

    return_table:
        If True, also return full accumulated DP table.

    Returns
    -------
    terminal_cost:
        Tensor with shape equal to leading batch dimensions.

    table:
        [..., Ls, Lt], only when return_table=True.

    Recurrence
    ----------
    Start:

        R_00 =
            d_00
            - semantic_start_00

    Boundaries:

        R_i0 =
            d_i0
            + R_(i-1)0
            - semantic_vertical_i0

        R_0j =
            d_0j
            + R_0(j-1)
            - semantic_horizontal_0j

    Interior:

        R_ij =
            d_ij
            +
            softmin(
                R_(i-1)(j-1) - semantic_diagonal_ij,
                R_(i-1)j     - semantic_vertical_ij,
                R_i(j-1)     - semantic_horizontal_ij
            )
    """

    ls, lt = _validate_inputs(
        feature_cost,
        semantic_bonus,
    )

    # --------------------------------------------------------
    # Semantic access helper
    # --------------------------------------------------------

    def sem(i, j, move):

        if semantic_bonus is None:
            return 0.0

        return semantic_bonus[
            ...,
            i,
            j,
            move,
        ]

    # --------------------------------------------------------
    # Build DP table without unsafe in-place recurrence.
    # Each cell is a tensor over any leading batch dimensions.
    # --------------------------------------------------------

    rows = []

    for i in range(ls):

        row = []

        for j in range(lt):

            local = feature_cost[
                ...,
                i,
                j,
            ]

            # ------------------------------------------------
            # START
            # ------------------------------------------------

            if i == 0 and j == 0:

                value = (
                    local
                    -
                    sem(
                        0,
                        0,
                        SEM_START,
                    )
                )

            # ------------------------------------------------
            # FIRST COLUMN = VERTICAL ONLY
            # ------------------------------------------------

            elif j == 0:

                value = (
                    local
                    +
                    rows[i - 1][0]
                    -
                    sem(
                        i,
                        0,
                        SEM_VERTICAL,
                    )
                )

            # ------------------------------------------------
            # FIRST ROW = HORIZONTAL ONLY
            # ------------------------------------------------

            elif i == 0:

                value = (
                    local
                    +
                    row[j - 1]
                    -
                    sem(
                        0,
                        j,
                        SEM_HORIZONTAL,
                    )
                )

            # ------------------------------------------------
            # INTERIOR
            # ------------------------------------------------

            else:

                diagonal = (
                    rows[i - 1][j - 1]
                    -
                    sem(
                        i,
                        j,
                        SEM_DIAGONAL,
                    )
                )

                vertical = (
                    rows[i - 1][j]
                    -
                    sem(
                        i,
                        j,
                        SEM_VERTICAL,
                    )
                )

                horizontal = (
                    row[j - 1]
                    -
                    sem(
                        i,
                        j,
                        SEM_HORIZONTAL,
                    )
                )

                candidates = torch.stack(
                    [
                        diagonal,
                        vertical,
                        horizontal,
                    ],
                    dim=-1,
                )

                value = (
                    local
                    +
                    softmin(
                        candidates,
                        gamma,
                    )
                )

            row.append(
                value
            )

        rows.append(
            row
        )

    terminal = rows[-1][-1]

    if not return_table:
        return terminal

    table = torch.stack(
        [
            torch.stack(
                row,
                dim=-1,
            )
            for row in rows
        ],
        dim=-2,
    )

    return terminal, table