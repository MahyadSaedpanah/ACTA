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
import torch.nn as nn
import torch.nn.functional as F


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


def _transition_aware_soft_dp_reference(
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

def _transition_aware_soft_dp_antidiagonal(
    feature_cost,
    semantic_bonus=None,
    gamma=0.1,
    return_table=False,
):
    """
    Vectorized anti-diagonal implementation of ACTA's
    transition-aware soft dynamic program.

    This implements exactly the same recurrence as
    _transition_aware_soft_dp_reference, but computes all
    cells on the same anti-diagonal in parallel.

    Cells satisfying

        i + j = k

    are independent given anti-diagonals k-1 and k-2.

    Therefore the number of sequential Python recurrence
    steps is reduced from

        Ls * Lt

    to

        Ls + Lt - 1.

    No ACTA objective, transition rule, semantic term,
    temperature, or boundary condition is changed.
    """

    ls, lt = _validate_inputs(
        feature_cost,
        semantic_bonus,
    )

    device = feature_cost.device

    # Each entry stores one complete anti-diagonal:
    #
    #     [..., number_of_cells_on_diagonal]
    #
    diagonals = []

    # Starting i-index associated with each anti-diagonal.
    # This lets us map a grid coordinate i to its position
    # inside the corresponding vectorized diagonal.
    diagonal_i_starts = []

    num_diagonals = ls + lt - 1

    for k in range(num_diagonals):

        # ----------------------------------------------------
        # Valid cells satisfying:
        #
        #     i + j = k
        #
        # ----------------------------------------------------

        i_start = max(
            0,
            k - (lt - 1),
        )

        i_end = min(
            ls - 1,
            k,
        )

        i_idx = torch.arange(
            i_start,
            i_end + 1,
            device=device,
            dtype=torch.long,
        )

        j_idx = k - i_idx

        diagonal_i_starts.append(
            i_start
        )

        # ----------------------------------------------------
        # Local feature costs for the entire anti-diagonal.
        #
        # Shape:
        #
        #     [..., M]
        #
        # where M is the number of cells on this diagonal.
        # ----------------------------------------------------

        local = feature_cost[
            ...,
            i_idx,
            j_idx,
        ]

        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        if k == 0:

            if semantic_bonus is None:

                sem_start = 0.0

            else:

                sem_start = semantic_bonus[
                    ...,
                    0,
                    0,
                    SEM_START,
                ]

            current = (
                local
                -
                sem_start.unsqueeze(-1)
                if torch.is_tensor(sem_start)
                else local - sem_start
            )

            diagonals.append(
                current
            )

            continue

        # ----------------------------------------------------
        # Semantic contributions for current cells
        # ----------------------------------------------------

        if semantic_bonus is None:

            sem_diagonal = 0.0
            sem_vertical = 0.0
            sem_horizontal = 0.0

        else:

            sem_diagonal = semantic_bonus[
                ...,
                i_idx,
                j_idx,
                SEM_DIAGONAL,
            ]

            sem_vertical = semantic_bonus[
                ...,
                i_idx,
                j_idx,
                SEM_VERTICAL,
            ]

            sem_horizontal = semantic_bonus[
                ...,
                i_idx,
                j_idx,
                SEM_HORIZONTAL,
            ]

        # ----------------------------------------------------
        # Previous anti-diagonal k - 1
        # ----------------------------------------------------

        previous = diagonals[
            k - 1
        ]

        previous_i_start = diagonal_i_starts[
            k - 1
        ]

        previous_length = previous.shape[
            -1
        ]

        # Horizontal predecessor:
        #
        #     (i, j - 1)
        #
        horizontal_pos = (
            i_idx
            -
            previous_i_start
        )

        # Vertical predecessor:
        #
        #     (i - 1, j)
        #
        vertical_pos = (
            i_idx
            -
            1
            -
            previous_i_start
        )

        # Boundary cells do not use all predecessor types.
        # We clamp unused indices only so vectorized indexing
        # remains valid. torch.where below ensures those
        # branches do not contribute to the selected result.
        horizontal_pos_safe = horizontal_pos.clamp(
            0,
            previous_length - 1,
        )

        vertical_pos_safe = vertical_pos.clamp(
            0,
            previous_length - 1,
        )

        horizontal_previous = previous[
            ...,
            horizontal_pos_safe,
        ]

        vertical_previous = previous[
            ...,
            vertical_pos_safe,
        ]

        horizontal_value = (
            local
            +
            horizontal_previous
            -
            sem_horizontal
        )

        vertical_value = (
            local
            +
            vertical_previous
            -
            sem_vertical
        )

        # ----------------------------------------------------
        # Interior recurrence
        # ----------------------------------------------------

        if k >= 2:

            two_back = diagonals[
                k - 2
            ]

            two_back_i_start = diagonal_i_starts[
                k - 2
            ]

            two_back_length = two_back.shape[
                -1
            ]

            diagonal_pos = (
                i_idx
                -
                1
                -
                two_back_i_start
            )

            diagonal_pos_safe = diagonal_pos.clamp(
                0,
                two_back_length - 1,
            )

            diagonal_previous = two_back[
                ...,
                diagonal_pos_safe,
            ]

            diagonal_candidate = (
                diagonal_previous
                -
                sem_diagonal
            )

            vertical_candidate = (
                vertical_previous
                -
                sem_vertical
            )

            horizontal_candidate = (
                horizontal_previous
                -
                sem_horizontal
            )

            candidates = torch.stack(
                [
                    diagonal_candidate,
                    vertical_candidate,
                    horizontal_candidate,
                ],
                dim=-1,
            )

            interior_value = (
                local
                +
                softmin(
                    candidates,
                    gamma,
                )
            )

        else:

            # k == 1 contains boundary cells only.
            # Placeholder is never selected for an interior
            # cell because no interior cell exists yet.
            interior_value = local

        # ----------------------------------------------------
        # Select the correct recurrence for each cell.
        #
        # Top row:
        #
        #     i == 0
        #
        # Left column:
        #
        #     j == 0
        #
        # Everything else is interior.
        # ----------------------------------------------------

        is_top_row = (
            i_idx == 0
        )

        is_left_column = (
            j_idx == 0
        )

        current = torch.where(
            is_top_row,
            horizontal_value,
            torch.where(
                is_left_column,
                vertical_value,
                interior_value,
            ),
        )

        diagonals.append(
            current
        )

    # --------------------------------------------------------
    # Final anti-diagonal always contains only (Ls-1, Lt-1).
    # --------------------------------------------------------

    terminal = diagonals[
        -1
    ][
        ...,
        0,
    ]

    if not return_table:
        return terminal

    # --------------------------------------------------------
    # Reconstruct the conventional [..., Ls, Lt] table.
    #
    # This path is primarily for tests / diagnostics.
    # Training only needs the terminal value.
    # --------------------------------------------------------

    rows = []

    for i in range(ls):

        row = []

        for j in range(lt):

            k = i + j

            position = (
                i
                -
                diagonal_i_starts[k]
            )

            row.append(
                diagonals[k][
                    ...,
                    position,
                ]
            )

        rows.append(
            torch.stack(
                row,
                dim=-1,
            )
        )

    table = torch.stack(
        rows,
        dim=-2,
    )

    return terminal, table


def transition_aware_soft_dp(
    feature_cost,
    semantic_bonus=None,
    gamma=0.1,
    return_table=False,
):
    """
    Public ACTA DP interface.

    Uses the exact-equivalent vectorized anti-diagonal
    implementation validated against the original reference
    recurrence in terminal cost, full DP table, and
    occupancy gradients.
    """
    return _transition_aware_soft_dp_antidiagonal(
        feature_cost=feature_cost,
        semantic_bonus=semantic_bonus,
        gamma=gamma,
        return_table=return_table,
    )

# ============================================================
# TEMPORAL FEATURE RESAMPLING
# ============================================================

def resample_temporal_features(
    features,
    output_length,
):
    """
    Resample temporal feature maps to ACTA's semantic grid.

    Input:
        [B, D, L]

    Output:
        [B, D, G]

    align_corners=True preserves the normalized temporal
    endpoints 0 and 1 used by ACTA's semantic geometry.
    """

    if features.ndim != 3:
        raise ValueError(
            "features must have shape [B, D, L]."
        )

    output_length = int(
        output_length
    )

    if output_length <= 0:
        raise ValueError(
            "output_length must be positive."
        )

    if features.shape[-1] == output_length:
        return features

    return F.interpolate(
        features,
        size=output_length,
        mode="linear",
        align_corners=True,
    )


# ============================================================
# FEATURE COST
# ============================================================

def cosine_feature_cost(
    source_features,
    target_features,
    eps=1e-8,
):
    """
    Pairwise temporal cosine distance:

        d_ij = 1 - cos(h_i^s, h_j^t)

    Inputs:
        source_features [B, D, Ls]
        target_features [B, D, Lt]

    Returns:
        [B, Ls, Lt]
    """

    if (
        source_features.ndim != 3
        or
        target_features.ndim != 3
    ):
        raise ValueError(
            "source_features and target_features "
            "must have shape [B,D,L]."
        )

    if (
        source_features.shape[0]
        != target_features.shape[0]
    ):
        raise ValueError(
            "Source and target batch sizes must match."
        )

    if (
        source_features.shape[1]
        != target_features.shape[1]
    ):
        raise ValueError(
            "Source and target feature dimensions "
            "must match."
        )

    source = F.normalize(
        source_features.transpose(
            1,
            2,
        ),
        p=2,
        dim=-1,
        eps=eps,
    )

    target = F.normalize(
        target_features.transpose(
            1,
            2,
        ),
        p=2,
        dim=-1,
        eps=eps,
    )

    similarity = torch.bmm(
        source,
        target.transpose(
            1,
            2,
        ),
    )

    # Numerical roundoff can occasionally make cosine
    # similarity microscopically larger than one.
    cost = (
        1.0 - similarity
    ).clamp_min(
        0.0
    )

    return cost


def scale_feature_cost(
    feature_cost,
    eps=1e-6,
):
    """
    Positive scale-only normalization.

        d_tilde =
            d /
            stopgrad(mean(d))

    This preserves the feature-only path ordering.

    Returns
    -------
    scaled_cost
    detached_scale
    """

    if feature_cost.ndim < 2:
        raise ValueError(
            "feature_cost must contain "
            "two temporal dimensions."
        )

    scale = (
        feature_cost
        .detach()
        .mean(
            dim=(-2, -1),
            keepdim=True,
        )
        .clamp_min(eps)
    )

    scaled = (
        feature_cost
        /
        scale
    )

    return (
        scaled,
        scale,
    )


# ============================================================
# DETACHED SOFT PATH EXTRACTION
# ============================================================

def extract_soft_alignment(
    scaled_feature_cost,
    semantic_bonus=None,
    gamma=0.1,
):
    """
    Select the soft temporal path using a detached feature
    cost.

    Path construction therefore does not backpropagate
    second-order derivatives through the DP.

    The returned soft alignment matrix is detached.

        A = d D_soft / d C_path
    """

    with torch.enable_grad():

        path_cost = (
            scaled_feature_cost
            .detach()
            .requires_grad_(
                True
            )
        )

        if semantic_bonus is not None:

            semantic_bonus = (
                semantic_bonus
                .detach()
            )

        terminal = (
            transition_aware_soft_dp(
                feature_cost=
                    path_cost,

                semantic_bonus=
                    semantic_bonus,

                gamma=
                    gamma,
            )
        )

        alignment = torch.autograd.grad(
            terminal.sum(),
            path_cost,
            create_graph=False,
            retain_graph=False,
        )[0]

    # --------------------------------------------------------
    # Numerical occupancy stabilization
    #
    # Mathematically, dD/dC is a soft path occupancy:
    #
    #     0 <= A_ij <= 1
    #
    # Float32 autograd through a long soft-DP recurrence can
    # produce tiny violations such as 1.00004.
    #
    # First reject any LARGE violation so that clamping cannot
    # hide a real implementation bug, then enforce the exact
    # mathematical range.
    # --------------------------------------------------------

    alignment = alignment.detach()

    numerical_tolerance = 1e-3

    if (
        alignment.min().item()
        < -numerical_tolerance
        or
        alignment.max().item()
        >
        1.0 + numerical_tolerance
    ):
        raise RuntimeError(
            "Soft alignment occupancy is outside "
            "the expected numerical range. "
            f"min={alignment.min().item()}, "
            f"max={alignment.max().item()}."
        )

    alignment = alignment.clamp(
        min=0.0,
        max=1.0,
    )

    return (
        alignment,
        terminal.detach(),
    )


# ============================================================
# ACTA PAIR ALIGNMENT CORE
# ============================================================

class ACTAAlignmentCore(nn.Module):
    """
    Pair-level alignment primitive shared by UTA and ACTA.

    UTA:
        use_semantics=False

    ACTA:
        use_semantics=True

    When lambda_sem == 0, ACTA becomes feature-only UTA.
    """

    def __init__(
        self,
        semantic_bank,
        gamma=0.1,
        lambda_sem=1.0,
        eps=1e-6,
        detach_source=True,
    ):
        super().__init__()

        self.semantic_bank = (
            semantic_bank
        )

        self.gamma = float(
            gamma
        )

        self.lambda_sem = float(
            lambda_sem
        )

        self.eps = float(
            eps
        )

        self.detach_source = bool(
            detach_source
        )


    def forward(
        self,
        source_features,
        target_features,
        class_id=None,
        use_semantics=True,
        semantic_class_id=None,
        reliability_class_id=None,
        return_details=False,
    ):
        """
        Parameters
        ----------
        source_features:
            [B, D, Ls]

        target_features:
            [B, D, Lt]

        class_id:
            Semantic class shared by this pair batch.

        use_semantics:
            False -> UTA
            True  -> ACTA

        Returns
        -------
        scalar alignment loss

        or diagnostics dict if return_details=True.
        """

        if (
            source_features.ndim != 3
            or
            target_features.ndim != 3
        ):
            raise ValueError(
                "Expected feature maps [B,D,L]."
            )

        if (
            source_features.shape[0]
            != target_features.shape[0]
        ):
            raise ValueError(
                "Source/target pair batch mismatch."
            )

        if (
            source_features.shape[1]
            != target_features.shape[1]
        ):
            raise ValueError(
                "Source/target feature dimension mismatch."
            )

        batch_size = int(
            source_features.shape[0]
        )

        # ----------------------------------------------------
        # Alignment branch should not pull source-reference
        # features away from the supervised source geometry.
        #
        # Source CE will still update the shared encoder in
        # the full ACTA algorithm.
        # ----------------------------------------------------

        if self.detach_source:

            source_for_alignment = (
                source_features.detach()
            )

        else:

            source_for_alignment = (
                source_features
            )

        # ----------------------------------------------------
        # Semantic deployment resolution comes from package.
        # No dataset-specific hard-code.
        # ----------------------------------------------------

        semantic_length = int(
            self.semantic_bank
            .semantic_temporal_length
        )

        source_grid = (
            resample_temporal_features(
                source_for_alignment,
                semantic_length,
            )
        )

        target_grid = (
            resample_temporal_features(
                target_features,
                semantic_length,
            )
        )

        # ----------------------------------------------------
        # Feature geometry
        # ----------------------------------------------------

        feature_cost = (
            cosine_feature_cost(
                source_grid,
                target_grid,
            )
        )

        (
            scaled_feature_cost,
            feature_scale,
        ) = scale_feature_cost(
            feature_cost,
            eps=self.eps,
        )

        # ----------------------------------------------------
        # Semantic geometry
        # ----------------------------------------------------

        semantic = None

        semantic_is_active = (
            bool(use_semantics)
            and
            self.lambda_sem != 0.0
        )

        if semantic_is_active:

            if class_id is None:
                raise ValueError(
                    "class_id is required when "
                    "semantic alignment is active."
                )

            if semantic_class_id is None:
                semantic_class_id = class_id

            if reliability_class_id is None:
                reliability_class_id = class_id

            class_id = int(
                class_id
            )

            semantic_single = (
                self.semantic_bank
                .semantic_bonus(
                    class_id=
                        int(
                            semantic_class_id
                        ),

                    reliability_class_id=
                        int(
                            reliability_class_id
                        ),

                    source_length=
                        semantic_length,

                    target_length=
                        semantic_length,

                    lambda_sem=
                        self.lambda_sem,

                    device=
                        feature_cost.device,

                    dtype=
                        feature_cost.dtype,
                )
            )

            semantic = (
                semantic_single
                .unsqueeze(0)
                .expand(
                    batch_size,
                    -1,
                    -1,
                    -1,
                )
            )

        # ----------------------------------------------------
        # Structured path selection
        # ----------------------------------------------------

        (
            alignment,
            terminal_cost,
        ) = extract_soft_alignment(
            scaled_feature_cost=
                scaled_feature_cost,

            semantic_bonus=
                semantic,

            gamma=
                self.gamma,
        )

        # ----------------------------------------------------
        # Representation loss
        #
        # IMPORTANT:
        #
        # Path is selected using scaled detached cost.
        # Representation learning uses ORIGINAL,
        # non-detached feature distance.
        # ----------------------------------------------------

        alignment_mass = (
            alignment.sum(
                dim=(-2, -1)
            )
            .clamp_min(
                self.eps
            )
        )

        pair_loss = (
            (
                alignment
                *
                feature_cost
            )
            .sum(
                dim=(-2, -1)
            )
            /
            alignment_mass
        )

        loss = (
            pair_loss.mean()
        )

        if not return_details:
            return loss

        return {
            "loss":
                loss,

            "pair_loss":
                pair_loss,

            "alignment":
                alignment,

            "feature_cost":
                feature_cost,

            "scaled_feature_cost":
                scaled_feature_cost,

            "feature_scale":
                feature_scale,

            "semantic_bonus":
                semantic,

            "terminal_cost":
                terminal_cost,

            "semantic_length":
                semantic_length,
        }