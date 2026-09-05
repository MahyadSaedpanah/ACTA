"""
Frozen source-side semantic package interface for ACTA.

This module converts the source-only semantic package into
transition-specific semantic score grids usable by ACTA's DP.

For class c and transition arriving at temporal cell (i,j):

    raw_score = g_c(r_ijm)

    normalized_score =
        raw_score / rho_c

    semantic_bonus =
        lambda_sem * kappa_c * normalized_score

The path_bias of LocalSemanticEnergy is deliberately NOT used
inside the DP, because it is constant across candidate paths
for a fixed class.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from semantic_preparation.local_admissibility import (
    LocalSemanticEnergy,
)


SEM_START = 0
SEM_DIAGONAL = 1
SEM_VERTICAL = 2
SEM_HORIZONTAL = 3

NUM_TRANSITIONS = 4
LOCAL_FEATURE_DIM = 8


# ============================================================
# LOCAL TRANSITION FEATURE GRID
# ============================================================

def build_transition_feature_grid(
    source_length,
    target_length,
    device="cpu",
    dtype=torch.float32,
):
    """
    Build ACTA's local 8-D geometry for every:

        (source position i,
         target position j,
         transition type m)

    Returns
    -------
    Tensor:
        [Ls, Lt, 4, 8]

    Feature order:

        [
            u,
            v,
            v-u,
            |v-u|,
            start,
            diagonal,
            vertical,
            horizontal
        ]

    IMPORTANT:
    Semantic preparation constructs normalized coordinates
    in float64 and casts the final feature tensor afterwards.

    We reproduce that same numerical convention here.
    """

    ls = int(source_length)
    lt = int(target_length)

    if ls <= 0 or lt <= 0:
        raise ValueError(
            "source_length and target_length "
            "must be positive."
        )

    denom_s = max(
        ls - 1,
        1,
    )

    denom_t = max(
        lt - 1,
        1,
    )

    # ========================================================
    # Build geometry in float64, exactly like preparation.
    # Cast to requested dtype only at the end.
    # ========================================================

    geometry_dtype = torch.float64

    u = torch.arange(
        ls,
        device=device,
        dtype=geometry_dtype,
    ) / float(denom_s)

    v = torch.arange(
        lt,
        device=device,
        dtype=geometry_dtype,
    ) / float(denom_t)

    uu = u[:, None].expand(
        ls,
        lt,
    )

    vv = v[None, :].expand(
        ls,
        lt,
    )

    base = torch.stack(
        [
            uu,
            vv,
            vv - uu,
            torch.abs(vv - uu),
        ],
        dim=-1,
    )

    # [Ls, Lt, 4 transition types, 4 geometry features]
    features = base[
        :,
        :,
        None,
        :
    ].expand(
        ls,
        lt,
        NUM_TRANSITIONS,
        4,
    ).clone()

    move_onehot = torch.eye(
        NUM_TRANSITIONS,
        device=device,
        dtype=geometry_dtype,
    )

    move_onehot = (
        move_onehot[
            None,
            None,
            :,
            :
        ]
        .expand(
            ls,
            lt,
            NUM_TRANSITIONS,
            NUM_TRANSITIONS,
        )
    )

    features = torch.cat(
        [
            features,
            move_onehot,
        ],
        dim=-1,
    )

    # Only now cast to the requested dtype.
    features = features.to(
        dtype=dtype
    )

    if features.shape != (
        ls,
        lt,
        NUM_TRANSITIONS,
        LOCAL_FEATURE_DIM,
    ):
        raise RuntimeError(
            "Unexpected transition feature shape."
        )

    return features


# ============================================================
# FROZEN SEMANTIC BANK
# ============================================================

class FrozenSemanticBank(nn.Module):
    """
    Load all class-specific LocalSemanticEnergy models
    from one source semantic package.

    All semantic models remain frozen during adaptation.
    """

    def __init__(
        self,
        package_path,
    ):
        super().__init__()

        package_path = Path(
            package_path
        )

        if not package_path.exists():
            raise FileNotFoundError(
                package_path
            )

        package = torch.load(
            package_path,
            map_location="cpu",
            weights_only=False,
        )

        self.package_path = str(
            package_path
        )

        self.version = package[
            "version"
        ]

        self.dataset = package[
            "dataset"
        ]

        self.source_domain = str(
            package["source_domain"]
        )

        self.num_classes = int(
            package["num_classes"]
        )

        geometry = package[
            "semantic_geometry"
        ]

        self.semantic_temporal_length = int(
            geometry[
                "semantic_temporal_length"
            ]
        )

        self.semantic_channels = int(
            geometry[
                "semantic_channels"
            ]
        )

        self.models = nn.ModuleList()

        kappas = []
        score_scales = []

        for class_id in range(
            self.num_classes
        ):

            info = package[
                "classes"
            ][class_id]

            model = LocalSemanticEnergy()

            model.load_state_dict(
                info["state_dict"],
                strict=True,
            )

            model.eval()

            for parameter in (
                model.parameters()
            ):
                parameter.requires_grad_(
                    False
                )

            self.models.append(
                model
            )

            kappas.append(
                float(
                    info["kappa"]
                )
            )

            score_scales.append(
                float(
                    info["score_scale"]
                )
            )

        self.register_buffer(
            "kappas",
            torch.tensor(
                kappas,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "score_scales",
            torch.tensor(
                score_scales,
                dtype=torch.float32,
            ),
        )

        # Keep only lightweight metadata needed for checks.
        self.local_feature_order = list(
            package[
                "semantic_geometry"
            ][
                "local_feature_order"
            ]
        )

        self.move_convention = dict(
            package[
                "semantic_geometry"
            ][
                "move_convention"
            ]
        )


    def train(
        self,
        mode=True,
    ):
        """
        Semantic teacher is permanently frozen/eval.
        """

        super().train(False)

        for model in self.models:
            model.eval()

        return self


    def class_kappa(
        self,
        class_id,
    ):
        class_id = int(
            class_id
        )

        return self.kappas[
            class_id
        ]


    def class_score_scale(
        self,
        class_id,
    ):
        class_id = int(
            class_id
        )

        return self.score_scales[
            class_id
        ]


    @torch.no_grad()
    def raw_transition_scores(
        self,
        class_id,
        source_length,
        target_length,
        device=None,
        dtype=torch.float32,
    ):
        """
        Compute g_c(r_ijm).

        Returns
        -------
        [Ls, Lt, 4]
        """

        class_id = int(
            class_id
        )

        if (
            class_id < 0
            or
            class_id >= self.num_classes
        ):
            raise ValueError(
                f"Invalid class_id={class_id}."
            )

        model = self.models[
            class_id
        ]

        if device is None:
            device = self.kappas.device

        feature_grid = (
            build_transition_feature_grid(
                source_length=
                    source_length,

                target_length=
                    target_length,

                device=device,
                dtype=dtype,
            )
        )

        original_shape = (
            feature_grid.shape[:-1]
        )

        flat_features = (
            feature_grid.reshape(
                -1,
                LOCAL_FEATURE_DIM,
            )
        )

        raw = model.local_score(
            flat_features
        )

        raw = raw.reshape(
            *original_shape
        )

        return raw


    @torch.no_grad()
    def normalized_transition_scores(
        self,
        class_id,
        source_length,
        target_length,
        device=None,
        dtype=torch.float32,
    ):
        """
        Compute:

            g_c(r_ijm) / rho_c
        """

        raw = self.raw_transition_scores(
            class_id=class_id,
            source_length=source_length,
            target_length=target_length,
            device=device,
            dtype=dtype,
        )

        scale = self.class_score_scale(
            class_id
        ).to(
            device=raw.device,
            dtype=raw.dtype,
        )

        if scale.item() <= 0.0:
            raise RuntimeError(
                "Semantic score scale must "
                "be positive."
            )

        return (
            raw
            /
            scale
        )


    @torch.no_grad()
    def semantic_bonus(
        self,
        class_id,
        source_length,
        target_length,
        lambda_sem=1.0,
        device=None,
        dtype=torch.float32,
    ):
        """
        Compute final ACTA semantic transition bonus:

            lambda_sem
            * kappa_c
            * g_c(r_ijm) / rho_c

        Returns
        -------
        [Ls, Lt, 4]
        """

        normalized = (
            self.normalized_transition_scores(
                class_id=class_id,
                source_length=source_length,
                target_length=target_length,
                device=device,
                dtype=dtype,
            )
        )

        kappa = self.class_kappa(
            class_id
        ).to(
            device=normalized.device,
            dtype=normalized.dtype,
        )

        return (
            float(lambda_sem)
            *
            kappa
            *
            normalized
        )