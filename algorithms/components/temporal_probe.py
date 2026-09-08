"""
ACTA-v2 temporal probe bank.

This module operationalizes Temporal Semantic Admissibility (TSA) as a
semantic-preservation certificate rather than as a source-target attraction
path.

It loads a source-only temporal_probe_codebook.pt and provides:

1. a fixed bank of smooth temporal warps,
2. class-conditioned source-semantic probe weights,
3. vectorized target warping,
4. reliability-preserving TAC weighting,
5. Jensen-Shannon consistency utilities.

IMPORTANT
---------
The codebook stores

    A[c,m] = kappa_c * sigmoid(S_c(phi_m))

where sigmoid(S) is only a bounded monotone transform of the distilled
semantic path logit; it is not assumed calibrated.

For a target sample with detached EMA class probabilities p:

    w_m(x) = sum_c p(c|x) A[c,m].

TAC deliberately uses

    (1/M) * sum_m w_m * JS(...)

NOT normalization by sum_m w_m.

Why?
-----
Normalizing by sum_m w_m would almost cancel the class reliability kappa_c
for confident samples. The unnormalized mean preserves the intended source
semantic reliability: low-kappa classes exert less preservation pressure.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CODEBOOK_VERSION = "ACTA_TEMPORAL_PROBE_CODEBOOK_V1"


class TemporalProbeBank(nn.Module):
    """
    Frozen source-only ACTA-v2 temporal probe bank.
    """

    def __init__(
        self,
        codebook_path,
    ):
        super().__init__()

        codebook_path = Path(
            codebook_path
        )

        if not codebook_path.exists():
            raise FileNotFoundError(
                codebook_path
            )

        package = torch.load(
            codebook_path,
            map_location="cpu",
            weights_only=False,
        )

        version = str(
            package["version"]
        )

        if (
            version
            !=
            SUPPORTED_CODEBOOK_VERSION
        ):
            raise RuntimeError(
                "Unsupported temporal probe codebook "
                f"version: {version}"
            )

        if bool(
            package.get(
                "target_information_used",
                True,
            )
        ):
            raise RuntimeError(
                "Temporal probe codebook must be "
                "source-only."
            )

        self.codebook_path = str(
            codebook_path
        )

        self.version = version

        self.dataset = str(
            package["dataset"]
        )

        self.source_domain = str(
            package["source_domain"]
        )

        self.num_classes = int(
            package["num_classes"]
        )

        self.num_probes = int(
            package["num_probes"]
        )

        self.semantic_temporal_length = int(
            package[
                "semantic_temporal_length"
            ]
        )

        probes = package[
            "probes"
        ]

        if len(probes) != self.num_probes:
            raise RuntimeError(
                "Probe-count mismatch in codebook."
            )

        mappings = torch.stack(
            [
                torch.as_tensor(
                    p[
                        "normalized_mapping"
                    ],
                    dtype=torch.float32,
                )
                for p in probes
            ],
            dim=0,
        )

        if mappings.shape != (
            self.num_probes,
            self.semantic_temporal_length,
        ):
            raise RuntimeError(
                "Unexpected codebook mapping shape: "
                f"{tuple(mappings.shape)}"
            )

        if not torch.allclose(
            mappings[:, 0],
            torch.zeros(
                self.num_probes,
                dtype=mappings.dtype,
            ),
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Every temporal probe must start at 0."
            )

        if not torch.allclose(
            mappings[:, -1],
            torch.ones(
                self.num_probes,
                dtype=mappings.dtype,
            ),
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Every temporal probe must end at 1."
            )

        if torch.any(
            mappings[:, 1:]
            <
            mappings[:, :-1]
        ):
            raise RuntimeError(
                "Temporal probe mappings must be monotone."
            )

        class_weights = torch.as_tensor(
            package[
                "class_reliability_weighted_scores"
            ],
            dtype=torch.float32,
        )

        if class_weights.shape != (
            self.num_classes,
            self.num_probes,
        ):
            raise RuntimeError(
                "Unexpected class/probe weight shape: "
                f"{tuple(class_weights.shape)}"
            )

        if (
            class_weights.min().item()
            < -1e-6
            or
            class_weights.max().item()
            >
            1.0 + 1e-6
        ):
            raise RuntimeError(
                "Expected bounded class-probe weights "
                "inside [0,1]."
            )

        self.register_buffer(
            "normalized_mappings",
            mappings,
        )

        self.register_buffer(
            "class_probe_weights",
            class_weights.clamp(
                min=0.0,
                max=1.0,
            ),
        )

        self.register_buffer(
            "class_sigmoid_scores",
            torch.as_tensor(
                package[
                    "class_sigmoid_scores"
                ],
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "class_kappas",
            torch.as_tensor(
                package[
                    "class_kappas"
                ],
                dtype=torch.float32,
            ),
        )

        self.probe_names = [
            str(
                p["name"]
            )
            for p in probes
        ]


    def train(
        self,
        mode=True,
    ):
        """
        Probe bank is permanently frozen/eval.
        """

        super().train(False)
        return self


    @torch.no_grad()
    def expected_probe_weights(
        self,
        target_probabilities,
    ):
        """
        Compute source-semantic probe weights for target samples.

            w[b,m]
                =
                sum_c p[b,c] A[c,m]

        Parameters
        ----------
        target_probabilities:
            [B,C], expected to be detached EMA probabilities.

        Returns
        -------
        [B,M]
        """

        if target_probabilities.ndim != 2:
            raise ValueError(
                "target_probabilities must have "
                "shape [B,C]."
            )

        if target_probabilities.shape[1] != self.num_classes:
            raise ValueError(
                "Target probability class dimension "
                "does not match codebook."
            )

        probabilities = (
            target_probabilities
            .detach()
            .to(
                device=
                    self.class_probe_weights
                    .device,

                dtype=
                    self.class_probe_weights
                    .dtype,
            )
        )

        if not torch.isfinite(
            probabilities
        ).all():
            raise RuntimeError(
                "Non-finite target probabilities."
            )

        sums = probabilities.sum(
            dim=1
        )

        if not torch.allclose(
            sums,
            torch.ones_like(
                sums
            ),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "Target probabilities must sum to 1."
            )

        weights = (
            probabilities
            @
            self.class_probe_weights
        )

        return weights.clamp(
            min=0.0,
            max=1.0,
        )


    def _resampled_normalized_grid(
        self,
        raw_length,
        device,
        dtype,
    ):
        """
        Convert codebook's normalized mapping [M,G]
        into a sampling grid [M,T] for arbitrary raw T.

        The mapping itself is normalized to [0,1], so linear
        interpolation across its stored G points preserves the
        same continuous temporal warp at other sequence lengths.
        """

        raw_length = int(
            raw_length
        )

        if raw_length <= 1:
            raise ValueError(
                "Temporal probe requires sequence "
                "length >= 2."
            )

        mappings = (
            self.normalized_mappings
            .to(
                device=device,
                dtype=dtype,
            )
        )

        if (
            mappings.shape[-1]
            !=
            raw_length
        ):

            mappings = F.interpolate(
                mappings.unsqueeze(1),
                size=raw_length,
                mode="linear",
                align_corners=True,
            ).squeeze(1)

        # grid_sample uses [-1,1].
        x_grid = (
            2.0
            *
            mappings
            -
            1.0
        )

        return x_grid.clamp(
            min=-1.0,
            max=1.0,
        )


    def warp_all(
        self,
        x,
    ):
        """
        Apply every temporal probe to a target batch.

        Parameters
        ----------
        x:
            [B,C,T]

        Returns
        -------
        [B,M,C,T]

        Convention:
            output at target coordinate v samples the original
            sequence at source coordinate phi(v).

        grid_sample with align_corners=True preserves the same
        normalized endpoints used throughout ACTA's geometry.
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape [B,C,T]."
            )

        batch_size = int(
            x.shape[0]
        )

        channels = int(
            x.shape[1]
        )

        length = int(
            x.shape[2]
        )

        grid_x = (
            self._resampled_normalized_grid(
                raw_length=length,
                device=x.device,
                dtype=x.dtype,
            )
        )

        # Input for 2-D grid_sample:
        # [B,C,H=1,W=T]
        x_2d = x.unsqueeze(
            2
        )

        # Replicate each sample across M probes:
        # [B,M,C,1,T] -> [B*M,C,1,T]
        x_rep = (
            x_2d[
                :,
                None,
                :,
                :,
                :
            ]
            .expand(
                batch_size,
                self.num_probes,
                channels,
                1,
                length,
            )
            .reshape(
                batch_size
                *
                self.num_probes,

                channels,
                1,
                length,
            )
        )

        # grid_sample expects [N,H_out,W_out,2].
        #
        # x-coordinate = temporal sample position.
        # y-coordinate = 0 because H=1.
        gx = (
            grid_x[
                None,
                :,
                None,
                :
            ]
            .expand(
                batch_size,
                self.num_probes,
                1,
                length,
            )
            .reshape(
                batch_size
                *
                self.num_probes,

                1,
                length,
            )
        )

        gy = torch.zeros_like(
            gx
        )

        grid = torch.stack(
            [
                gx,
                gy,
            ],
            dim=-1,
        )

        warped = F.grid_sample(
            x_rep,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # [B*M,C,1,T] -> [B,M,C,T]
        warped = (
            warped
            .squeeze(2)
            .reshape(
                batch_size,
                self.num_probes,
                channels,
                length,
            )
        )

        return warped


    def flatten_warped(
        self,
        warped,
    ):
        """
        [B,M,C,T] -> [B*M,C,T]
        """

        if warped.ndim != 4:
            raise ValueError(
                "warped must have shape [B,M,C,T]."
            )

        if warped.shape[1] != self.num_probes:
            raise ValueError(
                "Probe dimension mismatch."
            )

        return warped.reshape(
            warped.shape[0]
            *
            warped.shape[1],

            warped.shape[2],
            warped.shape[3],
        )


    def unflatten_probe_logits(
        self,
        logits,
        batch_size,
    ):
        """
        [B*M,C] -> [B,M,C]
        """

        batch_size = int(
            batch_size
        )

        if logits.ndim != 2:
            raise ValueError(
                "Expected logits with shape [B*M,C]."
            )

        if logits.shape[0] != (
            batch_size
            *
            self.num_probes
        ):
            raise ValueError(
                "Probe-logit batch dimension mismatch."
            )

        return logits.reshape(
            batch_size,
            self.num_probes,
            logits.shape[-1],
        )


def jensen_shannon_per_probe(
    reference_probabilities,
    probe_logits,
    eps=1e-8,
):
    """
    Jensen-Shannon divergence between:

        detached reference p(x)       [B,C]

    and

        student probe predictions
        q(T_phi_m x)                  [B,M,C]

    Returns
    -------
    [B,M]

    Gradients flow only through probe_logits.
    """

    if reference_probabilities.ndim != 2:
        raise ValueError(
            "reference_probabilities must "
            "have shape [B,C]."
        )

    if probe_logits.ndim != 3:
        raise ValueError(
            "probe_logits must have shape "
            "[B,M,C]."
        )

    if (
        probe_logits.shape[0]
        !=
        reference_probabilities.shape[0]
        or
        probe_logits.shape[2]
        !=
        reference_probabilities.shape[1]
    ):
        raise ValueError(
            "Reference/probe prediction shape mismatch."
        )

    p = (
        reference_probabilities
        .detach()
        .to(
            device=probe_logits.device,
            dtype=probe_logits.dtype,
        )
        .clamp_min(
            eps
        )
    )

    p = (
        p
        /
        p.sum(
            dim=-1,
            keepdim=True,
        )
    )

    q = F.softmax(
        probe_logits,
        dim=-1,
    ).clamp_min(
        eps
    )

    q = (
        q
        /
        q.sum(
            dim=-1,
            keepdim=True,
        )
    )

    p_expand = p[
        :,
        None,
        :
    ].expand_as(
        q
    )

    midpoint = (
        0.5
        *
        (
            p_expand
            +
            q
        )
    ).clamp_min(
        eps
    )

    kl_p_m = (
        p_expand
        *
        (
            torch.log(
                p_expand
            )
            -
            torch.log(
                midpoint
            )
        )
    ).sum(
        dim=-1
    )

    kl_q_m = (
        q
        *
        (
            torch.log(
                q
            )
            -
            torch.log(
                midpoint
            )
        )
    ).sum(
        dim=-1
    )

    return (
        0.5
        *
        (
            kl_p_m
            +
            kl_q_m
        )
    )


def temporal_admissibility_consistency_loss(
    reference_probabilities,
    probe_logits,
    probe_weights,
):
    """
    ACTA-v2 TAC loss.

        L_TAC
          =
          mean_b [
              (1/M)
              sum_m
                  w_bm
                  JS(
                      p_bar(x_b),
                      p_theta(T_phi_m x_b)
                  )
          ]

    Reliability is intentionally NOT normalized away.
    """

    js = jensen_shannon_per_probe(
        reference_probabilities=
            reference_probabilities,

        probe_logits=
            probe_logits,
    )

    if probe_weights.shape != js.shape:
        raise ValueError(
            "probe_weights must match JS shape "
            f"{tuple(js.shape)}, "
            f"got {tuple(probe_weights.shape)}."
        )

    weights = (
        probe_weights
        .detach()
        .to(
            device=js.device,
            dtype=js.dtype,
        )
    )

    weighted_per_target = (
        weights
        *
        js
    ).mean(
        dim=1
    )

    loss = weighted_per_target.mean()

    return (
        loss,
        {
            "js_per_probe":
                js,

            "probe_weights":
                weights,

            "weighted_per_target":
                weighted_per_target,

            "mean_probe_weight":
                weights.mean(),

            "mean_js":
                js.mean(),
        },
    )