"""
Local additive temporal semantic admissibility model for ACTA.

For a temporal path pi of source length L_s:

    S_c(pi)
        =
        b_c
        +
        (1 / L_s)
        * sum_{e in pi} g_c(e)

where each edge is represented by the fixed 8-D ACTA
local geometry:

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

IMPORTANT
---------
The normalization is by SOURCE LENGTH, not path length.

The final scalar of g_c is linear.
Tanh is used only in the two hidden layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


LOCAL_FEATURE_DIM = 8
DEFAULT_HIDDEN_DIM = 32


class LocalSemanticEnergy(nn.Module):
    """
    Class-specific additive temporal semantic model.

    Architecture:

        8 -> 32 -> 32 -> 1

    with Tanh activations in the hidden layers.
    """

    def __init__(
        self,
        input_dim=LOCAL_FEATURE_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.local_net = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                self.hidden_dim,
                self.hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                self.hidden_dim,
                1
            ),
        )

        # Global class-specific path bias.
        self.path_bias = nn.Parameter(
            torch.zeros(1)
        )


    def local_score(
        self,
        local_x,
    ):
        """
        Evaluate g_c(e).

        Parameters
        ----------
        local_x:
            [..., 8]

        Returns
        -------
        Tensor
            [...]
        """

        if local_x.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected local feature dimension "
                f"{self.input_dim}, "
                f"got {local_x.shape[-1]}."
            )

        return (
            self.local_net(local_x)
            .squeeze(-1)
        )


    def forward(
        self,
        local_x,
        mask,
        source_length,
    ):
        """
        Compute additive path logits for a padded batch.

        Parameters
        ----------
        local_x:
            [N, L_max, 8]

        mask:
            [N, L_max]
            1 for real path cells,
            0 for padding.

        source_length:
            [N]
            Original source temporal length.

        Returns
        -------
        Tensor
            Path logits [N].
        """

        if local_x.ndim != 3:
            raise ValueError(
                "local_x must have shape "
                "[N, L, D]."
            )

        if mask.ndim != 2:
            raise ValueError(
                "mask must have shape [N, L]."
            )

        if (
            local_x.shape[0] != mask.shape[0]
            or
            local_x.shape[1] != mask.shape[1]
        ):
            raise ValueError(
                "local_x and mask dimensions "
                "do not match."
            )

        score = self.local_score(
            local_x
        )

        mask = mask.to(
            device=score.device,
            dtype=score.dtype,
        )

        source_length = torch.as_tensor(
            source_length,
            device=score.device,
            dtype=score.dtype,
        )

        if source_length.ndim == 0:
            source_length = (
                source_length
                .expand(score.shape[0])
            )

        source_length = (
            source_length
            .reshape(-1)
        )

        if source_length.shape[0] != score.shape[0]:
            raise ValueError(
                "source_length must contain "
                "one value per path."
            )

        if torch.any(
            source_length <= 0
        ):
            raise ValueError(
                "source_length must be positive."
            )

        path_sum = (
            score
            *
            mask
        ).sum(
            dim=1
        )

        path_logit = (
            self.path_bias
            +
            path_sum
            /
            source_length
        )

        return path_logit


    def score_single_path(
        self,
        local_x,
        source_length,
    ):
        """
        Convenience function for one unpadded path.

        local_x:
            [L_path, 8]
        """

        if local_x.ndim != 2:
            raise ValueError(
                "Single path must have shape "
                "[L_path, D]."
            )

        score = self.local_score(
            local_x
        )

        source_length = torch.as_tensor(
            source_length,
            device=score.device,
            dtype=score.dtype,
        )

        if source_length.numel() != 1:
            raise ValueError(
                "source_length must be scalar "
                "for score_single_path()."
            )

        if source_length.item() <= 0:
            raise ValueError(
                "source_length must be positive."
            )

        return (
            self.path_bias
            +
            score.sum()
            /
            source_length
        ).squeeze()