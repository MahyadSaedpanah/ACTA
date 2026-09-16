import copy

import torch
import torch.nn as nn

from algorithms.algorithms_base import Algorithm
from utils.module import *


class TemporalWarpBank(nn.Module):
    """
    Fixed temporal action bank for ACTA.

    13 actions total:
      - 1 identity
      - 12 smooth, monotone, endpoint-preserving temporal warps

    Families: quadratic, sine1, sine2
    Severities: |a| in {0.25, 0.50}
    Directions: negative / positive
    """

    def __init__(self):
        super().__init__()

        specs = [("identity", "identity", 0.0)]

        for family in ("quadratic", "sine1", "sine2"):
            for severity in (0.25, 0.50):
                specs.append(
                    (f"{family}_a{severity:.2f}_neg", family, -severity)
                )
                specs.append(
                    (f"{family}_a{severity:.2f}_pos", family, +severity)
                )

        self.specs = specs
        self.names = [name for name, _, _ in specs]
        self.num_warps = len(specs)

    @staticmethod
    def _mapping(u, family, amplitude):
        if family == "identity":
            return u

        if family == "quadratic":
            return u + amplitude * u * (1.0 - u)

        if family == "sine1":
            return (
                u
                + amplitude
                * torch.sin(2.0 * torch.pi * u)
                / (2.0 * torch.pi)
            )

        if family == "sine2":
            return (
                u
                + amplitude
                * torch.sin(4.0 * torch.pi * u)
                / (4.0 * torch.pi)
            )

        raise ValueError(f"Unknown temporal warp family: {family}")

    def normalized_mappings(self, length, device, dtype):
        if int(length) < 2:
            raise ValueError("Temporal length must be >= 2.")

        u = torch.linspace(
            0.0,
            1.0,
            steps=int(length),
            device=device,
            dtype=dtype,
        )

        mappings = [
            self._mapping(u, family, amplitude).clamp(0.0, 1.0)
            for _, family, amplitude in self.specs
        ]

        return torch.stack(mappings, dim=0)

    def validate(self, length=257):
        mappings = self.normalized_mappings(
            length=length,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        endpoint_error = max(
            mappings[:, 0].abs().max().item(),
            (mappings[:, -1] - 1.0).abs().max().item(),
        )

        min_step = (
            mappings[:, 1:]
            - mappings[:, :-1]
        ).min().item()

        identity_reference = torch.linspace(
            0.0,
            1.0,
            steps=int(length),
            dtype=torch.float64,
        )

        identity_error = (
            mappings[0] - identity_reference
        ).abs().max().item()

        if endpoint_error > 1e-10:
            raise RuntimeError("Warp endpoints are invalid.")

        if min_step <= 0.0:
            raise RuntimeError("Warp bank is not strictly monotone.")

        if identity_error > 1e-10:
            raise RuntimeError("Identity warp is invalid.")

        return {
            "num_warps": self.num_warps,
            "endpoint_error": endpoint_error,
            "min_step": min_step,
            "identity_error": identity_error,
        }

    def warp_all(self, x):
        """
        Apply all temporal warps.

        x:
            [B, C, T]

        returns:
            [B, M, C, T]
        """
        if x.ndim != 3:
            raise ValueError("Expected x with shape [B, C, T].")

        B, C, T = x.shape

        mappings = self.normalized_mappings(
            length=T,
            device=x.device,
            dtype=x.dtype,
        )

        grid_x = 2.0 * mappings - 1.0

        x_rep = (
            x[:, None, :, None, :]
            .expand(B, self.num_warps, C, 1, T)
            .reshape(B * self.num_warps, C, 1, T)
        )

        gx = (
            grid_x[None, :, None, :]
            .expand(B, self.num_warps, 1, T)
            .reshape(B * self.num_warps, 1, T)
        )

        gy = torch.zeros_like(gx)
        grid = torch.stack([gx, gy], dim=-1)

        warped = F.grid_sample(
            x_rep,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return (
            warped
            .squeeze(2)
            .reshape(B, self.num_warps, C, T)
        )

    def mix(self, x, alpha):
        """
        Build one differentiable sample-specific convex warp mixture.

        x:
            [B, C, T]

        alpha:
            [B, M], non-negative and each row sums to 1

        returns:
            corrected x with shape [B, C, T]
        """
        if x.ndim != 3:
            raise ValueError("Expected x with shape [B, C, T].")

        if alpha.ndim != 2:
            raise ValueError("Expected alpha with shape [B, M].")

        if alpha.shape[0] != x.shape[0]:
            raise ValueError("Batch mismatch between x and alpha.")

        if alpha.shape[1] != self.num_warps:
            raise ValueError("Warp-count mismatch in alpha.")

        if torch.any(alpha < -1e-7):
            raise ValueError("Warp weights must be non-negative.")

        row_sum = alpha.sum(dim=1, keepdim=True)

        if not torch.allclose(
            row_sum,
            torch.ones_like(row_sum),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("Warp weights must sum to one.")

        B, C, T = x.shape

        mappings = self.normalized_mappings(
            length=T,
            device=x.device,
            dtype=x.dtype,
        )

        mixed_mapping = alpha @ mappings

        gx = (2.0 * mixed_mapping - 1.0).unsqueeze(1)
        gy = torch.zeros_like(gx)
        grid = torch.stack([gx, gy], dim=-1)

        corrected = F.grid_sample(
            x.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return corrected.squeeze(2)


class ACTA(Algorithm):
    """
    ACTA - clean implementation skeleton.

    Stage 1 only:
        - benchmark CNN backbone
        - benchmark temporal classifier
        - source supervised task update
        - EMA teacher

    No TSA gate, selector, warp bank, discriminator, or DCG is added yet.
    Those will be integrated one-by-one after this skeleton is verified.
    """

    def __init__(self, configs, device, args):
        super(ACTA, self).__init__(configs)

        self.args = args
        self.device = device

        # Fixed temporal warp bank
        self.warp_bank = TemporalWarpBank()
        self.warp_bank_stats = self.warp_bank.validate()

        # Benchmark task model
        self.t_feature_extractor = CNN(configs)
        self.t_classifier = TemporalClassifierHead(
            self.t_feature_extractor.out_dim,
            configs.num_classes,
        )

        # EMA teacher
        self.ema_decay = float(getattr(args, "acta_ema", 0.99))

        self.ema_feature_extractor = copy.deepcopy(
            self.t_feature_extractor
        )
        self.ema_classifier = copy.deepcopy(
            self.t_classifier
        )
        self._freeze_ema()

        # Task optimizer
        self.optimizer_task = torch.optim.Adam(
            [
                {"params": self.t_feature_extractor.parameters()},
                {"params": self.t_classifier.parameters()},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def _freeze_ema(self):
        self.ema_feature_extractor.eval()
        self.ema_classifier.eval()

        for module in (
            self.ema_feature_extractor,
            self.ema_classifier,
        ):
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        d = self.ema_decay

        for ema_param, param in zip(
            self.ema_feature_extractor.parameters(),
            self.t_feature_extractor.parameters(),
        ):
            ema_param.mul_(d).add_(param, alpha=1.0 - d)

        for ema_param, param in zip(
            self.ema_classifier.parameters(),
            self.t_classifier.parameters(),
        ):
            ema_param.mul_(d).add_(param, alpha=1.0 - d)

        for ema_buffer, buffer in zip(
            self.ema_feature_extractor.buffers(),
            self.t_feature_extractor.buffers(),
        ):
            ema_buffer.copy_(buffer)

        for ema_buffer, buffer in zip(
            self.ema_classifier.buffers(),
            self.t_classifier.buffers(),
        ):
            ema_buffer.copy_(buffer)

        self._freeze_ema()

    def encode(self, x):
        return self.t_feature_extractor(x)

    def classify(self, z):
        return self.t_classifier(z)

    @torch.no_grad()
    def ema_predict(self, x):
        self.ema_feature_extractor.eval()
        self.ema_classifier.eval()

        feat = self.ema_feature_extractor(x)
        logits = self.ema_classifier(feat)

        return torch.softmax(logits, dim=-1)

    def update(self, src_x, src_y, trg_x):
        """
        Stage 1 intentionally performs only the source supervised
        task update.

        trg_x is accepted to preserve the benchmark DA trainer API,
        but it is not used yet.
        """

        self.t_feature_extractor.train()
        self.t_classifier.train()

        src_feat = self.t_feature_extractor(src_x)
        src_pred = self.t_classifier(src_feat)

        source_loss = self.cross_entropy(
            src_pred.squeeze(),
            src_y,
        )

        self.optimizer_task.zero_grad()
        source_loss.backward()
        self.optimizer_task.step()

        self.update_ema()

        return {
            "Source_loss": float(source_loss.detach().item()),
            "Warp_count": float(self.warp_bank.num_warps),
        }

    def predict(self, data):
        self.t_feature_extractor.eval()
        self.t_classifier.eval()

        with torch.no_grad():
            feat = self.t_feature_extractor(data)
            pred = self.t_classifier(feat)

        return pred

    def save_model(self, path):
        torch.save(
            {
                "t_encoder": self.t_feature_extractor.state_dict(),
                "t_classifier": self.t_classifier.state_dict(),
                "ema_encoder": self.ema_feature_extractor.state_dict(),
                "ema_classifier": self.ema_classifier.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        self.t_feature_extractor.load_state_dict(
            checkpoint["t_encoder"]
        )
        self.t_classifier.load_state_dict(
            checkpoint["t_classifier"]
        )

        if "ema_encoder" in checkpoint:
            self.ema_feature_extractor.load_state_dict(
                checkpoint["ema_encoder"]
            )
        else:
            self.ema_feature_extractor.load_state_dict(
                checkpoint["t_encoder"]
            )

        if "ema_classifier" in checkpoint:
            self.ema_classifier.load_state_dict(
                checkpoint["ema_classifier"]
            )
        else:
            self.ema_classifier.load_state_dict(
                checkpoint["t_classifier"]
            )

        self._freeze_ema()
        return self