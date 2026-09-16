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

        # Source-only TSA codebook buffers.
        self.register_buffer(
            "tsa_mean_excess_risk",
            torch.zeros(
                configs.num_classes,
                self.warp_bank.num_warps,
            ),
        )
        self.register_buffer(
            "tsa_rank_score",
            torch.ones(
                configs.num_classes,
                self.warp_bank.num_warps,
            ),
        )
        self.register_buffer(
            "tsa_teacher_accuracy",
            torch.zeros(configs.num_classes),
        )
        self.register_buffer(
            "tsa_order_stability",
            torch.zeros(configs.num_classes),
        )
        self.register_buffer(
            "tsa_kappa",
            torch.zeros(configs.num_classes),
        )
        self.register_buffer(
            "tsa_ready",
            torch.tensor(False),
        )

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

    @staticmethod
    def _positive_correlation(a, b, eps=1e-8):
        """
        Positive Pearson correlation in [0,1].

        Used only as a source-side reliability measure for whether
        two independent source subsets agree on the relative warp
        geometry of a class.
        """
        a = a - a.mean()
        b = b - b.mean()

        denom = (
            torch.sqrt((a * a).sum())
            *
            torch.sqrt((b * b).sum())
        )

        if float(denom.item()) <= eps:
            return a.new_tensor(0.0)

        corr = (a * b).sum() / denom
        return corr.clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def build_tsa_codebook(self, source_loader):
        """
        Build a SOURCE-ONLY Temporal Semantic Admissibility codebook.

        For class c and warp phi_m:

            Delta[c,m]
              =
              E[
                CE(f(T_phi_m(x)), c)
                -
                CE(f(T_identity(x)), c)
                | y=c
              ]

        Lower Delta means the warp better preserves the source task
        semantics of class c.

        We do NOT interpret these values as calibrated probabilities.
        Instead, the 12 non-identity warps are ranked within each
        class and converted to relative admissibility scores in [0,1].

        Reliability is kept separate:

            teacher_accuracy[c]
                = source task-model accuracy for class c

            order_stability[c]
                = agreement of warp excess-risk geometry between
                  two deterministic source subsets

            kappa[c]
                = teacher_accuracy[c] * order_stability[c]

        No target data is used.
        """

        self.t_feature_extractor.eval()
        self.t_classifier.eval()

        C = int(self.configs.num_classes)
        M = int(self.warp_bank.num_warps)

        # Two source subsets are accumulated independently so that
        # class-specific warp ordering reliability can be measured.
        split_sum = torch.zeros(
            2, C, M,
            device=self.device,
        )
        split_count = torch.zeros(
            2, C,
            device=self.device,
        )

        class_count = torch.zeros(
            C,
            device=self.device,
        )
        correct_count = torch.zeros(
            C,
            device=self.device,
        )

        global_index = 0

        for source_x, source_y in source_loader:
            source_x = source_x.float().to(self.device)
            source_y = source_y.long().to(self.device)

            B = int(source_x.shape[0])

            # Unwarped source prediction only measures teacher quality.
            base_feat = self.t_feature_extractor(source_x)
            base_logits = self.t_classifier(base_feat)
            base_pred = base_logits.argmax(dim=1)

            # Evaluate all temporal actions with the same frozen EMA teacher.
            warped = self.warp_bank.warp_all(source_x)

            flat_warped = warped.reshape(
                B * M,
                source_x.shape[1],
                source_x.shape[2],
            )

            warped_feat = self.t_feature_extractor(flat_warped)
            warped_logits = self.t_classifier(warped_feat)

            repeated_y = (
                source_y[:, None]
                .expand(B, M)
                .reshape(-1)
            )

            warped_ce = F.cross_entropy(
                warped_logits,
                repeated_y,
                reduction="none",
            ).reshape(B, M)

            # Identity is action 0, giving an exact matched baseline.
            excess = (
                warped_ce
                -
                warped_ce[:, 0:1]
            )

            # Deterministic alternating split for source-only stability.
            sample_ids = torch.arange(
                global_index,
                global_index + B,
                device=self.device,
            )
            subset_id = sample_ids.remainder(2)
            global_index += B

            for c in range(C):
                class_mask = source_y == c

                if not class_mask.any():
                    continue

                class_count[c] += class_mask.sum()
                correct_count[c] += (
                    base_pred[class_mask] == c
                ).sum()

                for s in (0, 1):
                    mask = class_mask & (subset_id == s)

                    if mask.any():
                        split_sum[s, c] += excess[mask].sum(dim=0)
                        split_count[s, c] += mask.sum()

        if torch.any(class_count <= 0):
            missing = torch.where(class_count <= 0)[0].tolist()
            raise RuntimeError(
                "TSA codebook cannot be built; missing source "
                f"classes: {missing}"
            )

        total_sum = split_sum.sum(dim=0)
        mean_excess = total_sum / class_count[:, None]
        mean_excess[:, 0] = 0.0

        # Relative admissibility ranks.
        rank_score = torch.ones_like(mean_excess)
        nonidentity = mean_excess[:, 1:]
        n_nonidentity = int(nonidentity.shape[1])

        for c in range(C):
            order = torch.argsort(
                nonidentity[c],
                descending=False,
            )

            ordered_scores = torch.linspace(
                1.0,
                0.0,
                steps=n_nonidentity,
                device=self.device,
                dtype=mean_excess.dtype,
            )

            class_score = torch.empty_like(nonidentity[c])
            class_score[order] = ordered_scores
            rank_score[c, 1:] = class_score

        # Identity is always the safe no-correction action.
        rank_score[:, 0] = 1.0

        teacher_accuracy = (
            correct_count / class_count
        ).clamp(0.0, 1.0)

        order_stability = torch.zeros(
            C,
            device=self.device,
        )

        for c in range(C):
            if (
                split_count[0, c] <= 0
                or
                split_count[1, c] <= 0
            ):
                continue

            split0 = (
                split_sum[0, c, 1:]
                /
                split_count[0, c]
            )
            split1 = (
                split_sum[1, c, 1:]
                /
                split_count[1, c]
            )

            order_stability[c] = self._positive_correlation(
                split0,
                split1,
            )

        kappa = (
            teacher_accuracy
            *
            order_stability
        ).clamp(0.0, 1.0)

        self.tsa_mean_excess_risk.copy_(mean_excess)
        self.tsa_rank_score.copy_(rank_score)
        self.tsa_teacher_accuracy.copy_(teacher_accuracy)
        self.tsa_order_stability.copy_(order_stability)
        self.tsa_kappa.copy_(kappa)
        self.tsa_ready.fill_(True)

        return {
            "ready": True,
            "num_classes": C,
            "num_warps": M,
            "teacher_acc_mean": float(
                teacher_accuracy.mean().item()
            ),
            "order_stability_mean": float(
                order_stability.mean().item()
            ),
            "kappa_mean": float(
                kappa.mean().item()
            ),
            "mean_abs_excess": float(
                mean_excess[:, 1:].abs().mean().item()
            ),
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
                "tsa_mean_excess_risk": self.tsa_mean_excess_risk.detach().cpu(),
                "tsa_rank_score": self.tsa_rank_score.detach().cpu(),
                "tsa_teacher_accuracy": self.tsa_teacher_accuracy.detach().cpu(),
                "tsa_order_stability": self.tsa_order_stability.detach().cpu(),
                "tsa_kappa": self.tsa_kappa.detach().cpu(),
                "tsa_ready": bool(self.tsa_ready.item()),
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

        if "tsa_mean_excess_risk" in checkpoint:
            self.tsa_mean_excess_risk.copy_(
                checkpoint["tsa_mean_excess_risk"].to(self.device)
            )

        if "tsa_rank_score" in checkpoint:
            self.tsa_rank_score.copy_(
                checkpoint["tsa_rank_score"].to(self.device)
            )

        if "tsa_teacher_accuracy" in checkpoint:
            self.tsa_teacher_accuracy.copy_(
                checkpoint["tsa_teacher_accuracy"].to(self.device)
            )

        if "tsa_order_stability" in checkpoint:
            self.tsa_order_stability.copy_(
                checkpoint["tsa_order_stability"].to(self.device)
            )

        if "tsa_kappa" in checkpoint:
            self.tsa_kappa.copy_(
                checkpoint["tsa_kappa"].to(self.device)
            )

        if bool(checkpoint.get("tsa_ready", False)):
            self.tsa_ready.fill_(True)

        self._freeze_ema()
        return self