"""
ACTA: Admissibility-Constrained Temporal Alignment

Stage-B algorithm skeleton.

This file connects:

    benchmark CNN + classifier
    source-only checkpoint theta_S
    EMA task teacher
    frozen source semantic package
    ACTA transition-aware alignment core

The actual multi-reference / class-conditioned update rule is added
in later integration steps.

IMPORTANT
---------
ACTA adaptation always starts from an explicitly prepared
source-only checkpoint.

The optimizer state from source pretraining is NOT reused.
Stage-B adaptation receives a fresh optimizer.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.algorithms_base import Algorithm

from algorithms.components.semantic_teacher import (
    FrozenSemanticBank,
)

from algorithms.components.acta_aligner import (
    ACTAAlignmentCore,
)

from utils.module import (
    CNN,
    TemporalClassifierHead,
)


# ============================================================
# ACTA
# ============================================================

class ACTA(Algorithm):

    def __init__(
        self,
        configs,
        device,
        args,
    ):
        super().__init__(configs)

        self.device_runtime = torch.device(
            device
        )

        self.args = args

        # ----------------------------------------------------
        # Exact benchmark model
        # ----------------------------------------------------

        self.feature_extractor = CNN(
            configs
        )

        self.classifier = (
            TemporalClassifierHead(
                self.feature_extractor.out_dim,
                configs.num_classes,
            )
        )

        # ----------------------------------------------------
        # EMA task teacher
        #
        # It is synchronized again after loading theta_S.
        # ----------------------------------------------------

        self.ema_feature_extractor = (
            copy.deepcopy(
                self.feature_extractor
            )
        )

        self.ema_classifier = (
            copy.deepcopy(
                self.classifier
            )
        )

        self._freeze_ema()

        # ----------------------------------------------------
        # Adaptation hyperparameters
        #
        # Defaults are only implementation defaults.
        # The final experiment protocol will specify them
        # explicitly.
        # ----------------------------------------------------

        self.lr = float(
            getattr(
                args,
                "lr",
                1e-3,
            )
        )

        self.weight_decay = float(
            getattr(
                args,
                "weight_decay",
                1e-4,
            )
        )

        self.beta = float(
            getattr(
                args,
                "acta_beta",
                1.0,
            )
        )

        self.lambda_sem = float(
            getattr(
                args,
                "acta_lambda",
                1.0,
            )
        )

        self.gamma = float(
            getattr(
                args,
                "acta_gamma",
                0.1,
            )
        )

        self.ema_momentum = float(
            getattr(
                args,
                "acta_ema",
                0.99,
            )
        )

        if not (
            0.0
            <= self.ema_momentum
            < 1.0
        ):
            raise ValueError(
                "acta_ema must satisfy "
                "0 <= momentum < 1."
            )

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        self.source_model_root = str(
            getattr(
                args,
                "source_model_root",
                "./source_models",
            )
        )

        self.semantic_root = str(
            getattr(
                args,
                "semantic_root",
                "./semantic_packages",
            )
        )

        # ----------------------------------------------------
        # Filled by configure_source_context()
        # ----------------------------------------------------

        self.dataset_name = None
        self.source_id = None
        self.seed = None

        self.source_checkpoint_path = None
        self.semantic_package_path = None

        self.semantic_bank = None
        self.alignment_core = None

        self.optimizer = None

        self.is_configured = False


    # ========================================================
    # Internal helpers
    # ========================================================

    def _freeze_ema(
        self,
    ):

        self.ema_feature_extractor.eval()
        self.ema_classifier.eval()

        for parameter in (
            self.ema_feature_extractor
            .parameters()
        ):
            parameter.requires_grad_(
                False
            )

        for parameter in (
            self.ema_classifier
            .parameters()
        ):
            parameter.requires_grad_(
                False
            )


    def _sync_ema_from_student(
        self,
    ):
        """
        EMA must initially be exactly theta_S.
        """

        self.ema_feature_extractor.load_state_dict(
            self.feature_extractor.state_dict(),
            strict=True,
        )

        self.ema_classifier.load_state_dict(
            self.classifier.state_dict(),
            strict=True,
        )

        self._freeze_ema()


    @staticmethod
    def _set_batchnorm_eval(
        module,
    ):
        """
        Freeze BN running statistics while leaving
        affine BN parameters trainable.

        Dropout remains in training mode.
        """

        for child in module.modules():

            if isinstance(
                child,
                nn.BatchNorm1d,
            ):
                child.eval()


    def _fresh_optimizer(
        self,
    ):
        """
        Stage-B gets a fresh Adam optimizer.

        Source-pretraining optimizer state is intentionally
        not inherited.
        """

        return torch.optim.Adam(
            list(
                self.feature_extractor
                .parameters()
            )
            +
            list(
                self.classifier
                .parameters()
            ),
            lr=self.lr,
            weight_decay=
                self.weight_decay,
        )


    def _require_configured(
        self,
    ):

        if not self.is_configured:
            raise RuntimeError(
                "ACTA source context has not been "
                "configured. Call "
                "configure_source_context() first."
            )


    # ========================================================
    # Stage-A -> Stage-B bridge
    # ========================================================

    def configure_source_context(
        self,
        dataset_name,
        source_id,
        seed,
        source_model_root=None,
        semantic_root=None,
    ):
        """
        Configure one matched ACTA adaptation run.

        Loads:

            source_models/<dataset>/source_<id>/seed_<seed>.pth

        and:

            semantic_packages/<dataset>/source_<id>/
                semantic_package.pt

        No target information is required.
        """

        dataset_name = str(
            dataset_name
        )

        source_id = str(
            source_id
        )

        seed = int(
            seed
        )

        if source_model_root is None:
            source_model_root = (
                self.source_model_root
            )

        if semantic_root is None:
            semantic_root = (
                self.semantic_root
            )

        source_checkpoint_path = (
            Path(source_model_root)
            /
            dataset_name
            /
            f"source_{source_id}"
            /
            f"seed_{seed}.pth"
        )

        semantic_package_path = (
            Path(semantic_root)
            /
            dataset_name
            /
            f"source_{source_id}"
            /
            "semantic_package.pt"
        )

        if not source_checkpoint_path.exists():
            raise FileNotFoundError(
                "Missing ACTA source checkpoint: "
                f"{source_checkpoint_path}"
            )

        if not semantic_package_path.exists():
            raise FileNotFoundError(
                "Missing ACTA semantic package: "
                f"{semantic_package_path}"
            )

        # ----------------------------------------------------
        # Load exact source-only student theta_S
        # ----------------------------------------------------

        checkpoint = torch.load(
            source_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if str(
            checkpoint["dataset"]
        ) != dataset_name:

            raise RuntimeError(
                "Source checkpoint dataset mismatch."
            )

        if str(
            checkpoint["source_domain"]
        ) != source_id:

            raise RuntimeError(
                "Source checkpoint domain mismatch."
            )

        if int(
            checkpoint["seed"]
        ) != seed:

            raise RuntimeError(
                "Source checkpoint seed mismatch."
            )

        self.feature_extractor.load_state_dict(
            checkpoint[
                "feature_extractor"
            ],
            strict=True,
        )

        self.classifier.load_state_dict(
            checkpoint[
                "classifier"
            ],
            strict=True,
        )

        # Move student before synchronizing EMA.
        self.feature_extractor.to(
            self.device_runtime
        )

        self.classifier.to(
            self.device_runtime
        )

        # ----------------------------------------------------
        # EMA starts EXACTLY from theta_S
        # ----------------------------------------------------

        self.ema_feature_extractor.to(
            self.device_runtime
        )

        self.ema_classifier.to(
            self.device_runtime
        )

        self._sync_ema_from_student()

        # ----------------------------------------------------
        # Frozen source-side semantics
        # ----------------------------------------------------

        semantic_bank = FrozenSemanticBank(
            semantic_package_path
        )

        if (
            semantic_bank.dataset
            != dataset_name
        ):
            raise RuntimeError(
                "Semantic package dataset mismatch."
            )

        if (
            semantic_bank.source_domain
            != source_id
        ):
            raise RuntimeError(
                "Semantic package source mismatch."
            )

        if (
            semantic_bank.num_classes
            != self.configs.num_classes
        ):
            raise RuntimeError(
                "Semantic package class-count mismatch."
            )

        semantic_bank.to(
            self.device_runtime
        )

        semantic_bank.eval()

        self.semantic_bank = (
            semantic_bank
        )

        # ----------------------------------------------------
        # Shared UTA / ACTA alignment primitive
        # ----------------------------------------------------

        self.alignment_core = (
            ACTAAlignmentCore(
                semantic_bank=
                    self.semantic_bank,

                gamma=
                    self.gamma,

                lambda_sem=
                    self.lambda_sem,

                detach_source=True,
            )
        )

        self.alignment_core.to(
            self.device_runtime
        )

        # ----------------------------------------------------
        # Fresh Stage-B optimizer.
        #
        # IMPORTANT:
        # checkpoint["optimizer"] is intentionally ignored.
        # ----------------------------------------------------

        self.optimizer = (
            self._fresh_optimizer()
        )

        # ----------------------------------------------------
        # Context metadata
        # ----------------------------------------------------

        self.dataset_name = (
            dataset_name
        )

        self.source_id = (
            source_id
        )

        self.seed = (
            seed
        )

        self.source_checkpoint_path = str(
            source_checkpoint_path
        )

        self.semantic_package_path = str(
            semantic_package_path
        )

        self.is_configured = True

        # Ensure adaptation mode has frozen BN semantics
        # whenever train() is called later.
        self._freeze_ema()
        self.semantic_bank.eval()

        return self


    # ========================================================
    # Training mode
    # ========================================================

    def train(
        self,
        mode=True,
    ):
        """
        ACTA training mode.

        Student dropout remains active.

        Student BatchNorm running statistics remain frozen.

        EMA teacher and source semantic teacher always remain
        in evaluation mode.
        """

        super().train(
            mode
        )

        if mode:

            self._set_batchnorm_eval(
                self.feature_extractor
            )

        if self.ema_feature_extractor is not None:
            self.ema_feature_extractor.eval()

        if self.ema_classifier is not None:
            self.ema_classifier.eval()

        if self.semantic_bank is not None:
            self.semantic_bank.eval()

        return self


    # ========================================================
    # EMA
    # ========================================================

    @torch.no_grad()
    def update_ema(
        self,
    ):
        """
        EMA parameter update after each optimizer step.

            theta_bar <-
                mu theta_bar
                +
                (1-mu) theta

        BN buffers are directly copied because ACTA freezes
        their running statistics during adaptation.
        """

        self._require_configured()

        mu = self.ema_momentum

        for ema_parameter, parameter in zip(
            self.ema_feature_extractor.parameters(),
            self.feature_extractor.parameters(),
        ):

            ema_parameter.mul_(
                mu
            ).add_(
                parameter.detach(),
                alpha=1.0 - mu,
            )

        for ema_parameter, parameter in zip(
            self.ema_classifier.parameters(),
            self.classifier.parameters(),
        ):

            ema_parameter.mul_(
                mu
            ).add_(
                parameter.detach(),
                alpha=1.0 - mu,
            )

        # Copy buffers exactly.
        for ema_buffer, buffer in zip(
            self.ema_feature_extractor.buffers(),
            self.feature_extractor.buffers(),
        ):
            ema_buffer.copy_(
                buffer.detach()
            )

        for ema_buffer, buffer in zip(
            self.ema_classifier.buffers(),
            self.classifier.buffers(),
        ):
            ema_buffer.copy_(
                buffer.detach()
            )

        self._freeze_ema()


    @torch.no_grad()
    def ema_probabilities(
        self,
        x,
    ):
        """
        Target class probabilities used later for soft
        class conditioning.

            p_t(c) = softmax(C_ema(E_ema(x)))
        """

        self._require_configured()

        self.ema_feature_extractor.eval()
        self.ema_classifier.eval()

        features = (
            self.ema_feature_extractor(
                x
            )
        )

        logits = (
            self.ema_classifier(
                features
            )
        )

        return F.softmax(
            logits,
            dim=-1,
        )


    # ========================================================
    # Convenience feature extraction
    # ========================================================

    def temporal_features(
        self,
        x,
    ):
        """
        Return temporal feature map before adaptive pooling.

        Shape:
            [B, D, L]
        """

        return (
            self.feature_extractor
            .forward_features(
                x
            )
        )


    # ========================================================
    # ACTA update
    #
    # Implemented in Step 7.3 after reference sampling.
    # ========================================================

    def update(
        self,
        src_x,
        src_y,
        trg_x,
    ):

        raise NotImplementedError(
            "ACTA.update() will be completed after "
            "the class-indexed source reference "
            "sampler is integrated."
        )


    # ========================================================
    # Save / load adapted model
    # ========================================================

    def save_model(
        self,
        path,
    ):

        self._require_configured()

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "format":
                "ACTA_ADAPTATION_V1",

            "context": {
                "dataset":
                    self.dataset_name,

                "source_id":
                    self.source_id,

                "seed":
                    int(
                        self.seed
                    ),
            },

            "feature_extractor":
                {
                    key:
                        value.detach().cpu()

                    for key, value
                    in self.feature_extractor
                    .state_dict()
                    .items()
                },

            "classifier":
                {
                    key:
                        value.detach().cpu()

                    for key, value
                    in self.classifier
                    .state_dict()
                    .items()
                },

            "ema_feature_extractor":
                {
                    key:
                        value.detach().cpu()

                    for key, value
                    in self.ema_feature_extractor
                    .state_dict()
                    .items()
                },

            "ema_classifier":
                {
                    key:
                        value.detach().cpu()

                    for key, value
                    in self.ema_classifier
                    .state_dict()
                    .items()
                },
        }

        torch.save(
            checkpoint,
            path
        )


    def load_model(
        self,
        path,
    ):

        path = Path(
            path
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            checkpoint.get(
                "format"
            )
            !=
            "ACTA_ADAPTATION_V1"
        ):
            raise RuntimeError(
                "Unsupported ACTA checkpoint format."
            )

        context = checkpoint[
            "context"
        ]

        # If this object was just constructed for evaluation,
        # reconstruct its source context automatically.
        if not self.is_configured:

            self.configure_source_context(
                dataset_name=
                    context["dataset"],

                source_id=
                    context["source_id"],

                seed=
                    int(
                        context["seed"]
                    ),
            )

        self.feature_extractor.load_state_dict(
            checkpoint[
                "feature_extractor"
            ],
            strict=True,
        )

        self.classifier.load_state_dict(
            checkpoint[
                "classifier"
            ],
            strict=True,
        )

        self.ema_feature_extractor.load_state_dict(
            checkpoint[
                "ema_feature_extractor"
            ],
            strict=True,
        )

        self.ema_classifier.load_state_dict(
            checkpoint[
                "ema_classifier"
            ],
            strict=True,
        )

        self.to(
            self.device_runtime
        )

        self._freeze_ema()

        if self.semantic_bank is not None:
            self.semantic_bank.eval()

        return self