import copy

import torch
import torch.nn as nn

from algorithms.algorithms_base import Algorithm
from utils.module import *


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
            "Source_loss": float(source_loss.detach().item())
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