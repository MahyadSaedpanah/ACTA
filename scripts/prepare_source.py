from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Repository root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator
from utils.module import CNN, TemporalClassifierHead


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Source model
# ============================================================

class SourceModel(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.feature_extractor = CNN(configs)

        self.classifier = TemporalClassifierHead(
            self.feature_extractor.out_dim,
            configs.num_classes
        )

    def forward(self, x):

        features = self.feature_extractor(x)

        logits = self.classifier(features)

        return logits


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device
):

    model.eval()

    predictions = []
    labels_all = []

    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:

        x = x.float().to(device)
        y = y.view(-1).long().to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits,
            y,
            reduction="sum"
        )

        pred = logits.argmax(dim=1)

        total_loss += loss.item()
        total_count += y.numel()

        predictions.append(
            pred.cpu()
        )

        labels_all.append(
            y.cpu()
        )

    predictions = (
        torch.cat(predictions)
        .numpy()
    )

    labels_all = (
        torch.cat(labels_all)
        .numpy()
    )

    accuracy = accuracy_score(
        labels_all,
        predictions
    )

    f1 = f1_score(
        labels_all,
        predictions,
        average="macro",
        zero_division=0
    )

    loss = (
        total_loss
        / max(total_count, 1)
    )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1)
    }


# ============================================================
# Portable checkpoint
# ============================================================

def cpu_state_dict(module):

    return {
        key: value.detach().cpu()
        for key, value
        in module.state_dict().items()
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    dataset,
    source_id,
    seed,
    epoch,
    configs,
    train_dataset,
    source_metrics
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "dataset":
            dataset,

        "source_domain":
            str(source_id),

        "seed":
            int(seed),

        "epoch":
            int(epoch),

        "feature_extractor":
            cpu_state_dict(
                model.feature_extractor
            ),

        "classifier":
            cpu_state_dict(
                model.classifier
            ),

        "optimizer":
            optimizer.state_dict(),

        "source_test_metrics":
            source_metrics,

        "normalization": {
            "mean":
                train_dataset.mean
                .detach()
                .cpu(),

            "std":
                train_dataset.std
                .detach()
                .cpu()
        },

        "model_config": {
            "input_channels":
                int(configs.input_channels),

            "sequence_len":
                int(configs.sequence_len),

            "num_classes":
                int(configs.num_classes),

            "kernel_size":
                int(configs.kernel_size),

            "stride":
                int(configs.stride),

            "dropout":
                float(configs.dropout),

            "mid_channels":
                int(configs.mid_channels),

            "t_feat_dim":
                int(configs.t_feat_dim),

            "features_len":
                int(configs.features_len)
        }
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# Train one source / seed
# ============================================================

def train_source(
    dataset,
    source_id,
    seed,
    data_path,
    save_root,
    device,
    epochs,
    batch_size,
    lr,
    weight_decay,
    num_workers
):

    print(
        "\n"
        "============================================="
    )

    print(
        f"Dataset: {dataset}"
    )

    print(
        f"Source:  {source_id}"
    )

    print(
        f"Seed:    {seed}"
    )

    print(
        f"Device:  {device}"
    )

    print(
        "============================================="
    )

    set_seed(seed)

    dataset_class = get_dataset_class(
        dataset
    )

    configs = dataset_class()

    loader_args = argparse.Namespace(

        bs=batch_size,

        shuffle=True,

        num_workers=num_workers,

        enc_in=configs.input_channels
    )

    domain_data_path = (
        Path(data_path)
        / dataset
    )

    train_file = (
        domain_data_path
        / f"train_{source_id}.pt"
    )

    test_file = (
        domain_data_path
        / f"test_{source_id}.pt"
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Missing source train file: "
            f"{train_file}"
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Missing source test file: "
            f"{test_file}"
        )

    train_loader, test_loader = (
        data_generator(
            str(domain_data_path),
            str(source_id),
            loader_args
        )
    )

    print(
        "Train samples:",
        len(train_loader.dataset)
    )

    print(
        "Test samples: ",
        len(test_loader.dataset)
    )

    model = SourceModel(
        configs
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # --------------------------------------------------------
    # Source-only training
    #
    # IMPORTANT:
    # No target data.
    # No target labels.
    # No source-test checkpoint selection.
    # Final epoch is always saved.
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        total_loss = 0.0
        total_count = 0

        correct = 0

        for x, y in train_loader:

            x = x.float().to(device)

            y = (
                y.view(-1)
                .long()
                .to(device)
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits,
                y
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            batch_size_now = (
                y.numel()
            )

            total_loss += (
                loss.item()
                * batch_size_now
            )

            total_count += (
                batch_size_now
            )

            correct += (
                logits.argmax(dim=1)
                .eq(y)
                .sum()
                .item()
            )

        train_loss = (
            total_loss
            / max(total_count, 1)
        )

        train_acc = (
            correct
            / max(total_count, 1)
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{epochs:03d} | "
            f"source CE={train_loss:.6f} | "
            f"train acc={100.0 * train_acc:.2f}%"
        )

    # --------------------------------------------------------
    # Evaluate ONCE, after training.
    #
    # This evaluation does NOT choose the checkpoint.
    # --------------------------------------------------------

    source_metrics = evaluate(
        model,
        test_loader,
        device
    )

    print(
        "\nFinal source-test:"
    )

    print(
        f"  loss = "
        f"{source_metrics['loss']:.6f}"
    )

    print(
        f"  acc  = "
        f"{100.0 * source_metrics['accuracy']:.2f}%"
    )

    print(
        f"  f1   = "
        f"{source_metrics['f1']:.6f}"
    )

    checkpoint_path = (
        Path(save_root)
        / dataset
        / f"source_{source_id}"
        / f"seed_{seed}.pth"
    )

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        dataset,
        source_id,
        seed,
        epochs,
        configs,
        train_loader.dataset,
        source_metrics
    )

    print(
        "\nSaved:"
    )

    print(
        checkpoint_path
    )

    return checkpoint_path


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare source-only checkpoints "
            "for ACTA."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="UCIHAR"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data"
    )

    parser.add_argument(
        "--save_root",
        type=str,
        default="./source_models"
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4]
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but no CUDA device "
            "is available."
        )

    device = torch.device(
        args.device
    )

    dataset_class = get_dataset_class(
        args.dataset
    )

    configs = dataset_class()

    if args.sources is None:

        # Unique source domains in benchmark order.
        sources = []

        for src, _ in configs.scenarios:

            src = str(src)

            if src not in sources:
                sources.append(src)

    else:

        sources = [
            str(source)
            for source
            in args.sources
        ]

    print(
        "ACTA Source Preparation"
    )

    print(
        "Dataset:",
        args.dataset
    )

    print(
        "Sources:",
        sources
    )

    print(
        "Seeds:",
        args.seeds
    )

    print(
        "Epochs:",
        args.epochs
    )

    for source_id in sources:

        for seed in args.seeds:

            train_source(
                dataset=args.dataset,
                source_id=source_id,
                seed=seed,
                data_path=args.data_path,
                save_root=args.save_root,
                device=device,
                epochs=args.epochs,
                batch_size=args.bs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers
            )


if __name__ == "__main__":
    main()