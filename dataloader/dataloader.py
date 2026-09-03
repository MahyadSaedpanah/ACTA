import os
import numpy as np

import torch
from torch.utils.data import Dataset


class Load_Dataset(Dataset):

    def __init__(
        self,
        dataset,
        input_channels=None,
        mean=None,
        std=None,
        eps=1e-6
    ):
        super(Load_Dataset, self).__init__()

        x = dataset["samples"]
        y = dataset["labels"]

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        x = x.float()
        y = y.long().view(-1)

        # --------------------------------------------------
        # Ensure shape = [N, C, T]
        # --------------------------------------------------
        if x.ndim == 2:
            x = x.unsqueeze(1)

        if x.ndim != 3:
            raise RuntimeError(
                f"Expected a 3-D tensor [N,C,T], "
                f"but got shape={tuple(x.shape)}"
            )

        if input_channels is not None:

            if x.shape[1] == input_channels:
                pass

            elif x.shape[2] == input_channels:
                x = x.permute(0, 2, 1).contiguous()

            else:
                raise RuntimeError(
                    f"Cannot locate channel dimension. "
                    f"shape={tuple(x.shape)}, "
                    f"expected input_channels={input_channels}"
                )

        else:
            # Backward-compatible fallback
            if x.shape[1] > x.shape[2]:
                x = x.permute(0, 2, 1).contiguous()

        # --------------------------------------------------
        # Normalization
        #
        # If mean/std are not supplied, this is the TRAIN
        # dataset and its statistics are estimated here.
        #
        # If supplied, e.g. TEST dataset, reuse TRAIN stats.
        # --------------------------------------------------
        if mean is None:
            mean = x.mean(
                dim=(0, 2),
                keepdim=True
            )

        if std is None:
            std = x.std(
                dim=(0, 2),
                keepdim=True
            )

        mean = mean.float()
        std = std.float().clamp_min(eps)

        # Normalize ONCE.
        x = (x - mean) / std

        self.x_data = x.contiguous()
        self.y_data = y.contiguous()

        # Convenient aliases for later ACTA components.
        self.x = self.x_data
        self.y = self.y_data

        self.mean = mean
        self.std = std

        self.num_channels = self.x_data.shape[1]
        self.len = self.x_data.shape[0]


    def __getitem__(self, index):

        # IMPORTANT:
        # no mutation / no repeated normalization
        return (
            self.x_data[index],
            self.y_data[index]
        )


    def __len__(self):
        return self.len


def data_generator(
    data_path,
    domain_id,
    args
):

    train_path = os.path.join(
        data_path,
        f"train_{domain_id}.pt"
    )

    test_path = os.path.join(
        data_path,
        f"test_{domain_id}.pt"
    )

    train_raw = torch.load(
        train_path,
        weights_only=False
    )

    test_raw = torch.load(
        test_path,
        weights_only=False
    )

    input_channels = getattr(
        args,
        "enc_in",
        None
    )

    # ------------------------------------------------------
    # TRAIN
    # Learn normalization statistics from train only.
    # ------------------------------------------------------
    train_dataset = Load_Dataset(
        train_raw,
        input_channels=input_channels
    )

    # ------------------------------------------------------
    # TEST
    # Reuse statistics from the corresponding train split.
    # ------------------------------------------------------
    test_dataset = Load_Dataset(
        test_raw,
        input_channels=input_channels,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    batch_size = args.bs

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=args.shuffle,
        drop_last=True,
        num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    return (
        train_loader,
        test_loader
    )