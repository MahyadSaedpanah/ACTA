"""
Deterministic class-indexed source reference pool for ACTA.

The pool uses the already-normalized SOURCE TRAIN dataset.

It is separate from the ordinary source minibatch used for
supervised source CE.

For a fixed seed and identical sequence of calls:

    UTA
    ACTA
    ACTA-ClassShuffle

receive exactly the same source reference indices.
"""

from __future__ import annotations

import torch


class ClassIndexedSourceReferencePool:
    """
    Class-indexed deterministic sampler over a source dataset.

    Parameters
    ----------
    dataset:
        Source TRAIN dataset.

        dataset[i] must return:
            (x, y)

    num_classes:
        Number of task classes.

    seed:
        Dedicated reference-sampling seed.

    Notes
    -----
    Sampling is performed without replacement within each
    individual K-reference draw.

    Across different optimization steps, an example may of
    course be sampled again.
    """

    def __init__(
        self,
        dataset,
        num_classes,
        seed,
    ):

        self.dataset = dataset

        self.num_classes = int(
            num_classes
        )

        self.seed = int(
            seed
        )

        if self.num_classes <= 0:
            raise ValueError(
                "num_classes must be positive."
            )

        if len(dataset) == 0:
            raise ValueError(
                "Source reference dataset is empty."
            )

        # ----------------------------------------------------
        # Dedicated CPU generator.
        #
        # This avoids coupling reference sampling to:
        # dropout / optimizer / other torch RNG usage.
        # ----------------------------------------------------

        self.generator = torch.Generator(
            device="cpu"
        )

        self.generator.manual_seed(
            self.seed
        )

        # ----------------------------------------------------
        # Obtain labels once.
        #
        # Prefer the clean dataloader's stored labels when
        # available; otherwise fall back to dataset indexing.
        # ----------------------------------------------------

        labels = None

        for attribute in (
            "y_data",
            "y",
        ):

            if hasattr(
                dataset,
                attribute,
            ):

                labels = getattr(
                    dataset,
                    attribute,
                )

                break

        if labels is None:

            collected = []

            for index in range(
                len(dataset)
            ):

                _, label = dataset[
                    index
                ]

                collected.append(
                    int(
                        torch.as_tensor(
                            label
                        ).item()
                    )
                )

            labels = torch.tensor(
                collected,
                dtype=torch.long,
            )

        else:

            labels = torch.as_tensor(
                labels
            ).long().view(-1)

        if len(labels) != len(dataset):
            raise RuntimeError(
                "Source dataset label count mismatch."
            )

        self.labels = (
            labels.detach()
            .cpu()
            .clone()
        )

        # ----------------------------------------------------
        # Build class index sets
        # ----------------------------------------------------

        self.class_indices = {}

        for class_id in range(
            self.num_classes
        ):

            indices = torch.where(
                self.labels
                ==
                class_id
            )[0]

            indices = (
                indices
                .long()
                .cpu()
            )

            if indices.numel() == 0:
                raise RuntimeError(
                    "Source reference pool contains "
                    f"no examples for class {class_id}."
                )

            self.class_indices[
                class_id
            ] = indices


    def class_count(
        self,
        class_id,
    ):

        class_id = int(
            class_id
        )

        if class_id not in (
            self.class_indices
        ):
            raise ValueError(
                f"Invalid class_id={class_id}."
            )

        return int(
            self.class_indices[
                class_id
            ].numel()
        )


    def sample_indices(
        self,
        class_id,
        k,
    ):
        """
        Sample K SOURCE TRAIN indices from one class.

        Returns
        -------
        Tensor [K], CPU long.
        """

        class_id = int(
            class_id
        )

        k = int(
            k
        )

        if k <= 0:
            raise ValueError(
                "k must be positive."
            )

        if class_id not in (
            self.class_indices
        ):
            raise ValueError(
                f"Invalid class_id={class_id}."
            )

        candidates = (
            self.class_indices[
                class_id
            ]
        )

        n = int(
            candidates.numel()
        )

        if k > n:
            raise RuntimeError(
                f"Requested K={k} references "
                f"from class {class_id}, "
                f"but only {n} source examples exist."
            )

        permutation = torch.randperm(
            n,
            generator=self.generator,
        )

        selected = candidates[
            permutation[:k]
        ]

        return (
            selected.clone()
        )


    def fetch_indices(
        self,
        indices,
        device=None,
    ):
        """
        Fetch raw normalized source samples by dataset index.

        Parameters
        ----------
        indices:
            Sequence / tensor of source dataset indices.

        Returns
        -------
        x:
            [K, C, T]

        y:
            [K]
        """

        indices = torch.as_tensor(
            indices,
            dtype=torch.long,
        ).view(-1)

        samples = []
        labels = []

        for index in indices.tolist():

            x, y = self.dataset[
                int(index)
            ]

            samples.append(
                torch.as_tensor(
                    x
                ).float()
            )

            labels.append(
                int(
                    torch.as_tensor(
                        y
                    ).item()
                )
            )

        x = torch.stack(
            samples,
            dim=0,
        )

        y = torch.tensor(
            labels,
            dtype=torch.long,
        )

        if device is not None:

            x = x.to(
                device
            )

            y = y.to(
                device
            )

        return x, y


    def sample(
        self,
        class_id,
        k,
        device=None,
        return_indices=False,
    ):
        """
        Sample and fetch K same-class references.
        """

        indices = self.sample_indices(
            class_id=class_id,
            k=k,
        )

        x, y = self.fetch_indices(
            indices,
            device=device,
        )

        # Defensive semantic check.
        if not torch.all(
            y.detach().cpu()
            ==
            int(class_id)
        ):
            raise RuntimeError(
                "Reference sampler returned "
                "a wrong-class sample."
            )

        if return_indices:

            return (
                x,
                y,
                indices,
            )

        return x, y