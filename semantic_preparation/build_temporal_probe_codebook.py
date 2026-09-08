"""
Build ACTA-v2 Temporal Admissibility Codebook.

The codebook converts the already validated SOURCE-ONLY local semantic
teacher into a fixed bank of executable temporal probes.

Key principle
-------------
TSA is used here as a semantic-preservation certificate, NOT as a
source-target attraction path.

For each class c and temporal probe phi_m:

    S_cm
      = 0.5 * [
            S_c(path(phi_m))
            +
            S_c(reverse(path(phi_m)))
        ]

where S_c is the existing LocalSemanticEnergy path logit, including its
class-specific path_bias and SOURCE-LENGTH normalization.

We store:

    sigmoid_score_cm = sigmoid(S_cm)

and the reliability-weighted score

    weight_cm = kappa_c * sigmoid_score_cm

The sigmoid is used only as a bounded monotone transform of the distilled
teacher logit; it is NOT claimed to be a calibrated probability.

No target ID, target data, or target labels are used.

Probe bank
----------
We use a fixed, source-independent bank of smooth monotone endpoint-preserving
warps. This avoids selecting probe shapes using target performance.

Families:
    quadratic:
        phi(u) = u + a*u*(1-u)

    sine-1:
        phi(u) = u + a*sin(2*pi*u)/(2*pi)

    sine-2:
        phi(u) = u + a*sin(4*pi*u)/(4*pi)

For |a| < 1, all derivatives remain positive.
We lock severities |a| in {0.25, 0.50} and both signs.

Total non-identity probes:
    3 families * 2 severities * 2 signs = 12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algorithms.components.semantic_teacher import FrozenSemanticBank
from semantic_preparation.dtw_utils import (
    local_edge_features,
    validate_path,
)


CODEBOOK_VERSION = "ACTA_TEMPORAL_PROBE_CODEBOOK_V1"

PROBE_FAMILIES = (
    "quadratic",
    "sine1",
    "sine2",
)

PROBE_SEVERITIES = (
    0.25,
    0.50,
)


# ============================================================
# 1. CONTINUOUS MONOTONE PROBES
# ============================================================

def normalized_probe_mapping(
    family: str,
    signed_amplitude: float,
    num_points: int,
) -> np.ndarray:
    """
    Build normalized target->source sampling map:

        u_source = phi(v_target)

    with endpoints fixed at 0 and 1.

    Returns
    -------
    np.ndarray [num_points], float64
    """

    family = str(family)
    a = float(signed_amplitude)
    n = int(num_points)

    if family not in PROBE_FAMILIES:
        raise ValueError(
            f"Unknown probe family: {family}"
        )

    if n < 2:
        raise ValueError(
            "num_points must be >= 2."
        )

    if abs(a) >= 1.0:
        raise ValueError(
            "Probe amplitude must satisfy |a| < 1 "
            "to guarantee monotonicity."
        )

    v = np.linspace(
        0.0,
        1.0,
        n,
        dtype=np.float64,
    )

    if family == "quadratic":

        phi = (
            v
            +
            a
            *
            v
            *
            (1.0 - v)
        )

    elif family == "sine1":

        phi = (
            v
            +
            a
            *
            np.sin(
                2.0
                *
                math.pi
                *
                v
            )
            /
            (
                2.0
                *
                math.pi
            )
        )

    elif family == "sine2":

        phi = (
            v
            +
            a
            *
            np.sin(
                4.0
                *
                math.pi
                *
                v
            )
            /
            (
                4.0
                *
                math.pi
            )
        )

    else:
        raise RuntimeError(
            "Unreachable probe family."
        )

    # Exact endpoints.
    phi[0] = 0.0
    phi[-1] = 1.0

    # Analytically monotone, but verify numerically.
    diffs = np.diff(phi)

    if np.any(
        diffs <= 0.0
    ):
        raise RuntimeError(
            f"Probe is not strictly monotone: "
            f"family={family}, amplitude={a}."
        )

    if (
        phi.min() < -1e-12
        or
        phi.max() > 1.0 + 1e-12
    ):
        raise RuntimeError(
            "Probe mapping left [0,1]."
        )

    return np.clip(
        phi,
        0.0,
        1.0,
    )


def build_probe_specifications() -> List[Dict]:
    """
    Fixed preregistered probe bank.
    """

    specs = []

    for family in PROBE_FAMILIES:

        for severity in PROBE_SEVERITIES:

            for sign in (
                -1.0,
                +1.0,
            ):

                amplitude = (
                    sign
                    *
                    float(severity)
                )

                sign_name = (
                    "neg"
                    if sign < 0
                    else
                    "pos"
                )

                name = (
                    f"{family}_"
                    f"a{severity:.2f}_"
                    f"{sign_name}"
                )

                specs.append(
                    {
                        "name":
                            name,

                        "family":
                            family,

                        "severity":
                            float(severity),

                        "signed_amplitude":
                            float(amplitude),
                    }
                )

    if len(specs) != 12:
        raise RuntimeError(
            "Unexpected probe-bank size."
        )

    return specs


# ============================================================
# 2. MAPPING -> ACTA MONOTONE PATH
# ============================================================

def mapping_to_acta_path(
    phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert target->source mapping:

        source_i ~= phi(target_j)

    into ACTA's exact monotone path convention.

    Coordinates:
        pi = source index i
        pj = target index j

    The path uses only:
        diagonal (1,1)
        vertical (1,0)
        horizontal (0,1)
    """

    phi = np.asarray(
        phi,
        dtype=np.float64,
    )

    if (
        phi.ndim != 1
        or
        len(phi) < 2
    ):
        raise ValueError(
            "phi must be a 1-D mapping of length >= 2."
        )

    g = int(
        len(phi)
    )

    source_index = np.rint(
        phi
        *
        float(
            g - 1
        )
    ).astype(
        np.int64
    )

    source_index[0] = 0
    source_index[-1] = (
        g - 1
    )

    source_index = np.maximum.accumulate(
        source_index
    )

    source_index = np.clip(
        source_index,
        0,
        g - 1,
    )

    pi = [0]
    pj = [0]

    current_i = 0
    current_j = 0

    for next_j in range(
        1,
        g,
    ):

        next_i = int(
            source_index[
                next_j
            ]
        )

        delta_i = (
            next_i
            -
            current_i
        )

        if delta_i < 0:
            raise RuntimeError(
                "Rounded probe path became non-monotone."
            )

        if delta_i == 0:

            # horizontal:
            # target advances, source fixed
            current_j = next_j

            pi.append(
                current_i
            )

            pj.append(
                current_j
            )

        else:

            # First advance both axes once.
            current_i += 1
            current_j = next_j

            pi.append(
                current_i
            )

            pj.append(
                current_j
            )

            # If source mapping jumped by >1,
            # finish with vertical moves.
            while (
                current_i
                <
                next_i
            ):

                current_i += 1

                pi.append(
                    current_i
                )

                pj.append(
                    current_j
                )

    pi = np.asarray(
        pi,
        dtype=np.int64,
    )

    pj = np.asarray(
        pj,
        dtype=np.int64,
    )

    validate_path(
        pi,
        pj,
        g,
        g,
    )

    return (
        pi,
        pj,
    )


# ============================================================
# 3. EXACT LOCAL-STUDENT PATH SCORING
# ============================================================

@torch.no_grad()
def score_probe_for_class(
    semantic_bank: FrozenSemanticBank,
    class_id: int,
    pi: np.ndarray,
    pj: np.ndarray,
    source_length: int,
) -> Dict[str, float]:
    """
    Score both forward and reversed path orientation.

    Global teacher was trained on symmetric geometry, so the
    codebook uses their average rather than privileging an
    arbitrary orientation.
    """

    class_id = int(
        class_id
    )

    g = int(
        source_length
    )

    model = semantic_bank.models[
        class_id
    ]

    device = next(
        model.parameters()
    ).device

    dtype = next(
        model.parameters()
    ).dtype

    local_forward = (
        local_edge_features(
            pi,
            pj,
            g,
            g,
        )
    )

    local_reverse = (
        local_edge_features(
            pj,
            pi,
            g,
            g,
        )
    )

    local_forward_t = torch.as_tensor(
        local_forward,
        device=device,
        dtype=dtype,
    )

    local_reverse_t = torch.as_tensor(
        local_reverse,
        device=device,
        dtype=dtype,
    )

    forward_logit = (
        model.score_single_path(
            local_forward_t,
            source_length=g,
        )
    )

    reverse_logit = (
        model.score_single_path(
            local_reverse_t,
            source_length=g,
        )
    )

    symmetric_logit = (
        0.5
        *
        (
            forward_logit
            +
            reverse_logit
        )
    )

    bounded_score = torch.sigmoid(
        symmetric_logit
    )

    kappa = semantic_bank.class_kappa(
        class_id
    ).to(
        device=bounded_score.device,
        dtype=bounded_score.dtype,
    )

    reliability_weighted = (
        kappa
        *
        bounded_score
    )

    return {
        "forward_logit":
            float(
                forward_logit.item()
            ),

        "reverse_logit":
            float(
                reverse_logit.item()
            ),

        "symmetric_logit":
            float(
                symmetric_logit.item()
            ),

        "sigmoid_score":
            float(
                bounded_score.item()
            ),

        "kappa":
            float(
                kappa.item()
            ),

        "reliability_weighted_score":
            float(
                reliability_weighted.item()
            ),

        "orientation_gap":
            float(
                torch.abs(
                    forward_logit
                    -
                    reverse_logit
                ).item()
            ),
    }


# ============================================================
# 4. CODEBOOK BUILD
# ============================================================

def build_codebook(
    semantic_package_path: Path,
    output_path: Path,
) -> Dict:
    """
    Build and save one source-domain codebook.
    """

    semantic_bank = FrozenSemanticBank(
        semantic_package_path
    )

    semantic_bank.eval()

    semantic_length = int(
        semantic_bank.semantic_temporal_length
    )

    specs = build_probe_specifications()

    probes = []

    class_logits = torch.empty(
        semantic_bank.num_classes,
        len(specs),
        dtype=torch.float32,
    )

    class_sigmoid_scores = torch.empty_like(
        class_logits
    )

    class_weighted_scores = torch.empty_like(
        class_logits
    )

    orientation_gaps = torch.empty_like(
        class_logits
    )

    for probe_id, spec in enumerate(
        specs
    ):

        phi = normalized_probe_mapping(
            family=
                spec["family"],

            signed_amplitude=
                spec[
                    "signed_amplitude"
                ],

            num_points=
                semantic_length,
        )

        pi, pj = mapping_to_acta_path(
            phi
        )

        probe_record = dict(
            spec
        )

        probe_record.update(
            {
                "probe_id":
                    int(
                        probe_id
                    ),

                "normalized_mapping":
                    torch.tensor(
                        phi,
                        dtype=torch.float32,
                    ),

                "path_pi":
                    torch.tensor(
                        pi,
                        dtype=torch.int64,
                    ),

                "path_pj":
                    torch.tensor(
                        pj,
                        dtype=torch.int64,
                    ),

                "path_length":
                    int(
                        len(pi)
                    ),
            }
        )

        probes.append(
            probe_record
        )

        for class_id in range(
            semantic_bank.num_classes
        ):

            score = score_probe_for_class(
                semantic_bank=
                    semantic_bank,

                class_id=
                    class_id,

                pi=
                    pi,

                pj=
                    pj,

                source_length=
                    semantic_length,
            )

            class_logits[
                class_id,
                probe_id,
            ] = score[
                "symmetric_logit"
            ]

            class_sigmoid_scores[
                class_id,
                probe_id,
            ] = score[
                "sigmoid_score"
            ]

            class_weighted_scores[
                class_id,
                probe_id,
            ] = score[
                "reliability_weighted_score"
            ]

            orientation_gaps[
                class_id,
                probe_id,
            ] = score[
                "orientation_gap"
            ]

    codebook = {
        "version":
            CODEBOOK_VERSION,

        "dataset":
            semantic_bank.dataset,

        "source_domain":
            semantic_bank.source_domain,

        "num_classes":
            int(
                semantic_bank.num_classes
            ),

        "semantic_temporal_length":
            semantic_length,

        "num_probes":
            int(
                len(probes)
            ),

        "probe_bank_definition": {
            "families":
                list(
                    PROBE_FAMILIES
                ),

            "severities":
                list(
                    PROBE_SEVERITIES
                ),

            "signs":
                [
                    -1,
                    +1,
                ],

            "mapping_convention":
                "target_to_source",

            "source_target_path_convention":
                "pi=source_index,pj=target_index",

            "selection_uses_target_information":
                False,
        },

        "probes":
            probes,

        "class_symmetric_logits":
            class_logits,

        "class_sigmoid_scores":
            class_sigmoid_scores,

        "class_reliability_weighted_scores":
            class_weighted_scores,

        "class_orientation_gaps":
            orientation_gaps,

        "class_kappas":
            semantic_bank.kappas
            .detach()
            .cpu(),

        "score_semantics": {
            "path_score":
                (
                    "Existing LocalSemanticEnergy path logit, "
                    "including path_bias and source-length normalization."
                ),

            "sigmoid_score":
                (
                    "Bounded monotone transform of the distilled logit; "
                    "not claimed to be a calibrated probability."
                ),

            "reliability_weighted_score":
                "kappa_c * sigmoid_score.",

            "target_information_used":
                False,
        },

        "source_semantic_package":
            str(
                semantic_package_path
            ),

        "target_information_used":
            False,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        codebook,
        output_path,
    )

    return codebook


# ============================================================
# 5. REPORT
# ============================================================

def print_summary(
    codebook: Dict,
    output_path: Path,
):
    """
    Compact sanity report.
    """

    scores = codebook[
        "class_reliability_weighted_scores"
    ]

    logits = codebook[
        "class_symmetric_logits"
    ]

    gaps = codebook[
        "class_orientation_gaps"
    ]

    print(
        "\n"
        +
        "=" * 78
    )

    print(
        "ACTA-v2 TEMPORAL ADMISSIBILITY CODEBOOK"
    )

    print(
        "=" * 78
    )

    print(
        f"dataset={codebook['dataset']}  "
        f"source={codebook['source_domain']}  "
        f"classes={codebook['num_classes']}  "
        f"probes={codebook['num_probes']}  "
        f"G={codebook['semantic_temporal_length']}"
    )

    for class_id in range(
        codebook["num_classes"]
    ):

        class_scores = scores[
            class_id
        ]

        class_logits = logits[
            class_id
        ]

        best_idx = int(
            torch.argmax(
                class_scores
            ).item()
        )

        best_name = codebook[
            "probes"
        ][
            best_idx
        ][
            "name"
        ]

        print(
            f"  c{class_id}: "
            f"kappa={float(codebook['class_kappas'][class_id]):.4f}  "
            f"logit_mean={float(class_logits.mean()):+.4f}  "
            f"weighted_mean={float(class_scores.mean()):.4f}  "
            f"best={best_name}({float(class_scores[best_idx]):.4f})  "
            f"orient_gap_mean={float(gaps[class_id].mean()):.4f}"
        )

    print(
        "-" * 78
    )

    print(
        f"target_information_used="
        f"{codebook['target_information_used']}"
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        "=" * 78
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--semantic_root",
        default="./semantic_packages",
    )

    parser.add_argument(
        "--output_root",
        default=None,
    )

    args = parser.parse_args()

    semantic_root = Path(
        args.semantic_root
    )

    if args.output_root is None:
        output_root = semantic_root
    else:
        output_root = Path(
            args.output_root
        )

    semantic_package_path = (
        semantic_root
        /
        str(
            args.dataset
        )
        /
        f"source_{args.source}"
        /
        "semantic_package.pt"
    )

    output_path = (
        output_root
        /
        str(
            args.dataset
        )
        /
        f"source_{args.source}"
        /
        "temporal_probe_codebook.pt"
    )

    if not semantic_package_path.exists():
        raise FileNotFoundError(
            "Missing semantic package: "
            f"{semantic_package_path}"
        )

    codebook = build_codebook(
        semantic_package_path=
            semantic_package_path,

        output_path=
            output_path,
    )

    print_summary(
        codebook,
        output_path,
    )


if __name__ == "__main__":
    main()