from algorithms.ACON import ACON
from algorithms.ACTA import ACTA

__all__ = [
    "ACON",
    "ACTA",
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(
                algorithm_name
            )
        )

    return globals()[algorithm_name]