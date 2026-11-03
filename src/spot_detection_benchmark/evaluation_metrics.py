from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize
import scipy.spatial.distance

EPS = 1e-12

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def linear_sum_assignment(
    matrix: np.ndarray, cutoff: float | None = None
) -> tuple[list, list]:
    """Solve the linear sum assignment problem with a cutoff.

    A problem instance is described by matrix matrix where each matrix[i, j]
    is the cost of matching i (worker) with j (job). The goal is to find the
    most optimal assignment of j to i if the given cost is below the cutoff.

    Args:
        matrix: Matrix containing cost/distance to assign cols to rows.
        cutoff: Maximum cost/distance value assignments can have.

    Returns:
        (rows, columns) corresponding to the matching assignment.
    """
    # Handle zero-sized matrices (occurs if true or pred has no items)
    if matrix.size == 0:
        return [], []

    # Prevent scipy to optimize on values above the cutoff
    if cutoff is not None and cutoff != 0:
        matrix = np.where(matrix >= cutoff, matrix.max(), matrix)

    row, col = scipy.optimize.linear_sum_assignment(matrix)

    if cutoff is None:
        return list(row), list(col)

    # As scipy will still assign all columns to rows
    # We here remove assigned values falling below the cutoff
    nrow = []
    ncol = []
    for r, c in zip(row, col):
        if matrix[r, c] <= cutoff:
            nrow.append(r)
            ncol.append(c)
    return nrow, ncol


def f1_integral(
    pred: np.ndarray,
    true: np.ndarray,
    mdist: float = 3.0,
    n_cutoffs: int = 50,
) -> float:
    """F1 integral calculation / area under F1 vs. cutoff.

    Compute the area under the curve when plotting F1 score vs cutoff values
    (n_cutoffs values from 0 to mdist), normalised to 0-1. Optimal score is ~1
    (floating point inaccuracy) when F1 is achieved across all cutoff values
    including 0.

    Args:
        pred: Array of shape (n, d) for predicted coordinates.
        true: Array of shape (n, d) for ground truth coordinates.
        mdist: Maximum cutoff distance to calculate F1.
        n_cutoffs: Number of intermediate cutoff steps.
    """
    if pred.size == 0 or true.size == 0:
        warnings.warn(
            f"Pred ({pred.shape}) and true ({true.shape}) must have size != 0.",
            RuntimeWarning,
        )
        return 0.0

    cutoffs = np.linspace(start=0, stop=mdist, num=n_cutoffs)
    matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")
    f1_scores = [_f1_at_cutoff(matrix, pred, true, cutoff) for cutoff in cutoffs]
    return _trapz(f1_scores, cutoffs) / mdist  # Norm. to 0-1


def _f1_at_cutoff(
    matrix: np.ndarray, pred: np.ndarray, true: np.ndarray, cutoff: float
) -> float:
    """F1 value at one cutoff.

    Args:
        matrix: Euclidean distances, pred along the rows and true along the columns.
        pred: All predicted coordinates.
        true: All ground truth coordinates.
        cutoff: Maximum distance of an assignment.
    """
    # Cannot assign coordinates on empty matrix
    if matrix.size == 0:
        return 0.0

    # Assignment of pred<-true and true<-pred
    pred_true_r, _ = linear_sum_assignment(matrix, cutoff)
    true_pred_r, _ = linear_sum_assignment(matrix.T, cutoff)

    # Calculation of tp/fn/fp based on number of assignments
    tp = len(true_pred_r)
    fn = len(true) - len(true_pred_r)
    fp = len(pred) - len(pred_true_r)

    recall = tp / (tp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    return (2 * precision * recall) / (precision + recall + EPS)
