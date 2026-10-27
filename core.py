"""Numerical operations used by MatrixMaster."""
from __future__ import annotations

import numpy as np


class MatrixOperationError(ValueError):
    """Raised when a matrix operation cannot be completed."""


def as_matrix(values, name: str = "Matrix") -> np.ndarray:
    """Convert user-entered values into a finite 2-D NumPy array."""
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MatrixOperationError(f"{name} contains an invalid number.") from exc

    if matrix.ndim != 2 or matrix.size == 0:
        raise MatrixOperationError(f"{name} must be a non-empty 2D matrix.")
    if not np.isfinite(matrix).all():
        raise MatrixOperationError(f"{name} must contain only finite numbers.")
    return matrix


def calculate(operation: str, a, b=None, scalar: float | None = None) -> np.ndarray | float | int:
    """Run a supported operation and return a NumPy result or scalar."""
    A = as_matrix(a, "Matrix A")
    B = as_matrix(b, "Matrix B") if b is not None else None

    if operation == "Add (A + B)":
        _same_shape(A, B, "Addition")
        return A + B
    if operation == "Subtract (A - B)":
        _same_shape(A, B, "Subtraction")
        return A - B
    if operation == "Multiply (A × B)":
        if A.shape[1] != B.shape[0]:
            raise MatrixOperationError(
                f"Multiplication needs columns of A ({A.shape[1]}) to equal rows of B ({B.shape[0]})."
            )
        return A @ B
    if operation == "Element-wise (A ⊙ B)":
        _same_shape(A, B, "Element-wise multiplication")
        return A * B
    if operation == "Transpose A":
        return A.T
    if operation == "Transpose B":
        return _require_b(B).T
    if operation == "Determinant A":
        _square(A, "Determinant")
        return float(np.linalg.det(A))
    if operation == "Determinant B":
        B = _require_b(B)
        _square(B, "Determinant")
        return float(np.linalg.det(B))
    if operation == "Inverse A":
        _square(A, "Inverse")
        if np.isclose(np.linalg.det(A), 0.0):
            raise MatrixOperationError("Matrix A is singular, so its inverse does not exist.")
        return np.linalg.inv(A)
    if operation == "Inverse B":
        B = _require_b(B)
        _square(B, "Inverse")
        if np.isclose(np.linalg.det(B), 0.0):
            raise MatrixOperationError("Matrix B is singular, so its inverse does not exist.")
        return np.linalg.inv(B)
    if operation == "Rank A":
        return int(np.linalg.matrix_rank(A))
    if operation == "Rank B":
        return int(np.linalg.matrix_rank(_require_b(B)))
    if operation == "Trace A":
        _square(A, "Trace")
        return float(np.trace(A))
    if operation == "Trace B":
        B = _require_b(B)
        _square(B, "Trace")
        return float(np.trace(B))
    if operation == "Scalar × A":
        if scalar is None:
            raise MatrixOperationError("Enter a scalar value first.")
        return float(scalar) * A
    raise MatrixOperationError("Please choose a supported matrix operation.")


def _require_b(matrix):
    if matrix is None:
        raise MatrixOperationError("This operation requires Matrix B.")
    return matrix


def _same_shape(a, b, operation: str) -> None:
    b = _require_b(b)
    if a.shape != b.shape:
        raise MatrixOperationError(
            f"{operation} requires matrices of the same size. A is {a.shape[0]}×{a.shape[1]} "
            f"and B is {b.shape[0]}×{b.shape[1]}."
        )


def _square(matrix, operation: str) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise MatrixOperationError(f"{operation} requires a square matrix; current size is {matrix.shape[0]}×{matrix.shape[1]}.")
