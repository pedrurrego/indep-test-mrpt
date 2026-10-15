"""Implementation of Yanagimoto's inside parallelogram, as counterexample against Hoeffding's independence statistic."""

import numpy as np


def yanagimoto_inside_and_vertical_sweep_param(
    X, Y, base: float, height: float, x0: float, y0: float, tol: float = 1e-12
):
    """
    Parameters
    ----------
    X, Y : 1D numpy arrays of equal length
        Points sampled in [0,1]^2 (or any real values).
    base, height, x0, y0 : floats
        Parallelogram parameters defining vertices:
        A=(x0,y0), B=(x0+base,y0), C=(x0+2*base,y0+height), D=(x0+base,y0+height)
    tol : float
        Tolerance for boundary inclusion.

    Returns
    -------
    inside : boolean array
        True where (X[i], Y[i]) is inside/on the parallelogram.
    v_norm : float array
        Normalized vertical sweep in [0,1] for inside points; NaN outside.
        (0 at AB (bottom), 1 at CD (top).)
    y_swept : float array
        Swept y in original units: y0 + height * v_norm for inside points; NaN outside.
    Y_total : float array
        Final Y coordinates after moving points toward diagonals.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X.shape == Y.shape and X.ndim == 1, (
        "X and Y must be 1D arrays with the same shape."
    )

    # Vertices
    Ax, Ay = x0, y0
    Bx, By = x0 + base, y0
    Dx, Dy = x0 + base, y0 + height  # (vertex D; C is implied)

    # Parallelogram edge vectors from A
    e1 = np.array([Bx - Ax, By - Ay])  # A -> B  (base direction)
    e2 = np.array([Dx - Ax, Dy - Ay])  # A -> D  (slanted "vertical" direction)

    # Parallelogram columns in matrix M = [e1 e2]
    # For small 2x2 matrix it's cheaper to use closed-form inverse
    a = e1[0]
    c = e1[1]
    b = e2[0]
    d = e2[1]

    det = a * d - b * c
    # Fallback to np.linalg.inv for degenerate/near-degenerate matrices
    if abs(det) < 1e-20:
        M = np.column_stack((e1, e2))  # 2x2
        invM = np.linalg.inv(M)
        P = np.vstack((X - Ax, Y - Ay))  # shape (2, N)
        UV = invM @ P
        u, v = UV[0], UV[1]
    else:
        inv00 = d / det
        inv01 = -b / det
        inv10 = -c / det
        inv11 = a / det

        dx = X - Ax
        dy = Y - Ay
        u = inv00 * dx + inv01 * dy
        v = inv10 * dx + inv11 * dy

    # Inside test (with tolerance)
    inside = (u >= -tol) & (u <= 1 + tol) & (v >= -tol) & (v <= 1 + tol)

    # Vertical sweep (normalized) and y in original units
    v_norm = np.full_like(v, np.nan, dtype=float)
    y_swept = np.full_like(v, np.nan, dtype=float)

    v_norm[inside] = np.clip(v[inside], 0.0, 1.0)
    y_swept[inside] = y0 + height * v_norm[inside]

    Y_new = Y.astype(float).copy()

    slope = height / base
    y_diag_left = y0 + slope * (X - x0)
    y_diag_right = y0 + slope * (X - (x0 + base))
    # Move toward diagonal without crossing it,
    # left half moves upward, right half moves downward
    left_half = inside & (X >= x0 - tol) & (X <= x0 + base + tol)
    right_half = inside & (X >= x0 + base - tol) & (X <= x0 + 2 * base + tol)

    Y_new[left_half] = np.maximum(Y_new[left_half], y_diag_left[left_half])
    Y_new[right_half] = np.minimum(Y_new[right_half], y_diag_right[right_half])

    Y_total = np.where(inside, Y_new, Y)

    return inside, v_norm, y_swept, Y_total
