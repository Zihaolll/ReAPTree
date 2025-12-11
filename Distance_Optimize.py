import numpy as np
from scipy.optimize import minimize


def project_simplex(y):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho_idxs = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho_idxs.size == 0:
        theta = 0.0
    else:
        rho = rho_idxs[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(y - theta, 0.0)


def weighted_median(values, weights):
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() == 0:
        return np.median(vals)
    order = np.argsort(vals)
    vals_s = vals[order]
    w_s = w[order] / w.sum()
    cumsum = np.cumsum(w_s)
    idx = np.searchsorted(cumsum, 0.5)
    idx = min(idx, len(vals_s) - 1)
    return vals_s[idx]


def eval_objective(A, W, p, metric="l2", eps=1e-12):
    A = np.asarray(A, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if metric == "l2":
        per = np.sum((A - p) ** 2, axis=1)
        return np.sum(W * per)
    elif metric == "l1":
        per = np.sum(np.abs(A - p), axis=1)
        return np.sum(W * per)
    elif metric == "kl":

        log_ratio = np.log(p + eps) - np.log(A + eps)
        per = np.sum(p * log_ratio, axis=1)
        return np.sum(W * per)
    else:

        if callable(metric):

            dvec = metric(p, A)
            return float(np.sum(W * np.asarray(dvec)))
        else:
            raise ValueError("Unsupported metric")


def barycenter(A, W, metric="L2", eps=1e-12, use_numeric_fallback=True):
    A = np.asarray(A, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be 2D array with shape (k, n)")
    k, n = A.shape
    if W.shape[0] != k:
        raise ValueError("W length must match number of rows in A")
    if np.any(W < 0):
        raise ValueError("Weights must be non-negative for these analytic formulas")

    W_sum = W.sum()
    if W_sum == 0:
        raise ValueError("Sum of weights is zero. Provide non-zero weights.")
    W_norm = W / W_sum

    info = {"method": None, "message": ""}

    # === analytic cases ===
    if metric == "L2":
        mean = np.sum(W_norm[:, None] * A, axis=0)
        p_star = project_simplex(mean)
        f_star_normW = eval_objective(A, W_norm, p_star, metric="l2")
        f_star_origW = eval_objective(A, W, p_star, metric="l2")
        info["method"] = "analytic_l2"

    elif metric == "KL":

        logA = np.log(A + eps)
        geo = np.exp(np.sum(W_norm[:, None] * logA, axis=0))
        p_star = geo / np.sum(geo)
        f_star_normW = eval_objective(A, W_norm, p_star, metric="kl", eps=eps)
        f_star_origW = eval_objective(A, W, p_star, metric="kl", eps=eps)
        info["method"] = "analytic_kl"

    elif metric == "L1":

        med = np.zeros(n, dtype=float)
        for j in range(n):
            med[j] = weighted_median(A[:, j], W_norm)
        p_star = project_simplex(med)
        f_star_normW = eval_objective(A, W_norm, p_star, metric="l1")
        f_star_origW = eval_objective(A, W, p_star, metric="l1")
        info["method"] = "analytic_l1"

    else:

        if not use_numeric_fallback:
            raise ValueError("Unsupported metric and numeric fallback disabled.")

        # objective for optimizer: returns scalar using normalized weights
        def obj_for_opt(p_flat):
            p = np.asarray(p_flat)
            # numerical safety: enforce small non-negativity
            p = np.maximum(p, 0)
            p = p / p.sum()  # normalize in-case optimizer strays
            return eval_objective(A, W_norm, p, metric=metric, eps=eps)

        # start from weighted mean projected
        x0 = project_simplex(np.sum(W_norm[:, None] * A, axis=0))

        cons = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0})
        bounds = [(0.0, 1.0)] * n

        res = minimize(obj_for_opt, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 1000})
        if not res.success:
            info["message"] = f"numeric solver failed: {res.message}"
        else:
            info["message"] = "numeric solver success"

        p_star = np.maximum(res.x, 0.0)
        p_star = p_star / p_star.sum()
        f_star_normW = eval_objective(A, W_norm, p_star, metric=metric, eps=eps)
        f_star_origW = eval_objective(A, W, p_star, metric=metric, eps=eps)
        info["method"] = "numeric"

    return p_star, f_star_normW, f_star_origW, info
