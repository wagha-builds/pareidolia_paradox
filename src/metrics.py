import numpy as np
from sklearn.metrics import balanced_accuracy_score


def apply_threshold(p, t):
    return (np.asarray(p) >= t).astype(int)


def plateau_threshold(y, p, grid=np.linspace(0.01, 0.99, 197), tol=0.001):
    ba = np.array([balanced_accuracy_score(y, apply_threshold(p, t)) for t in grid])
    ok = np.append(ba >= ba.max() - tol, False)
    best, start = (0, 0), None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best == (0, 0):
        # Fallback if no valid plateau found
        best_idx = np.argmax(ba)
        return float(grid[best_idx]), grid, ba
    return float((grid[best[0]] + grid[best[1] - 1]) / 2), grid, ba


def paired_bootstrap(y, p1, p2, group_ids, n_iter=1000, seed=42):
    """Group-level paired bootstrap on Balanced Accuracy."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(group_ids)
    diffs = []

    y = np.asarray(y)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    for _ in range(n_iter):
        sample_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        # Create mask for sampled groups
        mask = np.concatenate([np.where(group_ids == g)[0] for g in sample_groups])
        if len(np.unique(y[mask])) < 2:
            continue

        # We need thresholds here, assuming p1 and p2 are hard predictions or we threshold at 0.5
        # If they are probs, apply a default 0.5 threshold for this bootstrap unless threshold passed
        y_samp = y[mask]
        p1_samp = p1[mask] if p1.dtype != np.float64 else apply_threshold(p1[mask], 0.5)
        p2_samp = p2[mask] if p2.dtype != np.float64 else apply_threshold(p2[mask], 0.5)

        ba1 = balanced_accuracy_score(y_samp, p1_samp)
        ba2 = balanced_accuracy_score(y_samp, p2_samp)
        diffs.append(ba1 - ba2)

    diffs = np.array(diffs)
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
