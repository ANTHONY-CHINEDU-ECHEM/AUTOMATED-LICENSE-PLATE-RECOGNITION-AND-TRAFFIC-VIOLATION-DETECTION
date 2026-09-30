"""Small statistical helpers shared by the analysis modules."""

from operator import sub

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg


def wilson_interval(successes, trials, z=1.96):
    """Wilson score interval for a binomial proportion (robust at small counts)."""
    if trials == 0:
        return (np.nan, np.nan)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denom
    margin = z * np.sqrt(p * sub(1, p) / trials + z ** 2 / (4 * trials ** 2)) / denom
    return (max(0.0, sub(centre, margin)), min(1.0, centre + margin))


def rate_table(df, group, flag, min_count=1):
    """Proportion of rows where flag is true per group, with Wilson intervals."""
    grouped = df.groupby(group, observed=True)[flag].agg(["sum", "count"])
    grouped = grouped[grouped["count"] >= min_count]
    rows = []
    for key, row in grouped.iterrows():
        low, high = wilson_interval(row["sum"], row["count"])
        rows.append(
            {
                group: key,
                "records": int(row["count"]),
                "events": int(row["sum"]),
                "rate_pct": round(row["sum"] / row["count"] * 100, 2),
                "ci_low_pct": round(low * 100, 2),
                "ci_high_pct": round(high * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def chi_square_independence(df, group, flag):
    """Chi square test that the flag rate is the same across groups."""
    table = pd.crosstab(df[group], df[flag].astype(bool))
    if table.shape[1] < 2 or table.shape[0] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "dof": 0, "cramers_v": np.nan}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape)
    cramers_v = np.sqrt(chi2 / (n * sub(k, 1))) if k > 1 else np.nan
    return {"chi2": round(float(chi2), 3), "p_value": float(p), "dof": int(dof),
            "cramers_v": round(float(cramers_v), 4)}


def kruskal_by_group(df, group, value):
    """Kruskal Wallis test that a continuous measure has the same distribution per group."""
    samples = [g[value].dropna().to_numpy() for _, g in df.groupby(group, observed=True)]
    samples = [s for s in samples if len(s) > 1]
    if len(samples) < 2:
        return {"h_statistic": np.nan, "p_value": np.nan, "epsilon_squared": np.nan}
    h, p = stats.kruskal(*samples)
    n = sum(len(s) for s in samples)
    epsilon_sq = h / sub(n, 1) if n > 1 else np.nan
    return {"h_statistic": round(float(h), 3), "p_value": float(p),
            "epsilon_squared": round(float(epsilon_sq), 5)}


def bootstrap_quantile_ci(values, quantile=0.5, iterations=cfg.BOOTSTRAP_ITERATIONS,
                          seed=cfg.RANDOM_STATE, level=0.95):
    """Percentile bootstrap confidence interval for a quantile of a sample."""
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for i in range(iterations):
        estimates[i] = np.quantile(rng.choice(values, size=len(values), replace=True), quantile)
    tail = sub(1, level) / 2
    return (float(np.quantile(estimates, tail)), float(np.quantile(estimates, sub(1, tail))))
