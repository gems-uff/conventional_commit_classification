from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

STATISTICS_VERSION = "1.0.0"


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(pvals[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum()); w_neg = float(ranks[d < 0].sum())
    return 0.0 if (w_pos + w_neg) == 0 else (w_pos - w_neg) / (w_pos + w_neg)


def hedges_g_av(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y); x = x[mask]; y = y[mask]
    n = len(x)
    if n < 2:
        return float("nan")
    sd_av = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0)
    if sd_av == 0:
        return 0.0
    d_av = float(np.mean(x - y) / sd_av)
    df = 2 * n - 2
    j = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    return d_av * j


def kendall_w_friedman(chi2: float, n: int, k: int) -> float:
    return float(chi2 / (n * (k - 1))) if n > 0 and k > 1 else float("nan")


def paired_two_condition(x, y, *, alpha: float = 0.05) -> dict:
    """Study route for one paired contrast: Shapiro(delta) -> t or Wilcoxon."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired arrays must have equal shape")
    delta = x - y
    sw = stats.shapiro(delta)
    if float(sw.pvalue) >= alpha:
        test = stats.ttest_rel(x, y)
        return {
            "normality_W": float(sw.statistic), "normality_p": float(sw.pvalue),
            "test": "paired_t", "statistic": float(test.statistic), "p": float(test.pvalue),
            "effect": "hedges_g_av", "effect_value": hedges_g_av(x, y),
            "mean_difference": float(np.mean(delta)),
        }
    if np.allclose(delta, 0):
        stat, p = 0.0, 1.0
    else:
        w = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        stat, p = float(w.statistic), float(w.pvalue)
    return {
        "normality_W": float(sw.statistic), "normality_p": float(sw.pvalue),
        "test": "wilcoxon_signed_rank", "statistic": stat, "p": p,
        "effect": "rank_biserial", "effect_value": rank_biserial_paired(x, y),
        "mean_difference": float(np.mean(delta)),
    }


def related_conditions(scores: pd.DataFrame, *, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frozen >2-related-conditions route used by the final RQ4 evaluation.

    Returns (normality, omnibus, pairwise). Pairwise tests are run only after a
    significant omnibus result and Holm-corrected over the full pairwise family.
    """
    if scores.shape[1] < 3:
        raise ValueError("related_conditions requires at least 3 conditions")
    if scores.isna().any().any():
        raise ValueError("Scores contain missing values")
    models = list(scores.columns); n, k = scores.shape
    normality = pd.DataFrame([
        {"condition": m, "W": float((s := stats.shapiro(scores[m].to_numpy(float))).statistic),
         "p": float(s.pvalue), "normality_rejected": bool(s.pvalue < alpha)}
        for m in models
    ])
    nonparam = bool(normality["normality_rejected"].any())
    omnibus_rows = []; pair_rows = []
    if nonparam:
        fr = stats.friedmanchisquare(*[scores[m].to_numpy(float) for m in models])
        p_omni = float(fr.pvalue)
        omnibus_rows.append({"route": "nonparametric", "test": "Friedman", "statistic": float(fr.statistic),
                             "p": p_omni, "effect": "Kendall_W", "effect_value": kendall_w_friedman(float(fr.statistic), n, k)})
        if p_omni < alpha:
            temp = []; raw = []
            for a, b in itertools.combinations(models, 2):
                x = scores[a].to_numpy(float); y = scores[b].to_numpy(float)
                if np.allclose(x-y, 0): stat, p = 0.0, 1.0
                else:
                    w = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                    stat, p = float(w.statistic), float(w.pvalue)
                raw.append(p); temp.append({"condition_a": a, "condition_b": b, "test": "Wilcoxon signed-rank",
                                            "statistic": stat, "p_raw": p, "effect": "rank_biserial",
                                            "effect_value": rank_biserial_paired(x, y), "mean_difference_a_minus_b": float(np.mean(x-y))})
            for row, adj in zip(temp, holm_adjust(raw)):
                row["p_holm"] = adj; row["significant_holm"] = bool(adj < alpha); pair_rows.append(row)
    else:
        try:
            import pingouin as pg
        except ImportError as exc:
            raise RuntimeError("Parametric related-conditions route requires pingouin") from exc
        wide = scores.copy(); wide.index.name = "project"
        long = wide.reset_index().melt(id_vars="project", var_name="condition", value_name="score")
        spher_w = spher_p = np.nan
        try:
            sph = pg.sphericity(wide)
            spher_w = float(getattr(sph, "W", sph[1] if len(sph) > 1 else np.nan))
            spher_p = float(getattr(sph, "pval", sph[-1]))
        except Exception:
            pass
        aov = pg.rm_anova(data=long, dv="score", within="condition", subject="project", detailed=True, effsize="ng2", correction=True)
        row = aov.iloc[0]
        p_unc = float(row.get("p-unc", np.nan)); p_gg = float(row.get("p-GG-corr", np.nan)) if "p-GG-corr" in aov.columns else np.nan
        eps = float(row.get("eps", np.nan)) if "eps" in aov.columns else np.nan
        use_gg = np.isfinite(spher_p) and spher_p < alpha and np.isfinite(p_gg)
        p_omni = p_gg if use_gg else p_unc
        omnibus_rows.append({"route": "parametric", "test": "Repeated-measures ANOVA (Greenhouse-Geisser corrected)" if use_gg else "Repeated-measures ANOVA",
                             "statistic": float(row.get("F", np.nan)), "p": p_omni, "p_uncorrected": p_unc, "p_gg": p_gg,
                             "sphericity_W": spher_w, "sphericity_p": spher_p, "gg_epsilon": eps,
                             "effect": "generalized_eta_squared", "effect_value": float(row.get("ng2", np.nan))})
        if p_omni < alpha:
            temp = []; raw = []
            for a, b in itertools.combinations(models, 2):
                x = scores[a].to_numpy(float); y = scores[b].to_numpy(float)
                t = stats.ttest_rel(x, y); p = float(t.pvalue); raw.append(p)
                temp.append({"condition_a": a, "condition_b": b, "test": "paired Student t-test", "statistic": float(t.statistic),
                             "p_raw": p, "effect": "Hedges_g_av", "effect_value": hedges_g_av(x, y),
                             "mean_difference_a_minus_b": float(np.mean(x-y))})
            for row, adj in zip(temp, holm_adjust(raw)):
                row["p_holm"] = adj; row["significant_holm"] = bool(adj < alpha); pair_rows.append(row)
    return normality, pd.DataFrame(omnibus_rows), pd.DataFrame(pair_rows)
