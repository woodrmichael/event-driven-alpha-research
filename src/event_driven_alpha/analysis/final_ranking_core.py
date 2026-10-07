"""Deterministic split, metric, inference, and decision primitives for the final ranking study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OuterSplit:
    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def expanding_walk_forward_splits(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "event_timestamp_utc",
    event_id_column: str = "event_id",
    folds: int = 5,
    initial_train_fraction: float = 0.35,
) -> list[OuterSplit]:
    """Build five contiguous expanding validation blocks without peeking at labels."""
    if folds < 1 or not 0 < initial_train_fraction < 1:
        raise ValueError("Invalid expanding split parameters")
    ordered = frame[[timestamp_column, event_id_column]].copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], utc=True, errors="coerce")
    if ordered[timestamp_column].isna().any() or ordered[event_id_column].isna().any():
        raise ValueError("Split keys cannot be missing")
    ordered = ordered.sort_values([timestamp_column, event_id_column], kind="mergesort")
    n = len(ordered)
    first_validation = max(1, int(np.floor(n * initial_train_fraction)))
    boundaries = np.linspace(first_validation, n, folds + 1, dtype=int)
    result: list[OuterSplit] = []
    for fold in range(folds):
        validation_positions = np.arange(boundaries[fold], boundaries[fold + 1])
        training_positions = np.arange(0, boundaries[fold])
        if len(training_positions) == 0 or len(validation_positions) == 0:
            raise ValueError(f"Empty fold {fold + 1}")
        train_index = ordered.index.to_numpy()[training_positions]
        validation_index = ordered.index.to_numpy()[validation_positions]
        validation_times = ordered.iloc[validation_positions][timestamp_column]
        result.append(
            OuterSplit(
                fold=fold + 1,
                train_index=train_index,
                validation_index=validation_index,
                validation_start=validation_times.min(),
                validation_end=validation_times.max(),
            )
        )
    return result


def purge_training_rows(
    frame: pd.DataFrame,
    split: OuterSplit,
    *,
    timestamp_column: str = "event_timestamp_utc",
    event_id_column: str = "event_id",
    duplicate_keys: Iterable[str] = (
        "headline_cluster_key",
        "ticker_headline_cluster_key",
        "event_duplicate_key",
    ),
    embargo_trading_days: int = 2,
    trading_dates: Iterable[pd.Timestamp] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply the frozen two-observed-trading-day embargo and duplicate purge."""
    train = frame.loc[split.train_index].copy()
    validation = frame.loc[split.validation_index].copy()
    train[timestamp_column] = pd.to_datetime(train[timestamp_column], utc=True, errors="coerce")
    validation[timestamp_column] = pd.to_datetime(validation[timestamp_column], utc=True, errors="coerce")
    reasons: dict[object, set[str]] = {idx: set() for idx in train.index}

    if trading_dates is None:
        minimum = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce").min().normalize()
        maximum = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce").max().normalize()
        all_dates = pd.bdate_range(minimum, maximum, tz="UTC").tolist()
    else:
        all_dates = sorted({pd.Timestamp(value).tz_convert("UTC").normalize() if pd.Timestamp(value).tzinfo else pd.Timestamp(value).tz_localize("UTC").normalize() for value in trading_dates})
    validation_date = split.validation_start.normalize()
    earlier = [date for date in all_dates if date < validation_date]
    embargo_dates = set(earlier[-embargo_trading_days:])
    train_dates = train[timestamp_column].dt.normalize()
    for idx in train.index[train_dates.isin(embargo_dates)]:
        reasons[idx].add("two_trading_day_embargo")

    present_keys = [key for key in duplicate_keys if key in frame]
    if not present_keys:
        raise ValueError("No duplicate-cluster purge key is available")
    for key in present_keys:
        validation_values = set(validation[key].dropna().astype(str))
        if not validation_values:
            continue
        shared = train[key].notna() & train[key].astype(str).isin(validation_values)
        for idx in train.index[shared]:
            reasons[idx].add(f"shared_{key}")

    temporal = train[timestamp_column].ge(split.validation_start)
    for idx in train.index[temporal]:
        reasons[idx].add("not_strictly_before_validation")

    removed = [idx for idx, values in reasons.items() if values]
    audit = pd.DataFrame(
        [
            {
                "fold": split.fold,
                "event_id": frame.at[idx, event_id_column],
                "row_index": idx,
                "removed": True,
                "reason": "|".join(sorted(reasons[idx])),
            }
            for idx in removed
        ]
    )
    kept = train.index.difference(removed, sort=False).to_numpy()
    if len(kept) == 0:
        raise ValueError(f"Purge removed every training row in fold {split.fold}")
    max_train = pd.to_datetime(frame.loc[kept, timestamp_column], utc=True).max()
    min_validation = pd.to_datetime(frame.loc[split.validation_index, timestamp_column], utc=True).min()
    if not max_train < min_validation:
        raise ValueError(f"Temporal leakage remains in fold {split.fold}: {max_train} >= {min_validation}")
    return kept, audit


def spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 3:
        return float("nan")
    return float(pd.Series(actual[valid]).corr(pd.Series(predicted[valid]), method="spearman"))


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.clip(np.asarray(actual, dtype=float), 1e-10, None)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-10, None)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return float("nan")
    ratio = actual[valid] / predicted[valid]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def metric_deltas(actual: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "spearman_delta": spearman(actual, candidate) - spearman(actual, reference),
        "qlike_delta": qlike(actual, candidate) - qlike(actual, reference),
    }


def training_quantile_mapping(
    train_actual: np.ndarray,
    train_scores: np.ndarray,
    validation_actual: np.ndarray,
    validation_scores: np.ndarray,
    *,
    alert_rates: Iterable[float] = (0.005, 0.01, 0.02, 0.05),
) -> tuple[pd.DataFrame, float]:
    """Fit q95 and every score cutoff on training rows only."""
    train_actual = np.asarray(train_actual, dtype=float)
    train_scores = np.asarray(train_scores, dtype=float)
    validation_actual = np.asarray(validation_actual, dtype=float)
    validation_scores = np.asarray(validation_scores, dtype=float)
    q95 = float(np.nanquantile(train_actual, 0.95))
    rows = []
    base_rate = float(np.mean(validation_actual >= q95)) if len(validation_actual) else np.nan
    for rate in alert_rates:
        threshold = float(np.nanquantile(train_scores, 1.0 - rate))
        selected = validation_scores >= threshold
        positives = validation_actual >= q95
        precision = float(positives[selected].mean()) if selected.any() else np.nan
        capture = float(positives[selected].sum() / positives.sum()) if positives.any() else np.nan
        rows.append(
            {
                "alert_rate": float(rate),
                "score_threshold": threshold,
                "q95_threshold": q95,
                "selected_rows": int(selected.sum()),
                "q95_rows": int(positives.sum()),
                "precision": precision,
                "lift": precision / base_rate if np.isfinite(precision) and base_rate > 0 else np.nan,
                "capture": capture,
            }
        )
    return pd.DataFrame(rows), q95


def clustered_rank_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    actual_column: str,
    reference_column: str,
    candidate_column: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Cluster-bootstrap the paired Spearman delta using fixed pooled OOF ranks.

    Each sampled cluster multiplicity becomes an observation weight.  Ranking is
    frozen once on the original paired OOF sample, making this a reproducible
    weighted cluster bootstrap of the prespecified pooled-rank estimand.
    """
    work = frame[[cluster_column, actual_column, reference_column, candidate_column]].dropna().copy()
    work["_y"] = work[actual_column].rank(method="average").astype(float)
    work["_r"] = work[reference_column].rank(method="average").astype(float)
    work["_c"] = work[candidate_column].rank(method="average").astype(float)
    codes, clusters = pd.factorize(work[cluster_column].astype(str), sort=True)
    k = len(clusters)
    if k < 2:
        return np.full(samples, np.nan)

    def summaries(x: np.ndarray) -> np.ndarray:
        y = work["_y"].to_numpy()
        return np.stack(
            [
                np.bincount(codes, minlength=k),
                np.bincount(codes, weights=x, minlength=k),
                np.bincount(codes, weights=y, minlength=k),
                np.bincount(codes, weights=x * x, minlength=k),
                np.bincount(codes, weights=y * y, minlength=k),
                np.bincount(codes, weights=x * y, minlength=k),
            ],
            axis=1,
        )

    ref = summaries(work["_r"].to_numpy())
    cand = summaries(work["_c"].to_numpy())
    rng = np.random.default_rng(seed)
    result = np.empty(samples, dtype=float)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        draws = rng.integers(0, k, size=(size, k))
        weights = np.apply_along_axis(lambda row: np.bincount(row, minlength=k), 1, draws)

        def correlations(summary: np.ndarray) -> np.ndarray:
            totals = weights @ summary
            n, sx, sy, sxx, syy, sxy = totals.T
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            return cov / np.sqrt(vx * vy)

        result[start : start + size] = correlations(cand) - correlations(ref)
    return result


def clustered_paired_randomization(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    actual_column: str,
    reference_column: str,
    candidate_column: str,
    samples: int,
    seed: int,
) -> tuple[float, np.ndarray, float]:
    """Swap paired candidate/reference rank scores by cluster under the null."""
    work = frame[[cluster_column, actual_column, reference_column, candidate_column]].dropna().copy()
    work["_y"] = work[actual_column].rank(method="average").astype(float)
    work["_r"] = work[reference_column].rank(method="average").astype(float)
    work["_c"] = work[candidate_column].rank(method="average").astype(float)
    observed = spearman(work[actual_column].to_numpy(dtype=float), work[candidate_column].to_numpy(dtype=float)) - spearman(
        work[actual_column].to_numpy(dtype=float), work[reference_column].to_numpy(dtype=float)
    )
    codes, clusters = pd.factorize(work[cluster_column].astype(str), sort=True)
    k = len(clusters)
    if k < 2:
        return observed, np.full(samples, np.nan), float("nan")
    y = work["_y"].to_numpy(dtype=float)

    def summaries(x: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.bincount(codes, minlength=k),
                np.bincount(codes, weights=x, minlength=k),
                np.bincount(codes, weights=y, minlength=k),
                np.bincount(codes, weights=x * x, minlength=k),
                np.bincount(codes, weights=y * y, minlength=k),
                np.bincount(codes, weights=x * y, minlength=k),
            ],
            axis=1,
        )

    ref = summaries(work["_r"].to_numpy(dtype=float))
    cand = summaries(work["_c"].to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    null = np.empty(samples, dtype=float)

    def correlation(total: np.ndarray) -> float:
        n, sx, sy, sxx, syy, sxy = total
        return float((sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n)))

    for index in range(samples):
        swap = rng.integers(0, 2, size=k).astype(bool)
        perm_candidate = np.where(swap[:, None], ref, cand).sum(axis=0)
        perm_reference = np.where(swap[:, None], cand, ref).sum(axis=0)
        null[index] = correlation(perm_candidate) - correlation(perm_reference)
    p_greater = (1.0 + float(np.sum(null >= observed))) / (samples + 1.0)
    return observed, null, p_greater


def foldwise_label_permutation(
    frame: pd.DataFrame,
    *,
    fold_column: str,
    actual_column: str,
    score_column: str,
    samples: int,
    seed: int,
) -> tuple[float, np.ndarray, float]:
    """Absolute Spearman replication with labels shuffled only inside outer folds."""
    work = frame[[fold_column, actual_column, score_column]].dropna().copy()
    y_rank = work[actual_column].rank(method="average").to_numpy(dtype=float)
    x_rank = work[score_column].rank(method="average").to_numpy(dtype=float)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    observed = float(np.dot(x_centered, y_centered) / denominator)
    positions = [
        np.flatnonzero(work[fold_column].astype(str).to_numpy() == fold)
        for fold in sorted(work[fold_column].astype(str).unique())
    ]
    rng = np.random.default_rng(seed)
    null = np.empty(samples, dtype=float)
    shuffled = y_centered.copy()
    for index in range(samples):
        for location in positions:
            shuffled[location] = rng.permutation(y_centered[location])
        null[index] = float(np.dot(x_centered, shuffled) / denominator)
    p_greater = (1.0 + float(np.sum(null >= observed))) / (samples + 1.0)
    return observed, null, p_greater


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for position, idx in enumerate(order):
        running = max(running, (m - position) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def decide_claim(
    *,
    primary_delta: float,
    adjusted_p: float,
    ticker_ci: tuple[float, float],
    month_ci: tuple[float, float],
    fold_deltas: Iterable[float],
    validity_gates_pass: bool,
) -> str:
    folds = np.asarray(list(fold_deltas), dtype=float)
    if not validity_gates_pass or len(folds) != 5 or not np.isfinite(
        [primary_delta, adjusted_p, *ticker_ci, *month_ci, *folds]
    ).all():
        return "insufficient_or_invalid_evidence"
    positive = int((folds > 0).sum())
    negative = int((folds < 0).sum())
    if (
        primary_delta > 0
        and adjusted_p < 0.05
        and ticker_ci[0] > 0
        and month_ci[0] > 0
        and positive >= 4
    ):
        return "event_increment_supported"
    if (
        primary_delta < 0
        and adjusted_p < 0.05
        and ticker_ci[1] < 0
        and month_ci[1] < 0
        and negative >= 4
    ):
        return "event_increment_negative"
    return "event_increment_not_established"
