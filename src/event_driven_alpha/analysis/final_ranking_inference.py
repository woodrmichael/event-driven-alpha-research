"""Paired inference and frozen claim decision for the final ranking study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from event_driven_alpha.analysis.final_ranking_core import (
    clustered_paired_randomization,
    clustered_rank_bootstrap,
    decide_claim,
    foldwise_label_permutation,
    holm_adjust,
    metric_deltas,
    qlike,
    spearman,
)
from event_driven_alpha.analysis.final_ranking_increment import (
    DEFAULT_OUTPUT,
    RUN_ID,
    atomic_csv,
    atomic_json,
    refuse_overwrite,
    require_frozen_specification,
    utc_now,
)


INFERENCE_OUTPUTS = (
    "paired_metric_deltas.csv",
    "cluster_intervals.csv",
    "permutation_tests.csv",
    "claim_decision.json",
    "inference_stage_manifest.json",
)
COMPARISONS = (
    "D_strong_market_options_plus_structured_event",
    "E_strong_market_options_plus_headline",
    "F_full_event",
)
REFERENCE = "C_strong_market_options"


def primary_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["event_timestamp_utc"] = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="coerce")
    return frame[
        frame["cohort"].eq("complete_case_common")
        & frame["timing_variant"].eq("delayed_20m")
        & frame["model_family"].eq("ridge_tfidf")
        & frame["headline_mode"].eq("original")
        & frame["feature_variant"].isin((REFERENCE,) + COMPARISONS)
    ].copy()


def paired_frame(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    reference = frame[frame["feature_variant"].eq(REFERENCE)].copy()
    candidate_frame = frame[frame["feature_variant"].eq(candidate)].copy()
    keep = [
        "event_id", "fold", "ticker", "calendar_month", "event_timestamp_utc", "source",
        "derived_event_family", "realized_variance", "realized_q95", "alert_0.02",
        "predicted_realized_variance",
    ]
    pair = reference[keep].merge(
        candidate_frame[["event_id", "fold", "alert_0.02", "predicted_realized_variance"]],
        on=["event_id", "fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )
    if len(pair) != len(reference) or len(pair) != len(candidate_frame):
        raise ValueError(f"Primary paired rows differ for {candidate}")
    return pair


def additive_cluster_bootstrap(
    pair: pd.DataFrame,
    *,
    cluster_column: str,
    value: np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
    codes, clusters = pd.factorize(pair[cluster_column].astype(str), sort=True)
    k = len(clusters)
    sums = np.bincount(codes, weights=value, minlength=k)
    counts = np.bincount(codes, minlength=k)
    rng = np.random.default_rng(seed)
    out = np.empty(samples, dtype=float)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        draws = rng.integers(0, k, size=(size, k))
        weights = np.apply_along_axis(lambda row: np.bincount(row, minlength=k), 1, draws)
        out[start : start + size] = (weights @ sums) / (weights @ counts)
    return out


def queue_precision_delta(pair: pd.DataFrame) -> float:
    positives = pair["realized_q95"].astype(bool).to_numpy()
    reference = pair["alert_0.02_reference"].astype(bool).to_numpy()
    candidate = pair["alert_0.02_candidate"].astype(bool).to_numpy()
    ref_precision = positives[reference].mean() if reference.any() else np.nan
    cand_precision = positives[candidate].mean() if candidate.any() else np.nan
    return float(cand_precision - ref_precision)


def queue_cluster_bootstrap(
    pair: pd.DataFrame,
    *,
    cluster_column: str,
    samples: int,
    seed: int,
) -> np.ndarray:
    codes, clusters = pd.factorize(pair[cluster_column].astype(str), sort=True)
    k = len(clusters)
    positive = pair["realized_q95"].astype(bool).to_numpy(dtype=float)
    reference = pair["alert_0.02_reference"].astype(bool).to_numpy(dtype=float)
    candidate = pair["alert_0.02_candidate"].astype(bool).to_numpy(dtype=float)
    summary = np.stack(
        [
            np.bincount(codes, weights=positive * reference, minlength=k),
            np.bincount(codes, weights=reference, minlength=k),
            np.bincount(codes, weights=positive * candidate, minlength=k),
            np.bincount(codes, weights=candidate, minlength=k),
        ],
        axis=1,
    )
    rng = np.random.default_rng(seed)
    result = np.empty(samples, dtype=float)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        draws = rng.integers(0, k, size=(size, k))
        weights = np.apply_along_axis(lambda row: np.bincount(row, minlength=k), 1, draws)
        ref_pos, ref_n, cand_pos, cand_n = (weights @ summary).T
        with np.errstate(divide="ignore", invalid="ignore"):
            result[start : start + size] = cand_pos / cand_n - ref_pos / ref_n
    return result


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    specification_sha256 = require_frozen_specification()
    refuse_overwrite(args.output_dir, INFERENCE_OUTPUTS)
    frame = primary_predictions(args.output_dir / "matched_oof_predictions.parquet")
    interval_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    pair_by_candidate: dict[str, pd.DataFrame] = {}
    for comparison_index, candidate in enumerate(COMPARISONS):
        pair = paired_frame(frame, candidate)
        pair_by_candidate[candidate] = pair
        actual = pair["realized_variance"].to_numpy(dtype=float)
        reference = pair["predicted_realized_variance_reference"].to_numpy(dtype=float)
        candidate_score = pair["predicted_realized_variance_candidate"].to_numpy(dtype=float)
        point = metric_deltas(actual, reference, candidate_score)
        point["queue_precision_delta_2pct"] = queue_precision_delta(pair)
        point_rows.append(
            {
                "cohort": "complete_case_common",
                "timing_variant": "delayed_20m",
                "model_family": "ridge_tfidf",
                "headline_mode": "original",
                "reference_feature_variant": REFERENCE,
                "candidate_feature_variant": candidate,
                "rows": len(pair),
                **point,
            }
        )
        comparison_p_values = []
        for cluster_index, cluster in enumerate(("ticker", "calendar_month")):
            seed = 20260809 + comparison_index * 1000 + cluster_index * 100
            spearman_boot = clustered_rank_bootstrap(
                pair,
                cluster_column=cluster,
                actual_column="realized_variance",
                reference_column="predicted_realized_variance_reference",
                candidate_column="predicted_realized_variance_candidate",
                samples=10000,
                seed=seed,
            )
            actual_clipped = np.clip(pair["realized_variance"].to_numpy(dtype=float), 1e-10, None)
            qlike_values = (
                actual_clipped
                / np.clip(candidate_score, 1e-10, None)
                - np.log(actual_clipped / np.clip(candidate_score, 1e-10, None))
                - 1.0
                - (
                    actual_clipped / np.clip(reference, 1e-10, None)
                    - np.log(actual_clipped / np.clip(reference, 1e-10, None))
                    - 1.0
                )
            )
            qlike_boot = additive_cluster_bootstrap(
                pair,
                cluster_column=cluster,
                value=qlike_values,
                samples=10000,
                seed=seed + 1,
            )
            queue_boot = queue_cluster_bootstrap(
                pair,
                cluster_column=cluster,
                samples=10000,
                seed=seed + 2,
            )
            for metric, observed, values in (
                ("spearman_delta", point["spearman_delta"], spearman_boot),
                ("qlike_delta", point["qlike_delta"], qlike_boot),
                ("queue_precision_delta_2pct", point["queue_precision_delta_2pct"], queue_boot),
            ):
                interval_rows.append(
                    {
                        "candidate_feature_variant": candidate,
                        "reference_feature_variant": REFERENCE,
                        "cluster": cluster,
                        "metric": metric,
                        "observed": observed,
                        "ci_low": float(np.nanquantile(values, 0.025)),
                        "ci_high": float(np.nanquantile(values, 0.975)),
                        "bootstrap_samples": 10000,
                        "clusters": int(pair[cluster].nunique()),
                        "events": len(pair),
                    }
                )
            observed, null, p_value = clustered_paired_randomization(
                pair,
                cluster_column=cluster,
                actual_column="realized_variance",
                reference_column="predicted_realized_variance_reference",
                candidate_column="predicted_realized_variance_candidate",
                samples=10000,
                seed=20260811 + comparison_index * 1000 + cluster_index * 100,
            )
            p_less = (1.0 + float(np.sum(null <= observed))) / (len(null) + 1.0)
            p_directional = p_value if observed >= 0 else p_less
            comparison_p_values.append(p_directional)
            permutation_rows.append(
                {
                    "test": "cluster_paired_candidate_reference_randomization",
                    "candidate_feature_variant": candidate,
                    "reference_feature_variant": REFERENCE,
                    "cluster": cluster,
                    "observed": observed,
                    "p_greater": p_value,
                    "p_less": p_less,
                    "p_directional_observed_sign": p_directional,
                    "null_ci_low": float(np.nanquantile(null, 0.025)),
                    "null_ci_high": float(np.nanquantile(null, 0.975)),
                    "permutations": 10000,
                }
            )
        raw_p_values.append(max(comparison_p_values))

    adjusted = holm_adjust(raw_p_values)
    for row, raw, corrected in zip(point_rows, raw_p_values, adjusted):
        row["randomization_p_conservative_max_cluster"] = raw
        row["holm_adjusted_p"] = corrected

    full_pair = pair_by_candidate["F_full_event"]
    full_rows = frame[frame["feature_variant"].eq("F_full_event")]
    absolute_observed, absolute_null, absolute_p = foldwise_label_permutation(
        full_rows,
        fold_column="fold",
        actual_column="realized_variance",
        score_column="predicted_realized_variance",
        samples=10000,
        seed=20260812,
    )
    permutation_rows.append(
        {
            "test": "absolute_spearman_label_permutation_secondary",
            "candidate_feature_variant": "F_full_event",
            "reference_feature_variant": None,
            "cluster": "within_outer_fold",
            "observed": absolute_observed,
            "p_greater": absolute_p,
            "null_ci_low": float(np.quantile(absolute_null, 0.025)),
            "null_ci_high": float(np.quantile(absolute_null, 0.975)),
            "permutations": 10000,
        }
    )

    full_point = next(row for row in point_rows if row["candidate_feature_variant"] == "F_full_event")
    interval = pd.DataFrame(interval_rows)
    ticker = interval[
        interval["candidate_feature_variant"].eq("F_full_event")
        & interval["cluster"].eq("ticker")
        & interval["metric"].eq("spearman_delta")
    ].iloc[0]
    month = interval[
        interval["candidate_feature_variant"].eq("F_full_event")
        & interval["cluster"].eq("calendar_month")
        & interval["metric"].eq("spearman_delta")
    ].iloc[0]
    fold_deltas = []
    for _, fold in full_pair.groupby("fold"):
        fold_deltas.append(
            spearman(fold["realized_variance"].to_numpy(dtype=float), fold["predicted_realized_variance_candidate"].to_numpy(dtype=float))
            - spearman(fold["realized_variance"].to_numpy(dtype=float), fold["predicted_realized_variance_reference"].to_numpy(dtype=float))
        )
    split_audit = pd.read_csv(args.output_dir / "split_leakage_audit.csv")
    point_in_time = pd.read_csv(args.output_dir / "point_in_time_leakage_audit.csv")
    temporal_gates = split_audit[split_audit["reason"].astype(str).str.contains("temporal_gate", na=False)]
    resolved = point_in_time.get(
        "resolved_by_exclusion_or_mask", pd.Series(False, index=point_in_time.index)
    ).fillna(False).astype(bool)
    invalid = ~point_in_time["point_in_time_valid"].fillna(False).astype(bool)
    unresolved_point_in_time = point_in_time[invalid & ~resolved]
    concentration = pd.read_csv(args.output_dir / "concentration.csv")
    concentration_gate_pass = (
        set(concentration.get("dimension", pd.Series(dtype=str)).astype(str))
        == {"ticker", "calendar_month", "source", "derived_event_family"}
        and concentration["selected_events"].gt(0).all()
        and concentration["clusters"].gt(1).all()
        and concentration["largest_cluster_share"].between(0.0, 1.0, inclusive="neither").all()
    )
    validity_gates_pass = bool(
        len(temporal_gates) >= 5
        and not point_in_time.empty
        and unresolved_point_in_time.empty
        and concentration_gate_pass
    )
    status = decide_claim(
        primary_delta=float(full_point["spearman_delta"]),
        adjusted_p=float(full_point["holm_adjusted_p"]),
        ticker_ci=(float(ticker["ci_low"]), float(ticker["ci_high"])),
        month_ci=(float(month["ci_low"]), float(month["ci_high"])),
        fold_deltas=fold_deltas,
        validity_gates_pass=validity_gates_pass,
    )
    decision = {
        "run_id": RUN_ID,
        "status": status,
        "created_at_utc": utc_now(),
        "primary_comparison": "F_full_event_minus_C_strong_market_options",
        "primary_spearman_delta": float(full_point["spearman_delta"]),
        "holm_adjusted_p": float(full_point["holm_adjusted_p"]),
        "ticker_cluster_95_ci": [float(ticker["ci_low"]), float(ticker["ci_high"])],
        "calendar_month_cluster_95_ci": [float(month["ci_low"]), float(month["ci_high"])],
        "fold_deltas": [float(value) for value in fold_deltas],
        "positive_folds": int(np.sum(np.asarray(fold_deltas) > 0)),
        "validity_gates_pass": validity_gates_pass,
        "concentration_gate_pass": bool(concentration_gate_pass),
        "concentration_gate_rule": (
            "all frozen dimensions present; nonempty 2% queue; more than one cluster; "
            "no single cluster supplies every selected event"
        ),
        "unresolved_point_in_time_violations": int(len(unresolved_point_in_time)),
        "queue_result_is_supporting_only": True,
    }
    paired_output = pd.DataFrame(point_rows)
    preliminary_path = args.output_dir / "preliminary_paired_metric_deltas.csv"
    if preliminary_path.exists():
        preliminary = pd.read_csv(preliminary_path)
        secondary = preliminary[
            ~(
                preliminary["cohort"].eq("complete_case_common")
                & preliminary["timing_variant"].eq("delayed_20m")
                & preliminary["model_family"].eq("ridge_tfidf")
                & preliminary["headline_mode"].eq("original")
            )
        ]
        paired_output = pd.concat([paired_output, secondary], ignore_index=True, sort=False)
    atomic_csv(interval, args.output_dir / "cluster_intervals.csv")
    atomic_csv(pd.DataFrame(permutation_rows), args.output_dir / "permutation_tests.csv")
    atomic_csv(paired_output, args.output_dir / "paired_metric_deltas.csv")
    atomic_json(decision, args.output_dir / "claim_decision.json")
    manifest = {
        "run_id": RUN_ID,
        "stage": "paired_inference",
        "created_at_utc": utc_now(),
        "specification_sha256": specification_sha256,
        "bootstrap_samples_per_cluster": 10000,
        "permutation_samples": 10000,
        "network_requests": 0,
        "paid_requests": 0,
        "sealed_holdout_paths_read": [],
    }
    atomic_json(manifest, args.output_dir / "inference_stage_manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    result = run_inference(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
