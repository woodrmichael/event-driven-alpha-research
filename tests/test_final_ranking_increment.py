from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from event_driven_alpha.analysis.final_ranking_increment import (
    COMMON_CATEGORICAL,
    FeatureVariant,
    OptionsCandidate,
    candidate_coverage,
    atomic_json,
    deduplicate_event_features,
    feature_variants,
    mask_sec_at_cutoff,
    paired_ablation_deltas,
    point_in_time_status,
    point_in_time_candidate_for_events,
    refuse_overwrite,
    require_frozen_specification,
    ridge_pipeline,
    use_pre_event_sec_cutoff,
)
from event_driven_alpha.analysis.final_ranking_core import (
    clustered_paired_randomization,
    clustered_rank_bootstrap,
    decide_claim,
    expanding_walk_forward_splits,
    holm_adjust,
    metric_deltas,
    purge_training_rows,
    training_quantile_mapping,
)
from event_driven_alpha.analysis.week6_volatility_causality import realized_stats_for_group


def test_frozen_specification_hash_matches() -> None:
    assert require_frozen_specification() == "4ede1f36fe41ad9b60af7890ef191e4b440369ebc725a797052c1755b81a105e"


def test_marketdata_snapshot_must_precede_event_date() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["good", "same_day", "missing"],
            "event_timestamp_utc": pd.to_datetime(
                ["2025-01-03T15:00:00Z", "2025-01-03T15:00:00Z", "2025-01-03T15:00:00Z"]
            ),
            "option_snapshot_date": ["2025-01-02", "2025-01-03", None],
            "option_snapshot_available_flag": [1, 1, 1],
            "atm_iv_30d": [0.2, 0.3, 0.4],
        }
    )
    candidate = OptionsCandidate("test", "MarketData", Path("unused"))
    audited = point_in_time_status(frame, candidate)
    assert audited["point_in_time_valid"].tolist() == [True, False, False]


def test_databento_quote_cannot_follow_feature_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["good", "future"],
            "event_timestamp_utc": pd.to_datetime(["2025-01-03T15:00:00Z"] * 2),
            "feature_cutoff_utc": pd.to_datetime(["2025-01-03T15:20:00Z"] * 2),
            "feature_quote_latest_ts_recv": pd.to_datetime(
                ["2025-01-03T15:19:59Z", "2025-01-03T15:20:01Z"]
            ),
            "feature_combined_mid": [2.0, 2.0],
            "feature_quote_uses_future_record": [False, False],
        }
    )
    candidate = OptionsCandidate("test", "Databento", Path("unused"))
    audited = point_in_time_status(frame, candidate)
    assert audited["point_in_time_valid"].tolist() == [True, False]


def test_candidate_coverage_uses_authoritative_label_timestamp(tmp_path: Path) -> None:
    option_path = tmp_path / "options.parquet"
    pd.DataFrame(
        {
            "event_id": ["e1"],
            "event_timestamp_utc": pd.to_datetime(["2025-01-03T15:01:00Z"]),
            "option_snapshot_date": ["2025-01-02"],
            "option_snapshot_available_flag": [1],
            "atm_iv_30d": [0.2],
        }
    ).to_parquet(option_path, index=False)
    labels = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["AAA"],
            "event_timestamp_utc": pd.to_datetime(["2025-01-03T15:00:00Z"]),
            "event_year": [2025],
            "event_date": ["2025-01-03"],
            "derived_event_family": ["earnings"],
            "source": ["test"],
        }
    )
    coverage, joined, leakage = candidate_coverage(
        OptionsCandidate("test", "MarketData", option_path), labels
    )
    assert len(joined) == 1
    assert joined.iloc[0]["event_timestamp_utc"] == labels.iloc[0]["event_timestamp_utc"]
    assert int(coverage.iloc[0]["point_in_time_valid_joined_events"]) == 1
    assert int(coverage.iloc[0]["unique_tickers"]) == 1
    assert int(coverage.iloc[0]["unique_ticker_date_pairs"]) == 1
    assert leakage["resolved_by_exclusion_or_mask"].all()


def test_model_option_join_uses_authoritative_event_timestamp(tmp_path: Path) -> None:
    option_path = tmp_path / "options.parquet"
    pd.DataFrame(
        {
            "event_id": ["e1"],
            "event_timestamp_utc": pd.to_datetime(["2025-01-04T15:00:00Z"]),
            "option_snapshot_date": ["2025-01-03"],
            "option_snapshot_available_flag": [1],
            "atm_iv_30d": [0.2],
        }
    ).to_parquet(option_path, index=False)
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "event_timestamp_utc": pd.to_datetime(["2025-01-03T15:00:00Z"]),
        }
    )
    joined = point_in_time_candidate_for_events(
        OptionsCandidate("test", "MarketData", option_path), events
    )
    assert joined.loc[0, "event_timestamp_utc"] == events.loc[0, "event_timestamp_utc"]
    assert not bool(joined.loc[0, "point_in_time_valid"])


def test_immutable_output_refuses_overwrite(tmp_path: Path) -> None:
    (tmp_path / "data_inventory.csv").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        refuse_overwrite(tmp_path, ["data_inventory.csv"])


def test_atomic_json_serializes_numpy_scalars(tmp_path: Path) -> None:
    path = tmp_path / "numpy.json"
    atomic_json({"valid": np.bool_(True), "count": np.int64(3)}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"count": 3, "valid": True}


def test_provider_deduplication_prefers_most_complete_event_row() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["e1", "e1", "e2"],
            "a": [1.0, 1.0, 3.0],
            "b": [pd.NA, 2.0, pd.NA],
        }
    )
    selected = deduplicate_event_features(frame, ["a", "b"]).set_index("event_id")
    assert len(selected) == 2
    assert selected.loc["e1", "b"] == 2.0


def test_sec_content_is_masked_when_acceptance_follows_cutoff() -> None:
    sec = pd.DataFrame(
        {
            "event_timestamp_utc": pd.to_datetime(["2025-01-03T15:00:00Z"] * 2),
            "nearest_sec_acceptance_datetime_utc": pd.to_datetime(
                ["2025-01-03T15:19:00Z", "2025-01-03T15:21:00Z"]
            ),
            "sec_8k_item_count": [2.0, 3.0],
            "sec_has_guidance_terms": [1.0, 1.0],
            "sec_form_type": ["8-K", "8-K"],
        }
    )
    masked = mask_sec_at_cutoff(sec, decision_lag_minutes=20)
    assert masked["sec_point_in_time_valid"].tolist() == [True, False]
    assert masked.loc[0, "sec_8k_item_count"] == 2.0
    assert pd.isna(masked.loc[1, "sec_8k_item_count"])
    assert pd.isna(masked.loc[1, "sec_form_type"])


def test_pre_event_sensitivity_uses_event_time_sec_values() -> None:
    frame = pd.DataFrame(
        {
            "sec_8k_item_count": [20.0],
            "pre_event_sec__sec_8k_item_count": [10.0],
            "sec_form_type": ["8-K"],
            "pre_event_sec__sec_form_type": [pd.NA],
        }
    )
    pre_event = use_pre_event_sec_cutoff(frame)
    assert pre_event.loc[0, "sec_8k_item_count"] == 10.0
    assert pd.isna(pre_event.loc[0, "sec_form_type"])


def test_frozen_common_controls_include_event_year() -> None:
    assert "event_year" in COMMON_CATEGORICAL


def test_broad_cohort_feature_blocks_include_explicit_availability_flags() -> None:
    frame = pd.DataFrame(
        {
            "lagged_available": [True],
            "reaction_available": [True],
            "provider_options_available": [True],
            "macro_available": [True],
            "sec_available": [True],
        }
    )
    variants = {variant.name: variant for variant in feature_variants(frame)}
    assert "lagged_available" in variants["A_lagged_market"].numeric
    assert "reaction_available" in variants["B_decision_market"].numeric
    assert "provider_options_available" in variants["C_strong_market_options"].numeric
    assert "macro_available" in variants["C_strong_market_options"].numeric
    assert "sec_available" in variants["D_strong_market_options_plus_structured_event"].numeric


def test_primary_preprocessing_is_inside_train_fitted_pipeline() -> None:
    pipeline = ridge_pipeline(
        FeatureVariant(
            "test",
            ("numeric",),
            ("categorical",),
            True,
        )
    )
    assert list(pipeline.named_steps) == ["features", "model"]
    transformer = pipeline.named_steps["features"]
    names = [name for name, _, _ in transformer.transformers]
    assert names == ["numeric", "categorical", "headline"]
    assert not hasattr(transformer, "transformers_")


def test_analysis_module_has_no_network_or_provider_sdk_imports() -> None:
    for path in (
        Path("src/event_driven_alpha/analysis/final_ranking_increment.py"),
        Path("src/event_driven_alpha/analysis/final_ranking_core.py"),
        Path("src/event_driven_alpha/analysis/final_ranking_inference.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert imported.isdisjoint({"requests", "httpx", "urllib", "socket", "databento", "marketdata"})


def synthetic_split_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02", periods=100, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "event_id": [f"e{idx:03d}" for idx in range(100)],
            "event_timestamp_utc": timestamps,
            "headline_cluster_key": [f"h{idx}" for idx in range(100)],
            "ticker_headline_cluster_key": [f"th{idx}" for idx in range(100)],
            "event_duplicate_key": [f"d{idx}" for idx in range(100)],
        }
    )


def test_expanding_splits_have_exact_paired_event_ids_and_temporal_order() -> None:
    frame = synthetic_split_frame()
    splits = expanding_walk_forward_splits(frame)
    assert len(splits) == 5
    assert [len(split.validation_index) for split in splits] == [13] * 5
    for split in splits:
        train, _ = purge_training_rows(frame, split)
        assert set(train).isdisjoint(split.validation_index)
        assert frame.loc[train, "event_timestamp_utc"].max() < frame.loc[split.validation_index, "event_timestamp_utc"].min()


def test_duplicate_cluster_purge_removes_shared_training_row() -> None:
    frame = synthetic_split_frame()
    splits = expanding_walk_forward_splits(frame)
    first_validation_index = splits[0].validation_index[0]
    frame.loc[0, "headline_cluster_key"] = frame.loc[first_validation_index, "headline_cluster_key"]
    kept, audit = purge_training_rows(frame, splits[0])
    assert 0 not in kept
    assert audit.loc[audit["row_index"].eq(0), "reason"].str.contains("headline_cluster_key").all()


def test_q95_and_score_thresholds_are_fit_from_training_only() -> None:
    rows, q95 = training_quantile_mapping(
        train_actual=[1, 2, 3, 4, 100],
        train_scores=[1, 2, 3, 4, 5],
        validation_actual=[1000, 2000],
        validation_scores=[6, 7],
        alert_rates=[0.2],
    )
    assert q95 == pytest.approx(80.8)
    assert rows.iloc[0]["score_threshold"] == pytest.approx(4.2)


def test_realized_variance_bars_begin_strictly_after_decision_timestamp() -> None:
    events = pd.DataFrame(
        {"event_timestamp_utc": pd.to_datetime(["2025-01-03T15:00:00Z"])}, index=[7]
    )
    bars = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2025-01-03T15:20:00Z", "2025-01-03T15:25:00Z"]
            ),
            "log_return": [1.0, 2.0],
            "high": [101.0, 102.0],
            "low": [99.0, 98.0],
        }
    )
    result = realized_stats_for_group(events, bars, decision_lag_minutes=20, horizon_minutes=60)
    assert result.loc[7, "post_bar_count"] == 1
    assert result.loc[7, "realized_variance"] == pytest.approx(4.0)


def test_metric_direction_is_candidate_minus_reference() -> None:
    actual = pd.Series([1.0, 2.0, 3.0, 4.0]).to_numpy()
    reference = pd.Series([4.0, 3.0, 2.0, 1.0]).to_numpy()
    candidate = actual.copy()
    deltas = metric_deltas(actual, reference, candidate)
    assert deltas["spearman_delta"] == pytest.approx(2.0)
    assert deltas["qlike_delta"] < 0


def test_paired_delta_collector_allows_prespecified_robustness_subset() -> None:
    rows = []
    for variant, scores in (
        ("C_strong_market_options", [1.0, 2.0, 3.0]),
        ("F_full_event", [1.0, 2.5, 4.0]),
    ):
        for index, score in enumerate(scores):
            rows.append(
                {
                    "cohort": "headline_shuffled_control",
                    "timing_variant": "delayed_20m",
                    "model_family": "ridge_tfidf",
                    "headline_mode": "shuffle_within_ticker_month_source",
                    "feature_variant": variant,
                    "event_id": f"e{index}",
                    "fold": 1,
                    "realized_variance": float(index + 1),
                    "predicted_realized_variance": score,
                }
            )
    deltas = paired_ablation_deltas(pd.DataFrame(rows))
    assert deltas["candidate_feature_variant"].tolist() == ["F_full_event"]
    assert deltas.loc[0, "rows"] == 3


def test_cluster_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B", "C", "C"],
            "actual": [1, 2, 3, 4, 5, 6],
            "reference": [2, 1, 4, 3, 6, 5],
            "candidate": [1, 2, 3, 4, 5, 6],
        }
    )
    one = clustered_rank_bootstrap(
        frame,
        cluster_column="ticker",
        actual_column="actual",
        reference_column="reference",
        candidate_column="candidate",
        samples=50,
        seed=42,
    )
    two = clustered_rank_bootstrap(
        frame,
        cluster_column="ticker",
        actual_column="actual",
        reference_column="reference",
        candidate_column="candidate",
        samples=50,
        seed=42,
    )
    assert one == pytest.approx(two)


def test_cluster_randomization_is_reproducible_and_paired() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B", "C", "C"],
            "actual": [1, 2, 3, 4, 5, 6],
            "reference": [2, 1, 4, 3, 6, 5],
            "candidate": [1, 2, 3, 4, 5, 6],
        }
    )
    observed_one, null_one, p_one = clustered_paired_randomization(
        frame,
        cluster_column="ticker",
        actual_column="actual",
        reference_column="reference",
        candidate_column="candidate",
        samples=40,
        seed=9,
    )
    observed_two, null_two, p_two = clustered_paired_randomization(
        frame,
        cluster_column="ticker",
        actual_column="actual",
        reference_column="reference",
        candidate_column="candidate",
        samples=40,
        seed=9,
    )
    assert observed_one == pytest.approx(observed_two)
    assert null_one == pytest.approx(null_two)
    assert p_one == pytest.approx(p_two)


def test_holm_and_claim_decision_logic() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    supported = decide_claim(
        primary_delta=0.02,
        adjusted_p=0.01,
        ticker_ci=(0.01, 0.03),
        month_ci=(0.005, 0.04),
        fold_deltas=[0.1, 0.1, 0.1, 0.1, -0.01],
        validity_gates_pass=True,
    )
    assert supported == "event_increment_supported"
    assert decide_claim(
        primary_delta=0.02,
        adjusted_p=0.2,
        ticker_ci=(-0.01, 0.03),
        month_ci=(-0.01, 0.04),
        fold_deltas=[0.1] * 5,
        validity_gates_pass=True,
    ) == "event_increment_not_established"
    assert decide_claim(
        primary_delta=0.02,
        adjusted_p=0.01,
        ticker_ci=(0.01, 0.03),
        month_ci=(0.01, 0.04),
        fold_deltas=[0.1] * 5,
        validity_gates_pass=False,
    ) == "insufficient_or_invalid_evidence"
