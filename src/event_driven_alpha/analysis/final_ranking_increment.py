"""Leakage-safe final event-information increment study.

The module is intentionally offline.  It reads only repository-local artifacts
and never imports a provider SDK or performs network I/O.  The audit stage is
kept separate from modeling so the supervised cohort and input provenance are
visible before any new model result is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from event_driven_alpha.analysis.week5_event_taxonomy import enrich_event_taxonomy
from event_driven_alpha.analysis.week6_volatility_causality import (
    MACRO_CATEGORICAL,
    MACRO_FLAGS,
    MACRO_NUMERIC,
    OPTIONS_FEATURES,
    SEC_CATEGORICAL,
    SEC_FLAGS,
    SEC_NUMERIC,
    SECTOR_EARLY_REACTION,
    SECTOR_PRE_EVENT,
    add_realized_volatility_targets,
)
from event_driven_alpha.analysis.final_ranking_core import (
    expanding_walk_forward_splits,
    metric_deltas,
    purge_training_rows,
    qlike,
    spearman,
    training_quantile_mapping,
)


RUN_ID = os.environ.get("FINAL_RANKING_RUN_ID", "final_ranking_increment_marketdata_20260809_v1")
SPECIFICATION = Path(f"configs/{RUN_ID}/research_specification.yaml")
SPECIFICATION_HASH = Path(f"configs/{RUN_ID}/research_specification.sha256")
DEFAULT_OUTPUT = Path(f"outputs/{RUN_ID}")
DEFAULT_REPORT = Path(f"reports/{RUN_ID}")
BASELINE = Path("data/processed/week5_direction_signal_features.parquet")
BAR_DIR = Path("data/raw/bars/5min")
SEC = Path("data/external/sec_event_features.parquet")
MACRO = Path("data/external/macro_regime_event_features.parquet")
SECTOR = Path("data/external/sector_reaction_event_features.parquet")
WEEK7_OPTIONS = Path("data/external/marketdata_options_event_features_week7_exact20486_20260706.parquet")
WEEK7_OPTIONS_CHAINS = Path("data/external/marketdata_options_chains_week7_exact20486_20260706.parquet")
NEWEST_MARKETDATA_OPTIONS = Path("data/external/marketdata_options_event_features_spend_7_30_20260718.parquet")
NEWEST_MARKETDATA_OPTIONS_CHAINS = Path("data/external/marketdata_options_chains_spend_7_30_20260718.parquet")
DATABENTO_V1_FEATURES = Path("data/processed/week8_databento_tradeability_20260712_v1/tradeability_features.parquet")
DATABENTO_V1_QUOTES = Path("data/processed/week8_databento_tradeability_20260712_v1/databento_exact_cmbp1_quotes.parquet")
DATABENTO_SAFE_PREFIX = Path(
    "data/processed/week9_benzinga_execution_aware_options_v3_20260714_v1/meeting_prefix/exact_option_state_features.parquet"
)
TAXONOMY_CODE = Path("src/event_driven_alpha/analysis/week5_event_taxonomy.py")

AUDIT_OUTPUTS = (
    "data_inventory.csv",
    "event_level_options_coverage.csv",
    "feature_availability_audit.csv",
    "point_in_time_leakage_audit.csv",
    "input_manifest.json",
    "audited_labeled_events.parquet",
)

MODEL_OUTPUTS = (
    "matched_oof_predictions.parquet",
    "ablation_metrics.csv",
    "preliminary_paired_metric_deltas.csv",
    "fold_results.csv",
    "queue_frontier.csv",
    "decile_calibration.csv",
    "subgroup_stability.csv",
    "concentration.csv",
    "split_leakage_audit.csv",
    "model_stage_manifest.json",
)

COMMON_CATEGORICAL = (
    "ticker",
    "event_hour_utc",
    "event_dayofweek_utc",
    "event_month",
    "event_year",
    "trading_session_bucket",
)
LAGGED_NUMERIC = (
    "pre_event_realized_vol_30m",
    "pre_event_volume_rel_30m",
    "pre_event_order_imbalance_proxy_30m",
    "prior_headline_cluster_count",
    "prior_ticker_headline_cluster_count",
    "minutes_since_same_ticker_cluster",
    "same_day_ticker_event_count_prior",
)
REACTION_NUMERIC = tuple(SECTOR_PRE_EVENT + SECTOR_EARLY_REACTION)
STRUCTURED_CATEGORICAL = (
    "source",
    "source_type",
    "event_route",
    "derived_event_family",
    "timestamp_precision",
    "timestamp_provenance",
    "timezone_assumption",
    "sec_form_type",
)


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    use_headline: bool


def unique_tuple(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def feature_variants(frame: pd.DataFrame, *, include_reaction: bool = True) -> list[FeatureVariant]:
    available = set(frame.columns)
    keep = lambda values: tuple(value for value in values if value in available)
    common = keep(COMMON_CATEGORICAL)
    lagged = keep(unique_tuple(LAGGED_NUMERIC, ("lagged_available",)))
    reaction = (
        keep(unique_tuple(REACTION_NUMERIC, ("reaction_available",))) if include_reaction else ()
    )
    options = keep(
        tuple(OPTIONS_FEATURES)
        + tuple(
            name
            for name in frame
            if name.startswith("feature_") and pd.api.types.is_numeric_dtype(frame[name])
        )
        + ("provider_options_available",)
    )
    macro = keep(tuple(MACRO_NUMERIC + MACRO_FLAGS) + ("macro_available",))
    market_categorical = unique_tuple(common, keep(tuple(MACRO_CATEGORICAL)))
    sec_numeric = keep(tuple(SEC_NUMERIC + SEC_FLAGS) + ("sec_available",))
    structured = keep(STRUCTURED_CATEGORICAL)
    base_a = lagged
    base_b = unique_tuple(base_a, reaction)
    base_c = unique_tuple(base_b, options, macro)
    return [
        FeatureVariant("A_lagged_market", base_a, common, False),
        FeatureVariant("B_decision_market", base_b, common, False),
        FeatureVariant("C_strong_market_options", base_c, market_categorical, False),
        FeatureVariant("D_strong_market_options_plus_structured_event", unique_tuple(base_c, sec_numeric), unique_tuple(market_categorical, structured), False),
        FeatureVariant("E_strong_market_options_plus_headline", base_c, market_categorical, True),
        FeatureVariant("F_full_event", unique_tuple(base_c, sec_numeric), unique_tuple(market_categorical, structured), True),
    ]


def ridge_pipeline(variant: FeatureVariant) -> Pipeline:
    transformers: list[tuple[str, object, object]] = []
    if variant.numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                list(variant.numeric),
            )
        )
    if variant.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(variant.categorical),
            )
        )
    if variant.use_headline:
        transformers.append(
            (
                "headline",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=60000,
                    sublinear_tf=True,
                    norm="l2",
                ),
                "headline",
            )
        )
    if not transformers:
        raise ValueError(f"No features are available for {variant.name}")
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("model", Ridge(alpha=5.0, solver="lsqr", tol=0.0001)),
        ]
    )


def hgb_pipeline(variant: FeatureVariant) -> Pipeline:
    numeric = list(variant.numeric)
    if not numeric:
        raise ValueError(f"No numeric features are available for {variant.name}")
    return Pipeline(
        [
            ("features", ColumnTransformer([("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric)])),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.04,
                    l2_regularization=0.05,
                    min_samples_leaf=40,
                    random_state=7,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class OptionsCandidate:
    name: str
    provider: str
    path: Path
    raw_support_path: Path | None = None
    safe_scope: str = "unrestricted_local"


OPTIONS_CANDIDATES = (
    OptionsCandidate("week7_exact20486", "MarketData", WEEK7_OPTIONS, raw_support_path=WEEK7_OPTIONS_CHAINS),
    OptionsCandidate(
        "marketdata_7_30_20260718",
        "MarketData",
        NEWEST_MARKETDATA_OPTIONS,
        raw_support_path=NEWEST_MARKETDATA_OPTIONS_CHAINS,
    ),
    OptionsCandidate(
        "databento_week8_v1",
        "Databento",
        DATABENTO_V1_FEATURES,
        raw_support_path=DATABENTO_V1_QUOTES,
        safe_scope="completed_v1_nonsealed",
    ),
    OptionsCandidate(
        "databento_v3_meeting_prefix",
        "Databento",
        DATABENTO_SAFE_PREFIX,
        safe_scope="meeting_prefix_only_no_v3_holdout_access",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(paths: Iterable[Path]) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(item)):
        file_hash = sha256_file(path)
        relative = str(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
        rows.append({"path": relative, "bytes": size, "sha256": file_hash})
    return digest.hexdigest(), rows


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows) if path.exists() else 0


def table_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names) if path.exists() else []


def require_frozen_specification() -> str:
    if not SPECIFICATION.exists() or not SPECIFICATION_HASH.exists():
        raise FileNotFoundError("Frozen research specification or hash receipt is missing")
    actual = sha256_file(SPECIFICATION)
    recorded = SPECIFICATION_HASH.read_text(encoding="utf-8").split()[0]
    if actual != recorded:
        raise ValueError(f"Research specification hash mismatch: expected {recorded}, observed {actual}")
    return actual


def frozen_primary_options_candidate() -> str | None:
    specification = yaml.safe_load(SPECIFICATION.read_text(encoding="utf-8"))
    cohort = specification.get("universe", {}).get("primary_cohort", {})
    candidate = cohort.get("primary_options_candidate")
    return str(candidate) if candidate else None


def refuse_overwrite(output_dir: Path, names: Iterable[str]) -> None:
    existing = [str(output_dir / name) for name in names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Immutable run artifacts already exist: {existing}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_base_with_labels() -> pd.DataFrame:
    base = pd.read_parquet(BASELINE)
    base = enrich_event_taxonomy(base) if "derived_event_family" not in base else base.copy()
    base["event_timestamp_utc"] = pd.to_datetime(base["event_timestamp_utc"], utc=True, errors="coerce")
    base = base[
        base.get("train_eligible", False).fillna(False).astype(bool)
        & base["event_id"].notna()
        & base["ticker"].notna()
        & base["event_timestamp_utc"].notna()
    ].copy()
    base = base.sort_values(["event_timestamp_utc", "event_id"]).drop_duplicates("event_id", keep="first")
    labeled = add_realized_volatility_targets(
        base,
        BAR_DIR,
        decision_lag_minutes=20,
        horizon_minutes=1440,
        min_post_bars=12,
    )
    labeled = labeled[labeled["realized_variance_available"].fillna(False).astype(bool)].copy()
    pre_event = add_realized_volatility_targets(
        base,
        BAR_DIR,
        decision_lag_minutes=0,
        horizon_minutes=1440,
        min_post_bars=12,
    )
    pre_event = pre_event[
        ["event_id", "post_bar_count", "realized_variance", "realized_volatility", "realized_variance_available"]
    ].rename(
        columns={
            "post_bar_count": "pre_event_variant_post_bar_count",
            "realized_variance": "pre_event_variant_realized_variance",
            "realized_volatility": "pre_event_variant_realized_volatility",
            "realized_variance_available": "pre_event_variant_realized_variance_available",
        }
    )
    labeled = labeled.merge(pre_event, on="event_id", how="left", validate="one_to_one")
    labeled["event_year"] = labeled["event_timestamp_utc"].dt.year.astype("Int64")
    labeled["event_date"] = labeled["event_timestamp_utc"].dt.date.astype(str)
    return labeled


def normalized_candidate(candidate: OptionsCandidate) -> pd.DataFrame:
    columns = table_columns(candidate.path)
    if not columns:
        return pd.DataFrame()
    wanted = {
        "event_id", "stable_event_id", "ticker", "event_timestamp_utc", "option_snapshot_date",
        "option_snapshot_available_flag", "feature_cutoff_utc", "score_timestamp_utc",
        "feature_quote_latest_ts_recv", "feature_quote_uses_future_record", "future_statistics_record_used",
        "source", "source_type", "derived_event_family", "source_canonical", "feature_status",
        "primary_tradeability_feature_eligible", "primary_contract_eligible",
    }
    wanted.update(OPTIONS_FEATURES)
    wanted.update(name for name in columns if name.startswith("feature_") and name not in {"feature_call_selected", "feature_put_selected"})
    frame = pd.read_parquet(candidate.path, columns=[name for name in columns if name in wanted])
    if "event_id" not in frame:
        if "stable_event_id" not in frame:
            raise ValueError(f"No event identifier in {candidate.path}")
        frame["event_id"] = frame["stable_event_id"]
    frame["event_id"] = frame["event_id"].astype(str)
    if "event_timestamp_utc" in frame:
        frame["event_timestamp_utc"] = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="coerce")
    return frame


def point_in_time_status(frame: pd.DataFrame, candidate: OptionsCandidate) -> pd.DataFrame:
    out = frame.copy()
    event_ts = pd.to_datetime(out.get("event_timestamp_utc"), utc=True, errors="coerce")
    cutoff = event_ts + pd.Timedelta(minutes=20)
    valid = pd.Series(True, index=out.index)
    reasons = pd.Series("valid", index=out.index, dtype="object")
    if candidate.provider == "MarketData":
        snapshot = pd.to_datetime(out.get("option_snapshot_date"), errors="coerce")
        event_day = event_ts.dt.tz_convert(None).dt.normalize()
        invalid = snapshot.isna() | event_day.isna() | snapshot.ge(event_day)
        valid &= ~invalid
        reasons.loc[invalid] = "snapshot_not_strictly_before_event_date"
        if "option_snapshot_available_flag" in out:
            unavailable = pd.to_numeric(out["option_snapshot_available_flag"], errors="coerce").fillna(0).ne(1)
            valid &= ~unavailable
            reasons.loc[unavailable] = "snapshot_unavailable"
        feature_cols = [name for name in OPTIONS_FEATURES if name in out]
        no_features = ~out[feature_cols].notna().any(axis=1) if feature_cols else pd.Series(True, index=out.index)
        valid &= ~no_features
        reasons.loc[no_features] = "no_populated_option_feature"
    else:
        if "feature_cutoff_utc" in out:
            feature_cutoff = pd.to_datetime(out["feature_cutoff_utc"], utc=True, errors="coerce")
            invalid = feature_cutoff.isna() | cutoff.isna() | feature_cutoff.gt(cutoff)
            valid &= ~invalid
            reasons.loc[invalid] = "feature_cutoff_after_frozen_cutoff"
        else:
            valid &= False
            reasons[:] = "feature_cutoff_missing"
        if "feature_quote_latest_ts_recv" in out:
            latest = pd.to_datetime(out["feature_quote_latest_ts_recv"], utc=True, errors="coerce")
            future = latest.notna() & pd.to_datetime(out["feature_cutoff_utc"], utc=True, errors="coerce").notna() & latest.gt(pd.to_datetime(out["feature_cutoff_utc"], utc=True, errors="coerce"))
            valid &= ~future
            reasons.loc[future] = "quote_after_feature_cutoff"
        for flag in ("feature_quote_uses_future_record", "future_statistics_record_used"):
            if flag in out:
                future = out[flag].fillna(False).astype(bool)
                valid &= ~future
                reasons.loc[future] = flag
        feature_cols = [name for name in out if name.startswith("feature_") and pd.api.types.is_numeric_dtype(out[name])]
        no_features = ~out[feature_cols].notna().any(axis=1) if feature_cols else pd.Series(True, index=out.index)
        valid &= ~no_features
        reasons.loc[no_features] = "no_populated_option_feature"
    out["point_in_time_valid"] = valid
    out["point_in_time_reason"] = reasons
    return out


def candidate_coverage(candidate: OptionsCandidate, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = normalized_candidate(candidate)
    if raw.empty:
        summary = pd.DataFrame([{"record_type": "summary", "candidate": candidate.name, "provider": candidate.provider, "status": "missing"}])
        return summary, pd.DataFrame(), pd.DataFrame()
    event = raw.sort_values([name for name in ("event_timestamp_utc", "event_id") if name in raw]).drop_duplicates("event_id", keep="last")
    event = event.rename(
        columns={
            name: f"option_{name}"
            for name in ("ticker", "event_timestamp_utc", "source", "derived_event_family")
            if name in event
        }
    )
    joined = labels[["event_id", "ticker", "event_timestamp_utc", "event_year", "event_date", "derived_event_family", "source"]].merge(
        event,
        on="event_id",
        how="inner",
        validate="one_to_one",
    )
    pit = point_in_time_status(joined, candidate)
    raw_ticker = raw.get("ticker", pd.Series(dtype="object"))
    ticker_from_join = len(raw_ticker) == 0 or not raw_ticker.notna().any()
    if ticker_from_join:
        raw_ticker = pit.get("ticker", pd.Series(dtype="object"))
    raw_event_ts = pd.to_datetime(
        pit.get("event_timestamp_utc") if ticker_from_join else raw.get("event_timestamp_utc"),
        utc=True,
        errors="coerce",
    )
    if raw_event_ts is None or not raw_event_ts.notna().any():
        raw_event_ts = pd.to_datetime(pit.get("event_timestamp_utc"), utc=True, errors="coerce")
    summary_row = {
        "record_type": "summary",
        "candidate": candidate.name,
        "provider": candidate.provider,
        "safe_scope": candidate.safe_scope,
        "status": "available",
        "raw_rows": parquet_rows(candidate.raw_support_path) if candidate.raw_support_path else int(len(raw)),
        "event_feature_rows": int(len(raw)),
        "supporting_raw_rows": parquet_rows(candidate.raw_support_path) if candidate.raw_support_path else int(len(raw)),
        "unique_events": int(raw["event_id"].nunique()),
        "unique_ticker_date_pairs": int(pd.DataFrame({"ticker": raw_ticker.reset_index(drop=True).astype(str), "date": raw_event_ts.reset_index(drop=True).dt.date.astype(str)}).drop_duplicates().shape[0]) if len(raw_ticker) == len(raw_event_ts) and len(raw_ticker) else 0,
        "unique_tickers": int(raw_ticker.nunique()) if len(raw_ticker) else 0,
        "date_min": raw_event_ts.min().isoformat() if raw_event_ts.notna().any() else None,
        "date_max": raw_event_ts.max().isoformat() if raw_event_ts.notna().any() else None,
        "joinable_labeled_events": int(pit["event_id"].nunique()),
        "point_in_time_valid_joined_events": int(pit.loc[pit["point_in_time_valid"], "event_id"].nunique()),
    }
    detail_rows: list[dict[str, Any]] = []
    for dimension, column in (("year", "event_year"), ("event_family", "derived_event_family"), ("source", "source")):
        for value, group in pit.groupby(column, dropna=False):
            detail_rows.append(
                {
                    "record_type": dimension,
                    "candidate": candidate.name,
                    "provider": candidate.provider,
                    "group": str(value),
                    "joinable_labeled_events": int(group["event_id"].nunique()),
                    "point_in_time_valid_joined_events": int(group.loc[group["point_in_time_valid"], "event_id"].nunique()),
                }
            )
    leakage = (
        pit.groupby(["point_in_time_valid", "point_in_time_reason"], dropna=False)
        .agg(rows=("event_id", "size"), unique_events=("event_id", "nunique"))
        .reset_index()
        .assign(candidate=candidate.name, provider=candidate.provider, audit_rule="options_point_in_time")
    )
    leakage["resolved_by_exclusion_or_mask"] = True
    return pd.concat([pd.DataFrame([summary_row]), pd.DataFrame(detail_rows)], ignore_index=True), pit, leakage


def point_in_time_candidate_for_events(
    candidate: OptionsCandidate, events: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate one provider row per event against the authoritative event time."""
    raw = normalized_candidate(candidate)
    if raw.empty:
        return raw
    sort_columns = [name for name in ("event_timestamp_utc", "event_id") if name in raw]
    selected = raw.sort_values(sort_columns, kind="mergesort").drop_duplicates(
        "event_id", keep="last"
    )
    if "event_timestamp_utc" in selected:
        selected = selected.rename(columns={"event_timestamp_utc": "option_event_timestamp_utc"})
    authoritative = events[["event_id", "event_timestamp_utc"]].copy()
    authoritative["event_id"] = authoritative["event_id"].astype(str)
    authoritative = authoritative.drop_duplicates("event_id", keep="first")
    joined = authoritative.merge(selected, on="event_id", how="inner", validate="one_to_one")
    return point_in_time_status(joined, candidate)


def availability_row(name: str, frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, Any]:
    present = [column for column in columns if column in frame]
    populated = frame[present].notna().any(axis=1) if present else pd.Series(False, index=frame.index)
    return {
        "feature_group": name,
        "rows": int(len(frame)),
        "available_columns": len(present),
        "requested_columns": len(list(columns)),
        "rows_with_any_feature": int(populated.sum()),
        "coverage_fraction": float(populated.mean()) if len(frame) else np.nan,
        "columns": "|".join(present),
    }


def deduplicate_event_features(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Choose one provider row per event by frozen completeness and source order."""
    out = frame.copy()
    out["event_id"] = out["event_id"].astype(str)
    present = [name for name in feature_columns if name in out]
    out["_available_feature_count"] = out[present].notna().sum(axis=1) if present else 0
    out["_source_row_order"] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values(
        ["event_id", "_available_feature_count", "_source_row_order"], kind="mergesort"
    ).drop_duplicates("event_id", keep="last")
    return out.drop(columns=["_available_feature_count", "_source_row_order"])


def load_modeling_frame(output_dir: Path) -> tuple[pd.DataFrame, str]:
    manifest_path = output_dir / "input_manifest.json"
    audited_path = output_dir / "audited_labeled_events.parquet"
    if not manifest_path.exists() or not audited_path.exists():
        raise FileNotFoundError("The verified audit stage must finish before modeling")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_name = str(manifest["primary_options_candidate"])
    candidate = next(item for item in OPTIONS_CANDIDATES if item.name == selected_name)
    base = pd.read_parquet(audited_path)
    base["event_id"] = base["event_id"].astype(str)
    base["event_timestamp_utc"] = pd.to_datetime(base["event_timestamp_utc"], utc=True, errors="coerce")
    base["event_hour_utc"] = base["event_timestamp_utc"].dt.hour
    base["event_dayofweek_utc"] = base["event_timestamp_utc"].dt.dayofweek
    base["event_month"] = base["event_timestamp_utc"].dt.month
    base["calendar_month"] = base["event_timestamp_utc"].dt.to_period("M").astype(str)
    base["trading_session_bucket"] = np.select(
        [
            base.get("is_premarket_event", False).fillna(False).astype(bool),
            base.get("is_regular_session_event", False).fillna(False).astype(bool),
            base.get("is_afterhours_event", False).fillna(False).astype(bool),
        ],
        ["premarket", "regular", "afterhours"],
        default=base.get("timing_regime", "unknown").fillna("unknown").astype(str),
    )

    option = point_in_time_candidate_for_events(candidate, base)
    option = option[option["point_in_time_valid"]]
    option_columns = ["event_id"] + [
        name
        for name in option
        if name in OPTIONS_FEATURES or (name.startswith("feature_") and pd.api.types.is_numeric_dtype(option[name]))
    ]
    option = deduplicate_event_features(option[option_columns], option_columns[1:])

    sec_raw = pd.read_parquet(SEC).rename(columns={"nearest_sec_form": "sec_form_type"})
    sec_raw["event_id"] = sec_raw["event_id"].astype(str)
    sec_features = [name for name in SEC_NUMERIC + SEC_FLAGS + SEC_CATEGORICAL if name in sec_raw]
    sec = mask_sec_at_cutoff(sec_raw, decision_lag_minutes=20)
    sec = sec.sort_values(["event_id", "nearest_sec_acceptance_datetime_utc"]).drop_duplicates("event_id", keep="last")
    sec = sec[["event_id"] + sec_features]
    pre_event_sec = mask_sec_at_cutoff(sec_raw, decision_lag_minutes=0)
    pre_event_sec = pre_event_sec.sort_values(
        ["event_id", "nearest_sec_acceptance_datetime_utc"]
    ).drop_duplicates("event_id", keep="last")
    pre_event_sec = pre_event_sec[["event_id"] + sec_features].rename(
        columns={name: f"pre_event_sec__{name}" for name in sec_features}
    )

    macro = pd.read_parquet(MACRO)
    macro["event_id"] = macro["event_id"].astype(str)
    macro_features = [name for name in MACRO_NUMERIC + MACRO_FLAGS + MACRO_CATEGORICAL if name in macro]
    macro = deduplicate_event_features(macro, macro_features)[["event_id"] + macro_features]

    sector = pd.read_parquet(SECTOR)
    sector["event_id"] = sector["event_id"].astype(str)
    sector_features = [name for name in SECTOR_PRE_EVENT + SECTOR_EARLY_REACTION if name in sector]
    sector = deduplicate_event_features(sector, sector_features)[["event_id"] + sector_features]

    frame = base.merge(option, on="event_id", how="left", validate="one_to_one")
    frame = frame.merge(sec, on="event_id", how="left", validate="one_to_one")
    frame = frame.merge(pre_event_sec, on="event_id", how="left", validate="one_to_one")
    frame = frame.merge(macro, on="event_id", how="left", validate="one_to_one")
    frame = frame.merge(sector, on="event_id", how="left", validate="one_to_one")
    frame["headline"] = frame["headline"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["provider_options_available"] = frame[[name for name in option_columns if name != "event_id"]].notna().any(axis=1)
    frame["macro_available"] = frame[macro_features].notna().any(axis=1)
    frame["reaction_available"] = frame[sector_features].notna().any(axis=1)
    frame["lagged_available"] = frame[[name for name in LAGGED_NUMERIC if name in frame]].notna().any(axis=1)
    frame["sec_available"] = frame[sec_features].notna().any(axis=1)
    frame["complete_case_common_eligible"] = (
        frame["provider_options_available"]
        & frame["macro_available"]
        & frame["reaction_available"]
        & frame["lagged_available"]
        & frame["realized_variance"].notna()
    )
    return frame.sort_values(["event_timestamp_utc", "event_id"]).reset_index(drop=True), selected_name


def use_pre_event_sec_cutoff(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace delayed-decision SEC values with event-time-safe values."""
    out = frame.copy()
    for name in SEC_NUMERIC + SEC_FLAGS + SEC_CATEGORICAL:
        pre_event_name = f"pre_event_sec__{name}"
        if pre_event_name in out:
            out[name] = out[pre_event_name]
    sec_features = [name for name in SEC_NUMERIC + SEC_FLAGS + SEC_CATEGORICAL if name in out]
    out["sec_available"] = out[sec_features].notna().any(axis=1)
    return out


def mask_sec_at_cutoff(sec: pd.DataFrame, *, decision_lag_minutes: int) -> pd.DataFrame:
    out = sec.copy()
    out["event_timestamp_utc"] = pd.to_datetime(out["event_timestamp_utc"], utc=True, errors="coerce")
    out["nearest_sec_acceptance_datetime_utc"] = pd.to_datetime(
        out["nearest_sec_acceptance_datetime_utc"], utc=True, errors="coerce"
    )
    out["_cutoff"] = out["event_timestamp_utc"] + pd.Timedelta(minutes=decision_lag_minutes)
    valid = out["nearest_sec_acceptance_datetime_utc"].notna() & out["nearest_sec_acceptance_datetime_utc"].le(out["_cutoff"])
    feature_columns = [name for name in SEC_NUMERIC + SEC_FLAGS + SEC_CATEGORICAL if name in out]
    for column in feature_columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype("Float64")
    out.loc[~valid, feature_columns] = np.nan
    out["sec_point_in_time_valid"] = valid
    return out


def _clip_predictions(train_log_target: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    low, high = np.quantile(train_log_target, [0.001, 0.999])
    return np.exp(np.clip(np.asarray(predictions, dtype=float), low, high)).clip(1e-10, None)


def frozen_trading_dates() -> list[pd.Timestamp]:
    spy_path = BAR_DIR / "SPY.csv"
    if not spy_path.exists():
        raise FileNotFoundError("SPY bar calendar is required for the two-trading-day embargo")
    timestamp = pd.read_csv(spy_path, usecols=["timestamp"])["timestamp"]
    return pd.to_datetime(timestamp, utc=True, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values().tolist()


def evaluate_matched_variants(
    frame: pd.DataFrame,
    *,
    cohort: str,
    timing_variant: str,
    model_family: str,
    variants_to_run: Iterable[str] | None = None,
    seed: int = 20260807,
    headline_mode: str = "original",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    include_reaction = timing_variant == "delayed_20m"
    variants = feature_variants(frame, include_reaction=include_reaction)
    if variants_to_run is not None:
        requested = set(variants_to_run)
        variants = [variant for variant in variants if variant.name in requested]
    target_column = "realized_variance" if timing_variant == "delayed_20m" else "pre_event_variant_realized_variance"
    working = frame.dropna(subset=[target_column, "event_timestamp_utc", "event_id", "ticker"]).copy()
    numeric_columns = sorted({name for variant in variants for name in variant.numeric})
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if working["event_id"].duplicated().any():
        raise ValueError("Modeling cohort contains duplicate event IDs")
    splits = expanding_walk_forward_splits(working)
    trading_dates = frozen_trading_dates()
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    queue_rows: list[pd.DataFrame] = []
    leakage_rows: list[pd.DataFrame] = []
    for split in splits:
        train_index, removed = purge_training_rows(working, split, trading_dates=trading_dates)
        validation_index = split.validation_index
        train = working.loc[train_index].copy()
        validation = working.loc[validation_index].copy()
        headline_changed_fraction = np.nan
        if headline_mode == "shuffle_within_ticker_month_source":
            original_train_headline = train["headline"].copy()
            for offset, part in enumerate((train, validation)):
                rng = np.random.default_rng(seed + split.fold * 100 + offset)
                shuffled = part["headline"].copy()
                strata = ["ticker", "calendar_month", "source"]
                for _, group in part.groupby(strata, dropna=False, sort=True):
                    shuffled.loc[group.index] = rng.permutation(group["headline"].to_numpy())
                part["headline"] = shuffled
            headline_changed_fraction = float((train["headline"] != original_train_headline).mean())
        elif headline_mode != "original":
            raise ValueError(f"Unsupported headline mode: {headline_mode}")
        if not removed.empty:
            removed = removed.assign(cohort=cohort, timing_variant=timing_variant, model_family=model_family)
            leakage_rows.append(removed)
        leakage_rows.append(
            pd.DataFrame(
                [
                    {
                        "fold": split.fold,
                        "event_id": None,
                        "row_index": None,
                        "removed": False,
                        "reason": "fold_temporal_gate_passed",
                        "cohort": cohort,
                        "timing_variant": timing_variant,
                        "model_family": model_family,
                        "headline_mode": headline_mode,
                        "train_rows_after_purge": len(train),
                        "validation_rows": len(validation),
                        "max_training_timestamp": train["event_timestamp_utc"].max(),
                        "min_validation_timestamp": validation["event_timestamp_utc"].min(),
                    }
                ]
            )
        )
        y_train_raw = train[target_column].to_numpy(dtype=float)
        y_validation = validation[target_column].to_numpy(dtype=float)
        y_train_log = np.log(np.clip(y_train_raw, 1e-10, None))
        for variant in variants:
            if model_family == "ridge_tfidf":
                model = ridge_pipeline(variant)
            elif model_family == "legacy_hgb_numeric":
                model = hgb_pipeline(variant)
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
            model.fit(train, y_train_log)
            train_prediction = _clip_predictions(y_train_log, model.predict(train))
            validation_prediction = _clip_predictions(y_train_log, model.predict(validation))
            queue, q95 = training_quantile_mapping(
                y_train_raw,
                train_prediction,
                y_validation,
                validation_prediction,
            )
            queue_rows.append(
                queue.assign(
                    cohort=cohort,
                    timing_variant=timing_variant,
                    model_family=model_family,
                    headline_mode=headline_mode,
                    feature_variant=variant.name,
                    fold=split.fold,
                )
            )
            prediction = validation[
                [
                    "event_id", "ticker", "event_timestamp_utc", "calendar_month", "source",
                    "derived_event_family", "headline",
                ]
            ].copy()
            prediction["cohort"] = cohort
            prediction["timing_variant"] = timing_variant
            prediction["model_family"] = model_family
            prediction["headline_mode"] = headline_mode
            prediction["feature_variant"] = variant.name
            prediction["fold"] = split.fold
            prediction["realized_variance"] = y_validation
            prediction["predicted_realized_variance"] = validation_prediction
            prediction["training_q95_threshold"] = q95
            prediction["realized_q95"] = y_validation >= q95
            decile_edges = np.quantile(train_prediction, np.linspace(0.0, 1.0, 11))
            prediction["score_decile"] = np.clip(
                np.searchsorted(decile_edges[1:-1], validation_prediction, side="right") + 1,
                1,
                10,
            )
            for alert_rate in (0.005, 0.01, 0.02, 0.05):
                score_cutoff = float(np.quantile(train_prediction, 1.0 - alert_rate))
                prediction[f"alert_{alert_rate:g}"] = validation_prediction >= score_cutoff
            predictions.append(prediction)
            folds.append(
                {
                    "cohort": cohort,
                    "timing_variant": timing_variant,
                    "model_family": model_family,
                    "headline_mode": headline_mode,
                    "feature_variant": variant.name,
                    "fold": split.fold,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "train_event_id_hash": hashlib.sha256("\n".join(sorted(train["event_id"].astype(str))).encode()).hexdigest(),
                    "validation_event_id_hash": hashlib.sha256("\n".join(sorted(validation["event_id"].astype(str))).encode()).hexdigest(),
                    "spearman": spearman(y_validation, validation_prediction),
                    "qlike": qlike(y_validation, validation_prediction),
                    "headline_changed_fraction": headline_changed_fraction,
                }
            )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(folds),
        pd.concat(queue_rows, ignore_index=True) if queue_rows else pd.DataFrame(),
        pd.concat(leakage_rows, ignore_index=True) if leakage_rows else pd.DataFrame(),
    )


def evaluate_cross_fitted_increment(
    frame: pd.DataFrame,
    *,
    cohort: str = "complete_case_common",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Condition event features on an outer-safe nonlinear market/options score."""
    working = frame.dropna(subset=["realized_variance", "event_timestamp_utc", "event_id", "ticker"]).copy()
    market_variant = next(item for item in feature_variants(working) if item.name == "C_strong_market_options")
    structured_variant = next(item for item in feature_variants(working) if item.name == "F_full_event")
    for column in sorted(set(market_variant.numeric) | set(structured_variant.numeric)):
        working[column] = pd.to_numeric(working[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    outer_splits = expanding_walk_forward_splits(working)
    trading_dates = frozen_trading_dates()
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    queue_rows: list[pd.DataFrame] = []
    audit_rows: list[pd.DataFrame] = []
    for outer in outer_splits:
        outer_train_index, outer_removed = purge_training_rows(working, outer, trading_dates=trading_dates)
        outer_train = working.loc[outer_train_index].copy()
        outer_validation = working.loc[outer.validation_index].copy()
        outer_train["_cross_fitted_market_score"] = np.nan
        inner_splits = expanding_walk_forward_splits(outer_train, folds=4)
        for inner in inner_splits:
            inner_train_index, inner_removed = purge_training_rows(outer_train, inner, trading_dates=trading_dates)
            inner_train = outer_train.loc[inner_train_index]
            inner_validation = outer_train.loc[inner.validation_index]
            y_inner_log = np.log(np.clip(inner_train["realized_variance"].to_numpy(dtype=float), 1e-10, None))
            market_model = hgb_pipeline(market_variant)
            market_model.fit(inner_train, y_inner_log)
            score = _clip_predictions(y_inner_log, market_model.predict(inner_validation))
            outer_train.loc[inner.validation_index, "_cross_fitted_market_score"] = score
            if not inner_removed.empty:
                audit_rows.append(
                    inner_removed.assign(
                        cohort=cohort,
                        timing_variant="delayed_20m",
                        model_family="cross_fitted_hgb_increment",
                        headline_mode="original",
                        outer_fold=outer.fold,
                        inner_fold=inner.fold,
                    )
                )
        meta_train = outer_train.dropna(subset=["_cross_fitted_market_score"]).copy()
        if len(meta_train) < 100:
            raise ValueError(f"Insufficient inner OOF meta-training rows in outer fold {outer.fold}")
        y_outer_log = np.log(np.clip(outer_train["realized_variance"].to_numpy(dtype=float), 1e-10, None))
        full_market_model = hgb_pipeline(market_variant)
        full_market_model.fit(outer_train, y_outer_log)
        outer_validation["_cross_fitted_market_score"] = _clip_predictions(
            y_outer_log, full_market_model.predict(outer_validation)
        )
        common = tuple(name for name in COMMON_CATEGORICAL if name in working)
        reference_variant = FeatureVariant(
            "C_crossfit_hgb_reference",
            ("_cross_fitted_market_score",),
            common,
            False,
        )
        candidate_variant = FeatureVariant(
            "F_crossfit_hgb_plus_event",
            unique_tuple(
                ("_cross_fitted_market_score",),
                tuple(name for name in structured_variant.numeric if name not in set(market_variant.numeric)),
            ),
            unique_tuple(common, tuple(name for name in STRUCTURED_CATEGORICAL if name in working)),
            True,
        )
        y_meta = meta_train["realized_variance"].to_numpy(dtype=float)
        y_meta_log = np.log(np.clip(y_meta, 1e-10, None))
        y_validation = outer_validation["realized_variance"].to_numpy(dtype=float)
        for variant in (reference_variant, candidate_variant):
            meta_model = ridge_pipeline(variant)
            meta_model.fit(meta_train, y_meta_log)
            train_prediction = _clip_predictions(y_meta_log, meta_model.predict(meta_train))
            validation_prediction = _clip_predictions(y_meta_log, meta_model.predict(outer_validation))
            queue, q95 = training_quantile_mapping(y_meta, train_prediction, y_validation, validation_prediction)
            queue_rows.append(
                queue.assign(
                    cohort=cohort,
                    timing_variant="delayed_20m",
                    model_family="cross_fitted_hgb_increment",
                    headline_mode="original",
                    feature_variant=variant.name,
                    fold=outer.fold,
                )
            )
            prediction = outer_validation[
                ["event_id", "ticker", "event_timestamp_utc", "calendar_month", "source", "derived_event_family", "headline"]
            ].copy()
            prediction = prediction.assign(
                cohort=cohort,
                timing_variant="delayed_20m",
                model_family="cross_fitted_hgb_increment",
                headline_mode="original",
                feature_variant=variant.name,
                fold=outer.fold,
                realized_variance=y_validation,
                predicted_realized_variance=validation_prediction,
                training_q95_threshold=q95,
                realized_q95=y_validation >= q95,
            )
            decile_edges = np.quantile(train_prediction, np.linspace(0.0, 1.0, 11))
            prediction["score_decile"] = np.clip(
                np.searchsorted(decile_edges[1:-1], validation_prediction, side="right") + 1, 1, 10
            )
            for alert_rate in (0.005, 0.01, 0.02, 0.05):
                prediction[f"alert_{alert_rate:g}"] = validation_prediction >= float(
                    np.quantile(train_prediction, 1.0 - alert_rate)
                )
            predictions.append(prediction)
            fold_rows.append(
                {
                    "cohort": cohort,
                    "timing_variant": "delayed_20m",
                    "model_family": "cross_fitted_hgb_increment",
                    "headline_mode": "original",
                    "feature_variant": variant.name,
                    "fold": outer.fold,
                    "train_rows": len(meta_train),
                    "validation_rows": len(outer_validation),
                    "train_event_id_hash": hashlib.sha256("\n".join(sorted(meta_train["event_id"].astype(str))).encode()).hexdigest(),
                    "validation_event_id_hash": hashlib.sha256("\n".join(sorted(outer_validation["event_id"].astype(str))).encode()).hexdigest(),
                    "spearman": spearman(y_validation, validation_prediction),
                    "qlike": qlike(y_validation, validation_prediction),
                }
            )
        if not outer_removed.empty:
            audit_rows.append(
                outer_removed.assign(
                    cohort=cohort,
                    timing_variant="delayed_20m",
                    model_family="cross_fitted_hgb_increment",
                    headline_mode="original",
                    outer_fold=outer.fold,
                    inner_fold=np.nan,
                )
            )
        audit_rows.append(
            pd.DataFrame(
                [
                    {
                        "fold": outer.fold,
                        "event_id": None,
                        "row_index": None,
                        "removed": False,
                        "reason": "outer_and_inner_crossfit_temporal_gates_passed",
                        "cohort": cohort,
                        "timing_variant": "delayed_20m",
                        "model_family": "cross_fitted_hgb_increment",
                        "headline_mode": "original",
                        "outer_fold": outer.fold,
                        "inner_fold": None,
                        "train_rows_after_purge": len(outer_train),
                        "validation_rows": len(outer_validation),
                        "max_training_timestamp": outer_train["event_timestamp_utc"].max(),
                        "min_validation_timestamp": outer_validation["event_timestamp_utc"].min(),
                    }
                ]
            )
        )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.concat(queue_rows, ignore_index=True),
        pd.concat(audit_rows, ignore_index=True, sort=False),
    )


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    specification_sha256 = require_frozen_specification()
    refuse_overwrite(args.output_dir, AUDIT_OUTPUTS)
    required = [BASELINE, SEC, MACRO, SECTOR, TAXONOMY_CODE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local inputs missing: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    labels = load_base_with_labels()
    coverage_frames: list[pd.DataFrame] = []
    candidate_frames: dict[str, pd.DataFrame] = {}
    leakage_frames: list[pd.DataFrame] = []
    for candidate in OPTIONS_CANDIDATES:
        coverage, joined, leakage = candidate_coverage(candidate, labels)
        coverage_frames.append(coverage)
        candidate_frames[candidate.name] = joined
        if not leakage.empty:
            leakage_frames.append(leakage)
    coverage = pd.concat(coverage_frames, ignore_index=True, sort=False)
    summaries = coverage[coverage["record_type"].eq("summary")].copy()
    week7_valid = int(summaries.loc[summaries["candidate"].eq("week7_exact20486"), "point_in_time_valid_joined_events"].fillna(0).max())
    databento_valid = int(summaries.loc[summaries["provider"].eq("Databento"), "point_in_time_valid_joined_events"].fillna(0).max())
    material_threshold = int(np.ceil(week7_valid * 1.05))
    frozen_candidate = frozen_primary_options_candidate()
    if frozen_candidate:
        selected = summaries[summaries["candidate"].eq(frozen_candidate)]
        if selected.empty or int(selected["point_in_time_valid_joined_events"].fillna(0).max()) == 0:
            raise ValueError(f"Frozen primary options candidate is unavailable: {frozen_candidate}")
        selection = frozen_candidate
        selection_reason = (
            "Frozen correction uses the later expanded MarketData pull because its previously audited "
            "point-in-time-valid labeled-event coverage materially exceeds Week 7"
        )
    else:
        selection = "newer_databento" if databento_valid >= material_threshold and week7_valid > 0 else "week7_exact20486"
        selection_reason = (
            "Databento clears frozen 5% unique valid labeled-event increase gate"
            if selection == "newer_databento"
            else "Databento does not clear frozen 5% unique valid labeled-event increase gate; retain Week 7 primary"
        )
    coverage["primary_options_selection"] = selection
    coverage["primary_options_selection_reason"] = selection_reason
    coverage["week7_valid_labeled_events"] = week7_valid
    coverage["databento_best_valid_labeled_events"] = databento_valid
    coverage["material_increase_threshold_events"] = material_threshold

    sec = pd.read_parquet(SEC)
    macro = pd.read_parquet(MACRO)
    sector = pd.read_parquet(SECTOR)
    if selection in candidate_frames:
        primary_options_name = selection
    else:
        primary_options_name = summaries.loc[
            summaries["provider"].eq("Databento")
        ].sort_values("point_in_time_valid_joined_events", ascending=False).iloc[0]["candidate"]
    primary_options = candidate_frames[primary_options_name]
    primary_valid = primary_options.loc[primary_options["point_in_time_valid"]].copy()
    primary_option_features = [
        name
        for name in primary_valid
        if name in OPTIONS_FEATURES
        or (name.startswith("feature_") and pd.api.types.is_numeric_dtype(primary_valid[name]))
    ]
    primary_valid = deduplicate_event_features(primary_valid, primary_option_features)
    primary_valid_ids = set(primary_valid["event_id"].astype(str))
    labels["event_id"] = labels["event_id"].astype(str)
    option_feature_columns = [name for name in primary_options if name.startswith("feature_")] + OPTIONS_FEATURES
    primary_options_for_availability = primary_options.copy()
    primary_options_for_availability.loc[
        ~primary_options_for_availability["point_in_time_valid"],
        [name for name in option_feature_columns if name in primary_options_for_availability],
    ] = np.nan
    renamed_sec = sec.rename(columns={"nearest_sec_form": "sec_form_type"})
    sec_for_availability = mask_sec_at_cutoff(renamed_sec, decision_lag_minutes=20)
    pre_event_sec_for_availability = mask_sec_at_cutoff(renamed_sec, decision_lag_minutes=0)
    availability = pd.DataFrame(
        [
            availability_row("headline_and_event_universe", labels, ["headline", "event_family", "derived_event_family", "source", "source_type", "event_route"]),
            availability_row(
                "lagged_market_state",
                labels,
                [
                    "pre_event_realized_vol_30m",
                    "pre_event_volume_rel_30m",
                    "pre_event_order_imbalance_proxy_30m",
                    "prior_headline_cluster_count",
                    "prior_ticker_headline_cluster_count",
                    "minutes_since_same_ticker_cluster",
                    "same_day_ticker_event_count_prior",
                ],
            ),
            availability_row("sector_and_early_reaction", sector, SECTOR_PRE_EVENT + SECTOR_EARLY_REACTION),
            availability_row("macro", macro, MACRO_NUMERIC + MACRO_FLAGS + MACRO_CATEGORICAL),
            availability_row("sec_point_in_time", sec_for_availability, SEC_NUMERIC + SEC_FLAGS + SEC_CATEGORICAL),
            availability_row(
                "sec_point_in_time_pre_event",
                pre_event_sec_for_availability,
                SEC_NUMERIC + SEC_FLAGS + SEC_CATEGORICAL,
            ),
            availability_row("primary_options_point_in_time", primary_options_for_availability, option_feature_columns),
        ]
    )

    sec_event = sec[[name for name in ["event_id", "event_timestamp_utc", "nearest_sec_acceptance_datetime_utc"] if name in sec]].drop_duplicates("event_id")
    sec_event["event_timestamp_utc"] = pd.to_datetime(sec_event["event_timestamp_utc"], utc=True, errors="coerce")
    sec_event["nearest_sec_acceptance_datetime_utc"] = pd.to_datetime(sec_event.get("nearest_sec_acceptance_datetime_utc"), utc=True, errors="coerce")
    sec_event["cutoff"] = sec_event["event_timestamp_utc"] + pd.Timedelta(minutes=20)
    sec_event["violation"] = sec_event["nearest_sec_acceptance_datetime_utc"].notna() & sec_event["nearest_sec_acceptance_datetime_utc"].gt(sec_event["cutoff"])
    leakage_frames.append(
        pd.DataFrame(
            [
                {
                    "candidate": "sec",
                    "provider": "SEC",
                    "audit_rule": "acceptance_at_or_before_feature_cutoff",
                    "point_in_time_valid": not bool(sec_event["violation"].any()),
                    "point_in_time_reason": "acceptance_after_cutoff_mask_required",
                    "rows": int(sec_event["violation"].sum()),
                    "unique_events": int(sec_event.loc[sec_event["violation"], "event_id"].nunique()),
                    "resolved_by_exclusion_or_mask": True,
                }
            ]
        )
    )
    pre_event_sec_event = sec_event.copy()
    pre_event_sec_event["cutoff"] = pre_event_sec_event["event_timestamp_utc"]
    pre_event_sec_event["violation"] = (
        pre_event_sec_event["nearest_sec_acceptance_datetime_utc"].notna()
        & pre_event_sec_event["nearest_sec_acceptance_datetime_utc"].gt(pre_event_sec_event["cutoff"])
    )
    leakage_frames.append(
        pd.DataFrame(
            [
                {
                    "candidate": "sec_pre_event_sensitivity",
                    "provider": "SEC",
                    "audit_rule": "acceptance_at_or_before_event_timestamp",
                    "point_in_time_valid": not bool(pre_event_sec_event["violation"].any()),
                    "point_in_time_reason": "acceptance_after_event_timestamp_mask_required",
                    "rows": int(pre_event_sec_event["violation"].sum()),
                    "unique_events": int(
                        pre_event_sec_event.loc[pre_event_sec_event["violation"], "event_id"].nunique()
                    ),
                    "resolved_by_exclusion_or_mask": True,
                }
            ]
        )
    )
    leakage_frames.append(
        pd.DataFrame(
            [
                {
                    "candidate": "realized_variance_delayed_20m",
                    "provider": "local_5min_bars",
                    "audit_rule": "every_return_bar_timestamp_strictly_after_decision_timestamp",
                    "point_in_time_valid": True,
                    "point_in_time_reason": "searchsorted_side_right_at_event_plus_20_minutes",
                    "rows": 0,
                    "unique_events": 0,
                    "resolved_by_exclusion_or_mask": True,
                },
                {
                    "candidate": "realized_variance_pre_event_variant",
                    "provider": "local_5min_bars",
                    "audit_rule": "every_return_bar_timestamp_strictly_after_event_timestamp",
                    "point_in_time_valid": True,
                    "point_in_time_reason": "searchsorted_side_right_at_event_timestamp",
                    "rows": 0,
                    "unique_events": 0,
                    "resolved_by_exclusion_or_mask": True,
                },
            ]
        )
    )
    leakage = pd.concat(leakage_frames, ignore_index=True, sort=False)

    inventory_paths = [BASELINE, SEC, MACRO, SECTOR, TAXONOMY_CODE, SPECIFICATION]
    for candidate in OPTIONS_CANDIDATES:
        if candidate.path.exists():
            inventory_paths.append(candidate.path)
        if candidate.raw_support_path and candidate.raw_support_path.exists():
            inventory_paths.append(candidate.raw_support_path)
    inventory_rows = []
    manifest_inputs = []
    for path in inventory_paths:
        file_hash = sha256_file(path)
        row = {
            "artifact": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_hash,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "rows": parquet_rows(path) if path.suffix == ".parquet" else np.nan,
            "columns": len(table_columns(path)) if path.suffix == ".parquet" else np.nan,
        }
        inventory_rows.append(row)
        manifest_inputs.append(row)
    bar_hash, bar_files = sha256_directory(BAR_DIR.glob("*.csv"))
    inventory_rows.append(
        {
            "artifact": str(BAR_DIR),
            "bytes": sum(item["bytes"] for item in bar_files),
            "sha256": bar_hash,
            "modified_utc": None,
            "rows": np.nan,
            "columns": np.nan,
        }
    )
    inventory = pd.DataFrame(inventory_rows)
    inventory = pd.concat(
        [
            inventory,
            pd.DataFrame(
                [
                    {
                        "artifact": "cached_text_representations",
                        "bytes": 0,
                        "sha256": None,
                        "modified_utc": None,
                        "rows": 0,
                        "columns": 0,
                        "status": "none_located; primary uses train-only TF-IDF",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    lagged_columns = [name for name in LAGGED_NUMERIC if name in labels]
    lagged_available_ids = set(
        labels.loc[labels[lagged_columns].notna().any(axis=1), "event_id"].astype(str)
    )
    macro_columns = [name for name in MACRO_NUMERIC + MACRO_FLAGS + MACRO_CATEGORICAL if name in macro]
    macro_deduplicated = deduplicate_event_features(macro, macro_columns)
    macro_available_ids = set(
        macro_deduplicated.loc[
            macro_deduplicated[macro_columns].notna().any(axis=1), "event_id"
        ].astype(str)
    )
    reaction_columns = [name for name in SECTOR_PRE_EVENT + SECTOR_EARLY_REACTION if name in sector]
    sector_deduplicated = deduplicate_event_features(sector, reaction_columns)
    reaction_available_ids = set(
        sector_deduplicated.loc[
            sector_deduplicated[reaction_columns].notna().any(axis=1), "event_id"
        ].astype(str)
    )
    provider_common_ids = (
        set(labels["event_id"])
        & lagged_available_ids
        & macro_available_ids
        & reaction_available_ids
    )
    candidate_common_counts = {
        name: len(
            set(candidate_frame.loc[candidate_frame["point_in_time_valid"], "event_id"].astype(str))
            & provider_common_ids
        )
        for name, candidate_frame in candidate_frames.items()
    }
    coverage["common_cohort_rows"] = coverage["candidate"].map(candidate_common_counts)
    common_ids = primary_valid_ids & provider_common_ids
    audited_keep = [
        "event_id", "event_duplicate_key", "duplicate_group_key", "ticker", "event_timestamp_utc",
        "source", "source_type", "event_route", "event_family", "derived_event_family", "headline",
        "timestamp_precision", "timestamp_provenance", "timestamp_confidence", "timezone_assumption",
        "label_volnorm_pretrend_baseline", "post_bar_count", "realized_variance", "realized_volatility",
        "pre_event_variant_post_bar_count", "pre_event_variant_realized_variance",
        "pre_event_variant_realized_volatility", "pre_event_variant_realized_variance_available",
        "headline_cluster_key", "ticker_headline_cluster_key", "pre_event_realized_vol_30m",
        "pre_event_volume_rel_30m", "pre_event_order_imbalance_proxy_30m", "prior_headline_cluster_count",
        "prior_ticker_headline_cluster_count", "minutes_since_same_ticker_cluster",
        "same_day_ticker_event_count_prior", "is_regular_session_event", "is_premarket_event",
        "is_afterhours_event", "timing_regime",
        "event_year", "event_date",
    ]
    audited = labels[[name for name in audited_keep if name in labels]].copy()
    audited["primary_options_point_in_time_valid"] = audited["event_id"].isin(primary_valid_ids)
    audited["complete_case_common_eligible"] = audited["event_id"].isin(common_ids)

    manifest = {
        "run_id": RUN_ID,
        "created_at_utc": utc_now(),
        "stage": "data_and_cohort_audit",
        "specification_sha256": specification_sha256,
        "network_requests": 0,
        "paid_requests": 0,
        "sealed_holdout_paths_read": [],
        "primary_options_selection": selection,
        "primary_options_candidate": primary_options_name,
        "primary_options_selection_reason": selection_reason,
        "labeled_events": int(len(labels)),
        "common_cohort_rows": int(len(common_ids)),
        "out_of_time_2026_status": "retrospective_out_of_time_outcomes_previously_inspected",
        "out_of_time_2026_basis": "Week 7 repository reports already summarize labels through 2026-06-21",
        "inputs": manifest_inputs,
        "bar_directory": {"path": str(BAR_DIR), "sha256": bar_hash, "files": bar_files},
    }

    atomic_csv(inventory, args.output_dir / "data_inventory.csv")
    atomic_csv(coverage, args.output_dir / "event_level_options_coverage.csv")
    atomic_csv(availability, args.output_dir / "feature_availability_audit.csv")
    atomic_csv(leakage, args.output_dir / "point_in_time_leakage_audit.csv")
    atomic_json(manifest, args.output_dir / "input_manifest.json")
    atomic_parquet(audited, args.output_dir / "audited_labeled_events.parquet")
    return manifest


def pooled_ablation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["cohort", "timing_variant", "model_family", "headline_mode", "feature_variant"]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, dropna=False):
        actual = group["realized_variance"].to_numpy(dtype=float)
        predicted = group["predicted_realized_variance"].to_numpy(dtype=float)
        order = np.argsort(-predicted, kind="mergesort")[: min(250, len(group))]
        overall_mean = float(np.mean(actual))
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "rows": len(group),
                "events": int(group["event_id"].nunique()),
                "folds": int(group["fold"].nunique()),
                "spearman": spearman(actual, predicted),
                "qlike": qlike(actual, predicted),
                "top250_realized_variance_lift": float(np.mean(actual[order]) / overall_mean) if overall_mean > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_ablation_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    context = ["cohort", "timing_variant", "model_family", "headline_mode"]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(context, dropna=False):
        crossfit = str(dict(zip(context, keys)).get("model_family")) == "cross_fitted_hgb_increment"
        reference_name = "C_crossfit_hgb_reference" if crossfit else "C_strong_market_options"
        candidates = ("F_crossfit_hgb_plus_event",) if crossfit else (
            "D_strong_market_options_plus_structured_event",
            "E_strong_market_options_plus_headline",
            "F_full_event",
        )
        reference = group[group["feature_variant"].eq(reference_name)]
        if reference.empty:
            continue
        for candidate_name in candidates:
            candidate = group[group["feature_variant"].eq(candidate_name)]
            if candidate.empty:
                continue
            pair = reference[
                ["event_id", "fold", "realized_variance", "predicted_realized_variance"]
            ].merge(
                candidate[["event_id", "fold", "predicted_realized_variance"]],
                on=["event_id", "fold"],
                how="inner",
                validate="one_to_one",
                suffixes=("_reference", "_candidate"),
            )
            if len(pair) != len(reference) or len(pair) != len(candidate):
                raise ValueError(f"Paired event mismatch for {keys}, {candidate_name}")
            delta = metric_deltas(
                pair["realized_variance"].to_numpy(dtype=float),
                pair["predicted_realized_variance_reference"].to_numpy(dtype=float),
                pair["predicted_realized_variance_candidate"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    **dict(zip(context, keys)),
                    "reference_feature_variant": reference_name,
                    "candidate_feature_variant": candidate_name,
                    "rows": len(pair),
                    "event_id_hash": hashlib.sha256("\n".join(sorted(pair["event_id"].astype(str))).encode()).hexdigest(),
                    **delta,
                }
            )
    return pd.DataFrame(rows)


def decile_table(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "timing_variant", "model_family", "headline_mode", "feature_variant", "fold", "score_decile"]
    return (
        predictions.groupby(keys, dropna=False)
        .agg(
            rows=("event_id", "size"),
            mean_realized_variance=("realized_variance", "mean"),
            median_realized_variance=("realized_variance", "median"),
            q95_rate=("realized_q95", "mean"),
            mean_score=("predicted_realized_variance", "mean"),
        )
        .reset_index()
    )


def subgroup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions[
        predictions["cohort"].eq("complete_case_common")
        & predictions["timing_variant"].eq("delayed_20m")
        & predictions["model_family"].eq("ridge_tfidf")
        & predictions["headline_mode"].eq("original")
        & predictions["feature_variant"].isin(["C_strong_market_options", "F_full_event"])
    ]
    rows: list[dict[str, Any]] = []
    for dimension in ("event_year", "derived_event_family"):
        if dimension == "event_year":
            primary = primary.assign(event_year=primary["event_timestamp_utc"].dt.year)
        for value, group in primary.groupby(dimension, dropna=False):
            if group["event_id"].nunique() < 250:
                continue
            ref = group[group["feature_variant"].eq("C_strong_market_options")]
            cand = group[group["feature_variant"].eq("F_full_event")]
            pair = ref[["event_id", "realized_variance", "predicted_realized_variance"]].merge(
                cand[["event_id", "predicted_realized_variance"]],
                on="event_id",
                suffixes=("_reference", "_candidate"),
            )
            if len(pair) < 250:
                continue
            q95_positive_events = int(ref.loc[ref["realized_q95"].astype(bool), "event_id"].nunique())
            if q95_positive_events < 20:
                continue
            rows.append(
                {
                    "dimension": dimension,
                    "group": str(value),
                    "rows": len(pair),
                    "q95_positive_events": q95_positive_events,
                    **metric_deltas(
                        pair["realized_variance"].to_numpy(dtype=float),
                        pair["predicted_realized_variance_reference"].to_numpy(dtype=float),
                        pair["predicted_realized_variance_candidate"].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def concentration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions[
        predictions["cohort"].eq("complete_case_common")
        & predictions["timing_variant"].eq("delayed_20m")
        & predictions["model_family"].eq("ridge_tfidf")
        & predictions["headline_mode"].eq("original")
        & predictions["feature_variant"].eq("F_full_event")
    ].copy()
    selected = primary[primary["alert_0.02"]]
    rows = []
    for dimension in ("ticker", "calendar_month", "source", "derived_event_family"):
        counts = selected.groupby(dimension, dropna=False)["event_id"].nunique().sort_values(ascending=False)
        rows.append(
            {
                "dimension": dimension,
                "selected_events": int(counts.sum()),
                "clusters": int(len(counts)),
                "largest_cluster_events": int(counts.iloc[0]) if len(counts) else 0,
                "largest_cluster_share": float(counts.iloc[0] / counts.sum()) if counts.sum() else np.nan,
                "top5_cluster_share": float(counts.head(5).sum() / counts.sum()) if counts.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def redact_ticker_identity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["headline"] = [
        " ".join("[IDENTITY]" if token.strip(".,:;!?()[]{}'\"").upper() == str(ticker).upper() else token for token in str(headline).split())
        for headline, ticker in zip(out["headline"], out["ticker"])
    ]
    return out


def databento_v1_robustness_frame(frame: pd.DataFrame) -> pd.DataFrame:
    candidate = next(item for item in OPTIONS_CANDIDATES if item.name == "databento_week8_v1")
    exact = point_in_time_candidate_for_events(candidate, frame)
    exact = exact[exact["point_in_time_valid"]]
    feature_columns = [
        name for name in exact if name.startswith("feature_") and pd.api.types.is_numeric_dtype(exact[name])
    ]
    if not feature_columns:
        return frame.iloc[0:0].copy()
    out = frame.drop(columns=[name for name in OPTIONS_FEATURES if name in frame]).merge(
        exact[["event_id"] + feature_columns], on="event_id", how="inner", validate="one_to_one"
    )
    out["provider_options_available"] = out[feature_columns].notna().any(axis=1)
    out["complete_case_common_eligible"] = (
        out["provider_options_available"]
        & out["macro_available"]
        & out["reaction_available"]
        & out["lagged_available"]
        & out["realized_variance"].notna()
    )
    return out[out["complete_case_common_eligible"]].copy()


def build_models(args: argparse.Namespace) -> dict[str, Any]:
    specification_sha256 = require_frozen_specification()
    refuse_overwrite(args.output_dir, MODEL_OUTPUTS)
    frame, options_name = load_modeling_frame(args.output_dir)
    complete = frame[frame["complete_case_common_eligible"]].copy()
    broad = frame.copy()
    pre_event_complete = use_pre_event_sec_cutoff(complete)
    exact = complete[
        complete["timestamp_precision"].astype(str).str.lower().isin({"exact", "second", "minute"})
        | pd.to_numeric(complete["timestamp_confidence"], errors="coerce").ge(0.95)
    ].copy()
    redacted = redact_ticker_identity(complete)
    databento_robustness = databento_v1_robustness_frame(frame)
    tasks = [
        (complete, "complete_case_common", "delayed_20m", "ridge_tfidf", None, "original"),
        (broad, "broad_left_joined", "delayed_20m", "ridge_tfidf", None, "original"),
        (pre_event_complete, "complete_case_common", "pre_event_only", "ridge_tfidf", None, "original"),
        (exact, "exact_high_confidence", "delayed_20m", "ridge_tfidf", ["C_strong_market_options", "D_strong_market_options_plus_structured_event", "E_strong_market_options_plus_headline", "F_full_event"], "original"),
        (redacted, "identity_redacted", "delayed_20m", "ridge_tfidf", ["C_strong_market_options", "E_strong_market_options_plus_headline", "F_full_event"], "original"),
        (complete, "headline_shuffled_control", "delayed_20m", "ridge_tfidf", ["C_strong_market_options", "F_full_event"], "shuffle_within_ticker_month_source"),
        (complete, "complete_case_common", "delayed_20m", "legacy_hgb_numeric", None, "original"),
        (databento_robustness, "exact_databento_robustness", "delayed_20m", "ridge_tfidf", ["C_strong_market_options", "D_strong_market_options_plus_structured_event", "E_strong_market_options_plus_headline", "F_full_event"], "original"),
    ]
    prediction_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    queue_frames: list[pd.DataFrame] = []
    leakage_frames: list[pd.DataFrame] = []
    for task_frame, cohort, timing, family, variants, headline_mode in tasks:
        if len(task_frame) < 100:
            continue
        prediction, folds, queue, leakage = evaluate_matched_variants(
            task_frame,
            cohort=cohort,
            timing_variant=timing,
            model_family=family,
            variants_to_run=variants,
            headline_mode=headline_mode,
        )
        prediction_frames.append(prediction)
        fold_frames.append(folds)
        queue_frames.append(queue)
        leakage_frames.append(leakage)
    cross_prediction, cross_folds, cross_queue, cross_leakage = evaluate_cross_fitted_increment(complete)
    prediction_frames.append(cross_prediction)
    fold_frames.append(cross_folds)
    queue_frames.append(cross_queue)
    leakage_frames.append(cross_leakage)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["event_timestamp_utc"] = pd.to_datetime(predictions["event_timestamp_utc"], utc=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    queue = pd.concat(queue_frames, ignore_index=True)
    leakage = pd.concat(leakage_frames, ignore_index=True, sort=False)
    ablations = pooled_ablation_metrics(predictions)
    deltas = paired_ablation_deltas(predictions)
    deciles = decile_table(predictions)
    subgroups = subgroup_table(predictions)
    concentration = concentration_table(predictions)

    # Every feature ablation in a paired context must share validation IDs and fold.
    hash_counts = folds.groupby(["cohort", "timing_variant", "model_family", "headline_mode", "fold"])["validation_event_id_hash"].nunique()
    if (hash_counts != 1).any():
        raise ValueError("Feature variants do not use exact matched validation event IDs")
    if leakage[leakage["reason"].eq("fold_temporal_gate_passed")].empty:
        raise ValueError("No temporal leakage gate evidence was produced")

    atomic_parquet(predictions, args.output_dir / "matched_oof_predictions.parquet")
    atomic_csv(ablations, args.output_dir / "ablation_metrics.csv")
    atomic_csv(deltas, args.output_dir / "preliminary_paired_metric_deltas.csv")
    atomic_csv(folds, args.output_dir / "fold_results.csv")
    atomic_csv(queue, args.output_dir / "queue_frontier.csv")
    atomic_csv(deciles, args.output_dir / "decile_calibration.csv")
    atomic_csv(subgroups, args.output_dir / "subgroup_stability.csv")
    atomic_csv(concentration, args.output_dir / "concentration.csv")
    atomic_csv(leakage, args.output_dir / "split_leakage_audit.csv")
    manifest = {
        "run_id": RUN_ID,
        "stage": "matched_models",
        "created_at_utc": utc_now(),
        "specification_sha256": specification_sha256,
        "primary_options_candidate": options_name,
        "complete_case_rows": len(complete),
        "broad_rows": len(broad),
        "exact_databento_robustness_rows": len(databento_robustness),
        "exact_databento_robustness_minimum_to_model": 100,
        "predictions": len(predictions),
        "network_requests": 0,
        "paid_requests": 0,
        "sealed_holdout_paths_read": [],
    }
    atomic_json(manifest, args.output_dir / "model_stage_manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "model"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "audit":
        result = build_audit(args)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.stage == "model":
        result = build_models(args)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
