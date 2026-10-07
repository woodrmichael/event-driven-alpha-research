"""Week 6 volatility targets, uncertainty, and reverse-causality checks."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from event_driven_alpha.analysis.week4_local_data import normalize_bar_frame, read_table
from event_driven_alpha.analysis.week5_event_taxonomy import enrich_event_taxonomy


DEFAULT_BASELINE = Path("data/processed/week4_headline_hpc_modeling_dataset.parquet")
DEFAULT_BAR_DIR = Path("data/raw/bars/5min")
DEFAULT_OPTIONS = Path("data/external/marketdata_options_event_features_expand_high_event_next116.parquet")
DEFAULT_SEC = Path("data/external/sec_event_features.parquet")
DEFAULT_MACRO = Path("data/external/macro_regime_event_features.parquet")
DEFAULT_SECTOR = Path("data/external/sector_reaction_event_features.parquet")
DEFAULT_OUTPUT_DIR = Path("outputs/week6_volatility_causality")
DEFAULT_REPORT_DIR = Path("reports/week6_volatility_causality")

RETURN_TARGET = "market_adjusted_return_20m_to_1d"
VOLNORM_TARGET = "target_volnorm_abs_2sigma_20m_1d"
RV_FLOOR = 1e-10

BASELINE_CATEGORICAL = [
    "ticker",
    "source",
    "source_type",
    "event_route",
    "event_family",
    "derived_event_family",
    "timestamp_precision",
    "timestamp_provenance",
    "timezone_assumption",
]
BASELINE_NUMERIC = ["timestamp_confidence", "event_hour_utc", "event_dayofweek_utc", "event_month"]

OPTIONS_FEATURES = [
    "atm_iv_7d",
    "atm_iv_30d",
    "atm_iv_60d",
    "atm_iv_90d",
    "iv_term_slope_7d_30d",
    "iv_term_slope_30d_60d",
    "atm_straddle_implied_move_7d",
    "atm_straddle_implied_move_30d",
    "put_skew_25d_30d",
    "call_skew_25d_30d",
    "risk_reversal_25d_30d",
    "put_call_volume_ratio_30d",
    "put_call_oi_ratio_30d",
    "option_volume_total_30d",
    "option_open_interest_total_30d",
]
SEC_NUMERIC = ["sec_8k_item_count", "minutes_from_sec_filing", "sec_text_relevance_score"]
SEC_FLAGS = [
    "sec_filing_near_event_flag",
    "sec_primary_release_flag",
    "sec_form_8k_flag",
    "sec_form_6k_flag",
    "sec_has_ex99_earnings_release",
    "sec_has_guidance_terms",
    "sec_has_merger_terms",
    "sec_has_financing_terms",
    "sec_has_executive_change_terms",
    "sec_has_impairment_terms",
    "sec_has_restructuring_terms",
]
SEC_CATEGORICAL = ["sec_form_type"]
MACRO_NUMERIC = [
    "vix_level",
    "vix_change_5d",
    "spy_trend_20d",
    "qqq_trend_20d",
    "rate_change_5d",
    "yield_curve_level",
    "yield_curve_change_20d",
    "credit_proxy_trend",
]
MACRO_FLAGS = ["near_cpi_release_flag", "near_fomc_release_flag", "near_jobs_release_flag", "near_macro_release_60m"]
MACRO_CATEGORICAL = ["market_regime_20d"]
SECTOR_PRE_EVENT = ["pre_return_30m", "pre_return_2h"]
SECTOR_EARLY_REACTION = [
    "stock_return_0m_5m",
    "stock_return_0m_15m",
    "stock_return_0m_20m",
    "stock_volume_rel_5m",
    "stock_volume_rel_15m",
    "spy_return_0m_15m",
    "sector_etf_return_0m_15m",
    "stock_minus_spy_return_15m",
    "stock_minus_sector_return_15m",
    "beta_adjusted_initial_reaction_15m",
    "distance_from_intraday_high",
    "distance_from_intraday_low",
    "continuation_setup_score",
]
AVAILABILITY_FEATURES = ["has_options_features", "has_sec_features", "has_macro_features", "has_sector_reaction_features"]
LAGGED_STATE_NUMERIC = [
    "pre_event_realized_vol_30m",
    "pre_event_volume_rel_30m",
    "pre_event_order_imbalance_proxy_30m",
    "prior_headline_cluster_count",
    "prior_ticker_headline_cluster_count",
    "minutes_since_same_ticker_cluster",
    "same_day_ticker_event_count_prior",
]
SPECIALIST_FEATURE_SET_NAMES = ["baseline_only", "baseline_plus_options", "baseline_plus_sec_options_macro_sector"]
ROBUSTNESS_FEATURE_SET_NAMES = ["baseline_only", "baseline_plus_options", "baseline_plus_sec_options_macro_sector"]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    use_headline: bool = False


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("week6_vol_%Y%m%dT%H%M%SZ")


def event_key(frame: pd.DataFrame) -> list[str]:
    return ["event_id"] if "event_id" in frame.columns else ["ticker", "event_timestamp_utc", "headline"]


def first_per_event(frame: pd.DataFrame, sort_columns: list[str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    present = [column for column in (sort_columns or []) if column in out.columns]
    if present:
        out = out.sort_values(present, na_position="last")
    return out.drop_duplicates(event_key(out), keep="first")


def available_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def read_external(path: Path, keep: list[str], aliases: dict[str, str] | None = None, sort_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path).rename(columns=aliases or {})
    keys = event_key(frame)
    columns = keys + [column for column in keep if column in frame.columns and column not in keys]
    return first_per_event(frame[columns], sort_columns=sort_columns)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_timestamp_utc"] = pd.to_datetime(out["event_timestamp_utc"], utc=True, errors="coerce")
    out["event_hour_utc"] = out["event_timestamp_utc"].dt.hour
    out["event_dayofweek_utc"] = out["event_timestamp_utc"].dt.dayofweek
    out["event_month"] = out["event_timestamp_utc"].dt.month
    out["event_year"] = out["event_timestamp_utc"].dt.year
    return out


def mask_sec_features_after_cutoff(frame: pd.DataFrame, decision_lag_minutes: int = 20) -> pd.DataFrame:
    """Mask SEC context that was not provably available by the feature cutoff.

    ``nearest_sec_acceptance_datetime_utc`` is the point-in-time availability
    timestamp for the matched filing.  A missing timestamp is not evidence of
    availability, so SEC values are also masked in that case.  The timestamp
    and explicit status columns remain available for leakage auditing, but are
    never included in model feature sets.
    """
    out = frame.copy()
    if "nearest_sec_acceptance_datetime_utc" not in out.columns:
        out["sec_feature_cutoff_status"] = "sec_timestamp_unavailable"
        out["sec_future_cutoff_violation"] = 0
        return out
    event_time = pd.to_datetime(out.get("event_timestamp_utc"), utc=True, errors="coerce")
    acceptance = pd.to_datetime(out["nearest_sec_acceptance_datetime_utc"], utc=True, errors="coerce")
    cutoff = event_time + pd.Timedelta(minutes=decision_lag_minutes)
    future = acceptance.notna() & cutoff.notna() & acceptance.gt(cutoff)
    unverifiable = acceptance.isna()
    mask = future | unverifiable
    feature_columns = available_columns(out, SEC_NUMERIC + SEC_CATEGORICAL + SEC_FLAGS)
    if feature_columns:
        out.loc[mask, feature_columns] = np.nan
    if "has_sec_features" in out.columns:
        out.loc[mask, "has_sec_features"] = 0
    out["sec_future_cutoff_violation"] = future.astype(int)
    out["sec_feature_cutoff_status"] = np.select(
        [future, unverifiable],
        ["after_feature_cutoff_masked", "availability_timestamp_missing_masked"],
        default="available_by_feature_cutoff",
    )
    return out


def load_joined_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    messages: list[str] = []
    base = pd.read_parquet(args.baseline)
    base = enrich_event_taxonomy(base) if "derived_event_family" not in base.columns else base.copy()
    base = add_time_features(base)
    base["headline"] = base.get("headline", "").map(clean_text)

    options = read_external(args.options, OPTIONS_FEATURES + ["option_snapshot_date", "option_snapshot_available_flag"])
    if options.empty:
        messages.append(f"Options features missing or empty: {args.options}")
    else:
        option_values = options[available_columns(options, OPTIONS_FEATURES)].notna().any(axis=1)
        option_flag = pd.to_numeric(options.get("option_snapshot_available_flag", 1), errors="coerce").fillna(0).astype(int).eq(1)
        options["has_options_features"] = (option_values & option_flag).astype(int)

    sec = read_external(
        args.sec,
        SEC_NUMERIC + SEC_CATEGORICAL + SEC_FLAGS + ["nearest_sec_acceptance_datetime_utc", "best_match_flag"],
        aliases={"nearest_sec_form": "sec_form_type"},
        sort_columns=["event_id", "best_match_flag", "minutes_from_sec_filing"],
    )
    if sec.empty:
        messages.append(f"SEC features missing or empty: {args.sec}")
    else:
        sec["has_sec_features"] = 1

    macro = read_external(args.macro, MACRO_NUMERIC + MACRO_CATEGORICAL + MACRO_FLAGS)
    if macro.empty:
        messages.append(f"Macro features missing or empty: {args.macro}")
    else:
        macro["has_macro_features"] = 1

    sector = read_external(args.sector, SECTOR_PRE_EVENT + SECTOR_EARLY_REACTION)
    if sector.empty:
        messages.append(f"Sector reaction features missing or empty: {args.sector}")
    else:
        sector["has_sector_reaction_features"] = 1

    joined = base
    for ext in [options, sec, macro, sector]:
        if not ext.empty:
            joined = joined.merge(ext, on=event_key(joined), how="left")
    for flag in AVAILABILITY_FEATURES:
        joined[flag] = joined.get(flag, 0).fillna(0).astype(int)
    joined = mask_sec_features_after_cutoff(joined, decision_lag_minutes=getattr(args, "decision_lag_minutes", 20))
    if "option_snapshot_date" in joined.columns:
        snap = pd.to_datetime(joined["option_snapshot_date"], errors="coerce").dt.date
        event_date = joined["event_timestamp_utc"].dt.date
        future_option = snap.notna() & (snap >= event_date)
        if future_option.any():
            joined.loc[future_option, available_columns(joined, OPTIONS_FEATURES)] = np.nan
            joined.loc[future_option, "has_options_features"] = 0
    return joined, messages


def load_bar_for_symbol(bar_dir: Path, ticker: str) -> pd.DataFrame:
    path = bar_dir / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    bars = normalize_bar_frame(read_table(path), symbol=ticker, source_file=str(path))
    if bars.empty:
        return bars
    bars = bars.sort_values("timestamp_utc").dropna(subset=["close"]).copy()
    close = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.loc[close.gt(0)].copy()
    bars["log_close"] = np.log(pd.to_numeric(bars["close"], errors="coerce"))
    bars["log_return"] = bars["log_close"].diff().replace([np.inf, -np.inf], np.nan)
    return bars.dropna(subset=["timestamp_utc", "log_return"])


def realized_stats_for_group(events: pd.DataFrame, bars: pd.DataFrame, decision_lag_minutes: int, horizon_minutes: int) -> pd.DataFrame:
    out = []
    times = bars["timestamp_utc"].to_numpy(dtype="datetime64[ns]")
    returns = pd.to_numeric(bars["log_return"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    for idx, event in events.iterrows():
        event_time = event["event_timestamp_utc"]
        start = event_time + pd.Timedelta(minutes=decision_lag_minutes)
        end = event_time + pd.Timedelta(minutes=horizon_minutes)
        left = np.searchsorted(times, np.datetime64(start.to_datetime64()), side="right")
        right = np.searchsorted(times, np.datetime64(end.to_datetime64()), side="right")
        window_returns = returns[left:right]
        valid_returns = window_returns[np.isfinite(window_returns)]
        rv = float(np.sum(valid_returns**2)) if len(valid_returns) else np.nan
        post_log_return = float(np.sum(valid_returns)) if len(valid_returns) else np.nan
        if len(valid_returns) >= 2:
            bv = float((math.pi / 2.0) * np.sum(np.abs(valid_returns[1:]) * np.abs(valid_returns[:-1])))
        else:
            bv = np.nan
        window_high = highs[left:right]
        window_low = lows[left:right]
        high = np.nanmax(window_high) if len(window_high) and np.isfinite(window_high).any() else np.nan
        low = np.nanmin(window_low) if len(window_low) and np.isfinite(window_low).any() else np.nan
        out.append(
            {
                "index": idx,
                "post_bar_count": int(len(valid_returns)),
                "realized_variance": rv,
                "realized_volatility": float(np.sqrt(rv)) if pd.notna(rv) else np.nan,
                "post_log_return": post_log_return,
                "absolute_underlying_return": (
                    float(abs(np.expm1(post_log_return)))
                    if pd.notna(post_log_return)
                    else np.nan
                ),
                "bipower_variation": bv,
                "jump_variance": max(rv - bv, 0.0) if pd.notna(rv) and pd.notna(bv) else np.nan,
                "jump_variance_share": max(rv - bv, 0.0) / rv if pd.notna(rv) and rv > 0 and pd.notna(bv) else np.nan,
                "post_high_low_log_range": float(np.log(high / low)) if pd.notna(high) and pd.notna(low) and high > 0 and low > 0 else np.nan,
            }
        )
    return pd.DataFrame(out).set_index("index")


def add_realized_volatility_targets(frame: pd.DataFrame, bar_dir: Path, decision_lag_minutes: int, horizon_minutes: int, min_post_bars: int) -> pd.DataFrame:
    pieces = []
    for ticker, events in frame.groupby(frame["ticker"].fillna("").astype(str).str.upper(), sort=False):
        bars = load_bar_for_symbol(bar_dir, ticker)
        if bars.empty:
            continue
        pieces.append(realized_stats_for_group(events, bars, decision_lag_minutes, horizon_minutes))
    out = frame.copy()
    stats = pd.concat(pieces, axis=0).sort_index() if pieces else pd.DataFrame()
    for column in [
        "post_bar_count",
        "realized_variance",
        "realized_volatility",
        "post_log_return",
        "absolute_underlying_return",
        "bipower_variation",
        "jump_variance",
        "jump_variance_share",
        "post_high_low_log_range",
    ]:
        out[column] = stats[column] if column in stats.columns else np.nan
    out["realized_variance_available"] = out["post_bar_count"].ge(min_post_bars) & pd.to_numeric(out["realized_variance"], errors="coerce").notna()
    valid_rv = pd.to_numeric(out.loc[out["realized_variance_available"], "realized_variance"], errors="coerce")
    threshold = float(valid_rv.quantile(0.75)) if valid_rv.notna().any() else np.nan
    out["target_top_quartile_realized_variance"] = (pd.to_numeric(out["realized_variance"], errors="coerce") >= threshold).astype(int)
    out.loc[~out["realized_variance_available"], "target_top_quartile_realized_variance"] = pd.NA
    out["jump_indicator"] = (
        pd.to_numeric(out["jump_variance_share"], errors="coerce").ge(0.5)
        & pd.to_numeric(out["jump_variance"], errors="coerce").gt(0)
        & out["realized_variance_available"]
    ).astype(int)
    out.loc[~out["realized_variance_available"], "jump_indicator"] = pd.NA
    out.attrs["top_quartile_realized_variance_threshold"] = threshold
    return out


def add_news_arrival_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["ticker", "event_timestamp_utc", "event_duplicate_key"], na_position="last").copy()
    out["next_same_ticker_articles_24h"] = 0
    out["next_same_ticker_clusters_24h"] = 0
    for _, group in out.groupby(out["ticker"].fillna("UNKNOWN").astype(str), sort=False):
        times = pd.to_datetime(group["event_timestamp_utc"], utc=True).to_numpy(dtype="datetime64[ns]")
        clusters = group.get("headline_cluster_key", pd.Series(group["event_id"], index=group.index)).fillna(group["event_id"]).astype(str).to_numpy()
        article_counts = []
        cluster_counts = []
        for pos, idx in enumerate(group.index):
            end = times[pos] + np.timedelta64(24, "h")
            right = np.searchsorted(times, end, side="right")
            article_counts.append(max(0, right - pos - 1))
            cluster_counts.append(len(set(clusters[pos + 1 : right])))
        out.loc[group.index, "next_same_ticker_articles_24h"] = article_counts
        out.loc[group.index, "next_same_ticker_clusters_24h"] = cluster_counts
    out["future_news_wave_24h"] = out["next_same_ticker_articles_24h"].ge(2).astype(int)
    out["future_cluster_wave_24h"] = out["next_same_ticker_clusters_24h"].ge(2).astype(int)
    return out.sort_values(["event_timestamp_utc", "event_duplicate_key"], na_position="last").reset_index(drop=True)


def modeling_frame(joined: pd.DataFrame, args: argparse.Namespace, mode: str) -> pd.DataFrame:
    frame = joined[joined.get("train_eligible", False).astype(bool)].copy()
    frame = frame.dropna(subset=["event_timestamp_utc", "headline", "ticker"]).copy()
    required_flags = [flag for flag in AVAILABILITY_FEATURES if flag in frame.columns]
    for flag in required_flags:
        frame = frame[frame[flag].eq(1)].copy()
    frame = add_realized_volatility_targets(
        frame,
        args.bar_dir,
        decision_lag_minutes=0 if mode == "immediate_event" else args.decision_lag_minutes,
        horizon_minutes=args.horizon_minutes,
        min_post_bars=args.min_post_bars,
    )
    frame = frame[frame["realized_variance_available"]].copy()
    frame = add_news_arrival_features(frame)
    frame = frame.sort_values(["event_timestamp_utc", "event_duplicate_key"], na_position="last").reset_index(drop=True)
    if args.max_rows > 0 and len(frame) > args.max_rows:
        frame = frame.sample(args.max_rows, random_state=args.seed).sort_values("event_timestamp_utc").reset_index(drop=True)
    return frame


def feature_sets_for(frame: pd.DataFrame, mode: str) -> list[FeatureSet]:
    sector_columns = SECTOR_PRE_EVENT + ([] if mode == "immediate_event" else SECTOR_EARLY_REACTION)
    return [
        FeatureSet("prior_only", (), (), False),
        FeatureSet("baseline_only", tuple(available_columns(frame, BASELINE_NUMERIC)), tuple(available_columns(frame, BASELINE_CATEGORICAL)), True),
        FeatureSet("headline_only", (), (), True),
        FeatureSet("baseline_plus_options", tuple(available_columns(frame, BASELINE_NUMERIC + OPTIONS_FEATURES)), tuple(available_columns(frame, BASELINE_CATEGORICAL)), True),
        FeatureSet("baseline_plus_sec", tuple(available_columns(frame, BASELINE_NUMERIC + SEC_NUMERIC + SEC_FLAGS)), tuple(available_columns(frame, BASELINE_CATEGORICAL + SEC_CATEGORICAL)), True),
        FeatureSet("baseline_plus_macro", tuple(available_columns(frame, BASELINE_NUMERIC + MACRO_NUMERIC + MACRO_FLAGS)), tuple(available_columns(frame, BASELINE_CATEGORICAL + MACRO_CATEGORICAL)), True),
        FeatureSet("baseline_plus_sector_reaction", tuple(available_columns(frame, BASELINE_NUMERIC + sector_columns)), tuple(available_columns(frame, BASELINE_CATEGORICAL)), True),
        FeatureSet(
            "baseline_plus_sec_options_macro_sector",
            tuple(available_columns(frame, BASELINE_NUMERIC + OPTIONS_FEATURES + SEC_NUMERIC + SEC_FLAGS + MACRO_NUMERIC + MACRO_FLAGS + sector_columns)),
            tuple(available_columns(frame, BASELINE_CATEGORICAL + SEC_CATEGORICAL + MACRO_CATEGORICAL)),
            True,
        ),
    ]


def lagged_causality_sets(frame: pd.DataFrame, mode: str) -> list[FeatureSet]:
    context = feature_sets_for(frame, mode)[-1]
    future_news = ["next_same_ticker_articles_24h", "next_same_ticker_clusters_24h", "future_news_wave_24h"]
    return [
        FeatureSet("lagged_market_state_only", tuple(available_columns(frame, LAGGED_STATE_NUMERIC + BASELINE_NUMERIC)), tuple(available_columns(frame, ["ticker", "derived_event_family"])), False),
        FeatureSet("headline_event_only", tuple(available_columns(frame, BASELINE_NUMERIC)), tuple(available_columns(frame, BASELINE_CATEGORICAL)), True),
        FeatureSet("lagged_state_plus_event_context", tuple(dict.fromkeys((*available_columns(frame, LAGGED_STATE_NUMERIC), *context.numeric))), context.categorical, True),
        FeatureSet("future_news_negative_control", tuple(available_columns(frame, LAGGED_STATE_NUMERIC + future_news + BASELINE_NUMERIC)), tuple(available_columns(frame, ["ticker", "derived_event_family"])), False),
    ]


def build_transformer(feature_set: FeatureSet, max_tfidf_features: int) -> ColumnTransformer:
    transformers = []
    if feature_set.numeric:
        transformers.append(("num", Pipeline([("imputer", median_imputer()), ("scale", StandardScaler(with_mean=False))]), list(feature_set.numeric)))
    if feature_set.categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(feature_set.categorical)))
    if feature_set.use_headline:
        transformers.append(("headline", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=5, max_features=max_tfidf_features), "headline"))
    if not transformers:
        transformers.append(("fallback", SimpleImputer(strategy="median"), ["event_hour_utc"]))
    return ColumnTransformer(transformers)


def numeric_only_feature_set(feature_set: FeatureSet, frame: pd.DataFrame) -> FeatureSet:
    numeric = feature_set.numeric
    if not numeric:
        numeric = tuple(available_columns(frame, BASELINE_NUMERIC))
    if not numeric:
        numeric = ("event_hour_utc",)
    return FeatureSet(f"{feature_set.name}_numeric", numeric, (), False)


def build_numeric_transformer(feature_set: FeatureSet) -> ColumnTransformer:
    transformers = []
    if feature_set.numeric:
        transformers.append(("num", median_imputer(), list(feature_set.numeric)))
    if not transformers:
        transformers.append(("fallback", median_imputer(), ["event_hour_utc"]))
    return ColumnTransformer(transformers)


def median_imputer() -> SimpleImputer:
    kwargs: dict[str, object] = {"strategy": "median"}
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        kwargs["keep_empty_features"] = True
    return SimpleImputer(**kwargs)


def build_regressor(feature_set: FeatureSet, max_tfidf_features: int, model_type: str = "ridge") -> object:
    if feature_set.name == "prior_only":
        return DummyRegressor(strategy="median")
    if model_type == "hgb":
        return Pipeline(
            [
                ("features", build_numeric_transformer(feature_set)),
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
    return Pipeline([("features", build_transformer(feature_set, max_tfidf_features)), ("model", Ridge(alpha=5.0))])


def build_classifier(feature_set: FeatureSet, max_tfidf_features: int, seed: int, model_type: str = "logistic") -> Pipeline:
    if feature_set.name == "prior_only":
        return Pipeline([("features", build_transformer(feature_set, max_tfidf_features)), ("model", DummyClassifier(strategy="prior", random_state=seed))])
    if model_type == "hgb":
        return Pipeline(
            [
                ("features", build_numeric_transformer(feature_set)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.04,
                        l2_regularization=0.05,
                        min_samples_leaf=40,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("features", build_transformer(feature_set, max_tfidf_features)),
            ("model", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=seed, max_iter=1000)),
        ]
    )


def classifier_metric_row(base: dict[str, object], y_true: np.ndarray, scores: np.ndarray, ks: list[int]) -> dict[str, object]:
    out = {
        **base,
        "status": "completed",
        "base_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
    }
    for k in ks:
        rate = topk_binary_rate(y_true, scores, k)
        out[f"top{k}_hit_rate"] = rate
        out[f"top{k}_lift"] = rate / out["base_rate"] if out["base_rate"] else np.nan
    return out


def chronological_splits(frame: pd.DataFrame, folds: int, test_fraction: float, min_train_fraction: float, embargo_minutes: int) -> list[dict[str, object]]:
    out = []
    n = len(frame)
    test_size = max(1, int(round(n * test_fraction)))
    min_train = max(test_size, int(round(n * min_train_fraction)))
    available = n - test_size
    for fold in range(folds):
        if available <= min_train:
            start = min_train
        else:
            step = max(1, (available - min_train) // max(1, folds - 1))
            start = min_train + step * fold
        start = min(start, n - 1)
        end = min(start + test_size, n)
        test_idx = frame.index[(frame.index >= start) & (frame.index < end)]
        if len(test_idx) == 0:
            continue
        embargo_start = frame.loc[test_idx, "event_timestamp_utc"].min() - pd.Timedelta(minutes=embargo_minutes)
        train_idx = frame.index[frame["event_timestamp_utc"] < embargo_start]
        out.append({"fold": fold, "split_id": f"walk_forward_{fold + 1}_of_{folds}", "train_index": train_idx, "test_index": test_idx, "embargo_minutes": embargo_minutes})
    return out


def cap_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows > 0 and len(frame) > max_rows:
        return frame.sample(max_rows, random_state=seed).sort_values("event_timestamp_utc")
    return frame.copy()


def cap_group_rows(frame: pd.DataFrame, column: str, max_rows_per_group: int, seed: int) -> pd.DataFrame:
    if max_rows_per_group <= 0 or column not in frame.columns or frame.empty:
        return frame.copy()
    pieces = []
    rng = np.random.default_rng(seed)
    for _, group in frame.groupby(frame[column].fillna("UNKNOWN").astype(str), sort=False):
        if len(group) > max_rows_per_group:
            pieces.append(group.sample(max_rows_per_group, random_state=int(rng.integers(0, 1_000_000))))
        else:
            pieces.append(group)
    return pd.concat(pieces, ignore_index=False).sort_values("event_timestamp_utc") if pieces else frame.iloc[0:0].copy()


def select_feature_sets(feature_sets: list[FeatureSet], names: list[str]) -> list[FeatureSet]:
    wanted = set(names)
    return [feature_set for feature_set in feature_sets if feature_set.name in wanted]


def model_feature_variants(feature_set: FeatureSet, frame: pd.DataFrame, model_types: list[str]) -> list[tuple[str, FeatureSet]]:
    if feature_set.name == "prior_only":
        return [("prior", feature_set)]
    out: list[tuple[str, FeatureSet]] = []
    for model_type in model_types:
        if model_type == "hgb":
            if not feature_set.numeric:
                continue
            out.append((model_type, numeric_only_feature_set(feature_set, frame)))
        else:
            out.append((model_type, feature_set))
    return out


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = np.clip(np.asarray(y_true, dtype=float), RV_FLOOR, None)
    pred = np.clip(np.asarray(y_pred, dtype=float), RV_FLOOR, None)
    ratio = actual / pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def topk_volatility_stats(y_true: np.ndarray, y_pred: np.ndarray, ks: list[int]) -> dict[str, float]:
    order = np.argsort(-np.asarray(y_pred, dtype=float))
    base = float(np.mean(y_true)) if len(y_true) else np.nan
    out: dict[str, float] = {"mean_realized_variance": base}
    for k in ks:
        take = order[: min(k, len(order))]
        mean_value = float(np.mean(y_true[take])) if len(take) else np.nan
        out[f"top{k}_mean_realized_variance"] = mean_value
        out[f"top{k}_realized_variance_lift"] = mean_value / base if base and not math.isnan(base) else np.nan
    return out


def regression_metric_row(base: dict[str, object], y_true: np.ndarray, y_pred: np.ndarray, ks: list[int]) -> dict[str, object]:
    pred = np.clip(np.asarray(y_pred, dtype=float), RV_FLOOR, None)
    actual = np.clip(np.asarray(y_true, dtype=float), RV_FLOOR, None)
    return {
        **base,
        "status": "completed",
        "mse_realized_variance": float(mean_squared_error(actual, pred)),
        "qlike": qlike_loss(actual, pred),
        "rmse_realized_volatility": float(np.sqrt(mean_squared_error(np.sqrt(actual), np.sqrt(pred)))),
        "spearman_realized_variance": pd.Series(pred).corr(pd.Series(actual), method="spearman") if len(actual) > 2 else np.nan,
        **topk_volatility_stats(actual, pred, ks),
    }


def fit_regression_metric(train: pd.DataFrame, test: pd.DataFrame, feature_set: FeatureSet, base: dict[str, object], args: argparse.Namespace, model_type: str = "ridge") -> dict[str, object]:
    y_train_raw = pd.to_numeric(train["realized_variance"], errors="coerce")
    y_test = pd.to_numeric(test["realized_variance"], errors="coerce")
    train = train.loc[y_train_raw.notna()].copy()
    test = test.loc[y_test.notna()].copy()
    y_train_raw = y_train_raw.loc[train.index].to_numpy(dtype=float)
    y_test = y_test.loc[test.index].to_numpy(dtype=float)
    base = {**base, "train_rows": len(train), "test_rows": len(test)}
    if train.empty or test.empty:
        return {**base, "status": "skipped_empty_split"}
    y_train = np.log(np.clip(y_train_raw, RV_FLOOR, None))
    model = clone(build_regressor(feature_set, args.max_tfidf_features, model_type=model_type))
    model.fit(train, y_train)
    pred_log = np.asarray(model.predict(test), dtype=float)
    lower = float(np.nanquantile(y_train, 0.001))
    upper = float(np.nanquantile(y_train, 0.999))
    pred_log = np.clip(pred_log, lower, upper)
    pred_rv = np.clip(np.exp(pred_log), RV_FLOOR, None)
    return regression_metric_row(base, y_test, pred_rv, args.top_ks)


def fit_classifier_metric(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_set: FeatureSet,
    base: dict[str, object],
    args: argparse.Namespace,
    model_type: str = "logistic",
) -> dict[str, object]:
    y_train = pd.to_numeric(train[target], errors="coerce")
    y_test = pd.to_numeric(test[target], errors="coerce")
    train = train.loc[y_train.notna()].copy()
    test = test.loc[y_test.notna()].copy()
    y_train = y_train.loc[train.index].astype(int)
    y_test = y_test.loc[test.index].astype(int)
    base = {**base, "train_rows": len(train), "test_rows": len(test)}
    if train.empty or test.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
        return {**base, "status": "skipped_insufficient_class_diversity"}
    model = build_classifier(feature_set, args.max_tfidf_features, args.seed, model_type=model_type)
    model.fit(train, y_train.to_numpy())
    scores = model.predict_proba(test)[:, 1]
    return classifier_metric_row(base, y_test.to_numpy(), scores, args.top_ks)


def evaluate_regression_models(frame: pd.DataFrame, mode: str, feature_sets: list[FeatureSet], args: argparse.Namespace, run_id: str, task: str = "volatility_forecast") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    splits = chronological_splits(frame, args.walk_forward_folds, args.test_fraction, args.min_train_fraction, args.embargo_minutes)
    for split in splits:
        raw_train = frame.loc[split["train_index"]].copy()
        raw_test = frame.loc[split["test_index"]].copy()
        train = cap_frame(raw_train, args.max_train_rows, args.seed + int(split["fold"]))
        test = cap_frame(raw_test, args.max_test_rows, args.seed + int(split["fold"]))
        y_train_raw = pd.to_numeric(train["realized_variance"], errors="coerce")
        y_test = pd.to_numeric(test["realized_variance"], errors="coerce")
        train = train.loc[y_train_raw.notna()].copy()
        test = test.loc[y_test.notna()].copy()
        y_train_raw = y_train_raw.loc[train.index].to_numpy(dtype=float)
        y_train = np.log(np.clip(y_train_raw, RV_FLOOR, None))
        y_test = y_test.loc[test.index].to_numpy(dtype=float)
        for source_feature_set in feature_sets:
            variants = model_feature_variants(source_feature_set, train, args.regression_model_types)
            for model_type, feature_set in variants:
                base = {
                    "run_id": run_id,
                    "task": task,
                    "mode": mode,
                    "fold": split["fold"],
                    "split_id": split["split_id"],
                    "feature_set": source_feature_set.name,
                    "model_feature_set": feature_set.name,
                    "model_type": model_type,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
                if train.empty or test.empty:
                    rows.append({**base, "status": "skipped_empty_split"})
                    continue
                model = clone(build_regressor(feature_set, args.max_tfidf_features, model_type=model_type))
                model.fit(train, y_train)
                pred_log = np.asarray(model.predict(test), dtype=float)
                lower = float(np.nanquantile(y_train, 0.001))
                upper = float(np.nanquantile(y_train, 0.999))
                pred_log = np.clip(pred_log, lower, upper)
                pred_rv = np.clip(np.exp(pred_log), RV_FLOOR, None)
                rows.append(regression_metric_row(base, y_test, pred_rv, args.top_ks))
                keep = ["event_id", "ticker", "event_timestamp_utc", "source", "source_type", "derived_event_family", "headline"]
                present = [column for column in keep if column in test.columns]
                preds.append(
                    test[present].assign(
                        run_id=run_id,
                        task=task,
                        mode=mode,
                        fold=split["fold"],
                        split_id=split["split_id"],
                        feature_set=source_feature_set.name,
                        model_feature_set=feature_set.name,
                        model_type=model_type,
                        realized_variance=y_test,
                        predicted_realized_variance=pred_rv,
                        realized_volatility=np.sqrt(np.clip(y_test, RV_FLOOR, None)),
                        predicted_realized_volatility=np.sqrt(pred_rv),
                    )
                )
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def evaluate_tail_classifiers(frame: pd.DataFrame, mode: str, feature_sets: list[FeatureSet], args: argparse.Namespace, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = ["target_top_quartile_realized_variance", "jump_indicator"]
    splits = chronological_splits(frame, args.walk_forward_folds, args.test_fraction, args.min_train_fraction, args.embargo_minutes)
    for split in splits:
        raw_train = frame.loc[split["train_index"]].copy()
        raw_test = frame.loc[split["test_index"]].copy()
        train = cap_frame(raw_train, args.max_train_rows, args.seed + int(split["fold"]))
        test = cap_frame(raw_test, args.max_test_rows, args.seed + int(split["fold"]))
        for target in targets:
            if target not in train.columns or target not in test.columns:
                continue
            y_train = pd.to_numeric(train[target], errors="coerce")
            y_test = pd.to_numeric(test[target], errors="coerce")
            train_target = train.loc[y_train.notna()].copy()
            test_target = test.loc[y_test.notna()].copy()
            y_train = y_train.loc[train_target.index].astype(int)
            y_test = y_test.loc[test_target.index].astype(int)
            for source_feature_set in feature_sets:
                variants = model_feature_variants(source_feature_set, train_target, args.classifier_model_types)
                for model_type, feature_set in variants:
                    base = {
                        "run_id": run_id,
                        "task": "extreme_tail_selection",
                        "mode": mode,
                        "fold": split["fold"],
                        "split_id": split["split_id"],
                        "target": target,
                        "feature_set": source_feature_set.name,
                        "model_feature_set": feature_set.name,
                        "model_type": model_type,
                        "train_rows": len(train_target),
                        "test_rows": len(test_target),
                    }
                    if train_target.empty or test_target.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
                        rows.append({**base, "status": "skipped_insufficient_class_diversity"})
                        continue
                    model = build_classifier(feature_set, args.max_tfidf_features, args.seed, model_type=model_type)
                    model.fit(train_target, y_train.to_numpy())
                    scores = model.predict_proba(test_target)[:, 1]
                    rows.append(classifier_metric_row(base, y_test.to_numpy(), scores, args.top_ks))
    return pd.DataFrame(rows)


def family_column(frame: pd.DataFrame) -> str:
    return "derived_event_family" if "derived_event_family" in frame.columns else "event_family"


def evaluate_family_specialists(frame: pd.DataFrame, mode: str, args: argparse.Namespace, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = family_column(frame)
    counts = frame[column].fillna("unknown").astype(str).value_counts()
    families = counts[counts >= args.min_family_rows].head(args.max_families).index.tolist()
    feature_sets = select_feature_sets(feature_sets_for(frame, mode), SPECIALIST_FEATURE_SET_NAMES)
    regression_rows: list[pd.DataFrame] = []
    classifier_rows: list[pd.DataFrame] = []
    for family in families:
        family_frame = frame[frame[column].fillna("unknown").astype(str).eq(family)].copy().reset_index(drop=True)
        metrics, _ = evaluate_regression_models(family_frame, mode, feature_sets, args, run_id, task="family_volatility_forecast")
        if not metrics.empty:
            metrics["family"] = family
            metrics["family_rows"] = len(family_frame)
            regression_rows.append(metrics)
        tail = evaluate_tail_classifiers(family_frame, mode, feature_sets, args, run_id)
        if not tail.empty:
            tail["family"] = family
            tail["family_rows"] = len(family_frame)
            classifier_rows.append(tail)
    return (
        pd.concat(regression_rows, ignore_index=True) if regression_rows else pd.DataFrame(),
        pd.concat(classifier_rows, ignore_index=True) if classifier_rows else pd.DataFrame(),
    )


def group_holdout_values(frame: pd.DataFrame, column: str, max_groups: int, min_rows: int) -> list[str]:
    if column not in frame.columns:
        return []
    counts = frame[column].fillna("UNKNOWN").astype(str).value_counts()
    return counts[counts >= min_rows].head(max_groups).index.tolist()


def evaluate_group_holdout_regression(frame: pd.DataFrame, mode: str, args: argparse.Namespace, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_sets = select_feature_sets(feature_sets_for(frame, mode), ROBUSTNESS_FEATURE_SET_NAMES)
    holdout_columns = [column for column in ["ticker", "source", "source_type", family_column(frame)] if column in frame.columns]
    for column in holdout_columns:
        values = group_holdout_values(frame, column, args.max_holdout_groups, args.min_holdout_rows)
        group_values = frame[column].fillna("UNKNOWN").astype(str)
        for value in values:
            train = frame.loc[~group_values.eq(value)].copy()
            test = frame.loc[group_values.eq(value)].copy()
            train = cap_frame(train, args.max_train_rows, args.seed)
            test = cap_frame(test, args.max_test_rows, args.seed)
            for source_feature_set in feature_sets:
                variants = model_feature_variants(source_feature_set, train, args.regression_model_types)
                for model_type, feature_set in variants:
                    base = {
                        "run_id": run_id,
                        "task": "robustness_group_holdout",
                        "mode": mode,
                        "robustness_type": "group_holdout",
                        "split_id": f"holdout_{column}_{safe_id(value)}",
                        "holdout_column": column,
                        "holdout_value": value,
                        "feature_set": source_feature_set.name,
                        "model_feature_set": feature_set.name,
                        "model_type": model_type,
                    }
                    rows.append(fit_regression_metric(train, test, feature_set, base, args, model_type=model_type))
    return pd.DataFrame(rows)


def evaluate_capped_concentration_regression(frame: pd.DataFrame, mode: str, args: argparse.Namespace, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_sets = select_feature_sets(feature_sets_for(frame, mode), ROBUSTNESS_FEATURE_SET_NAMES)
    splits = chronological_splits(frame, args.walk_forward_folds, args.test_fraction, args.min_train_fraction, args.embargo_minutes)
    for cap_column in [column for column in ["ticker", "source"] if column in frame.columns]:
        for split in splits:
            raw_train = frame.loc[split["train_index"]].copy()
            raw_test = frame.loc[split["test_index"]].copy()
            train = cap_group_rows(raw_train, cap_column, args.cap_rows_per_group, args.seed + int(split["fold"]))
            train = cap_frame(train, args.max_train_rows, args.seed + int(split["fold"]))
            test = cap_frame(raw_test, args.max_test_rows, args.seed + int(split["fold"]))
            for source_feature_set in feature_sets:
                variants = model_feature_variants(source_feature_set, train, args.regression_model_types)
                for model_type, feature_set in variants:
                    base = {
                        "run_id": run_id,
                        "task": "robustness_group_cap",
                        "mode": mode,
                        "robustness_type": "group_cap",
                        "split_id": split["split_id"],
                        "cap_column": cap_column,
                        "cap_rows_per_group": args.cap_rows_per_group,
                        "feature_set": source_feature_set.name,
                        "model_feature_set": feature_set.name,
                        "model_type": model_type,
                    }
                    rows.append(fit_regression_metric(train, test, feature_set, base, args, model_type=model_type))
    return pd.DataFrame(rows)


def evaluate_robustness_gauntlet(frame: pd.DataFrame, mode: str, args: argparse.Namespace, run_id: str) -> pd.DataFrame:
    pieces = [
        evaluate_group_holdout_regression(frame, mode, args, run_id),
        evaluate_capped_concentration_regression(frame, mode, args, run_id),
    ]
    return pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True) if any(not piece.empty for piece in pieces) else pd.DataFrame()


def safe_id(value: object) -> str:
    text = str(value)
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return out[:64] or "blank"


def options_coverage_balance(joined: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = joined[joined.get("train_eligible", False).astype(bool)].copy()
    if frame.empty or "has_options_features" not in frame.columns:
        return pd.DataFrame()
    frame["event_timestamp_utc"] = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="coerce")
    frame["event_year"] = frame["event_timestamp_utc"].dt.year
    frame["abs_return"] = pd.to_numeric(frame.get(RETURN_TARGET), errors="coerce").abs()
    frame["target_abs_return_2pct_any"] = frame["abs_return"].ge(0.02).astype(float)
    family = family_column(frame)
    strata_cols = [column for column in ["event_year", family, "source_type"] if column in frame.columns]
    options = frame[frame["has_options_features"].eq(1)].copy()
    no_options = frame[frame["has_options_features"].eq(0)].copy()
    rows = [
        balance_row(frame, "all_train_eligible"),
        balance_row(options, "options_covered"),
        balance_row(no_options, "options_missing"),
    ]
    if strata_cols and not options.empty and not no_options.empty:
        matched_options = []
        matched_controls = []
        rng = np.random.default_rng(args.seed)
        for keys, group in options.groupby(strata_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            mask = pd.Series(True, index=no_options.index)
            for column, value in zip(strata_cols, keys):
                mask &= no_options[column].fillna("MISSING").astype(str).eq(str(value))
            controls = no_options.loc[mask]
            n = min(len(group), len(controls), args.max_matched_rows_per_stratum)
            if n <= 0:
                continue
            matched_options.append(group.sample(n, random_state=int(rng.integers(0, 1_000_000))))
            matched_controls.append(controls.sample(n, random_state=int(rng.integers(0, 1_000_000))))
        if matched_options and matched_controls:
            rows.append(balance_row(pd.concat(matched_options, ignore_index=True), "matched_options_covered"))
            rows.append(balance_row(pd.concat(matched_controls, ignore_index=True), "matched_options_controls"))
    return pd.DataFrame(rows)


def balance_row(frame: pd.DataFrame, sample: str) -> dict[str, object]:
    if frame.empty:
        return {"sample": sample, "rows": 0}
    family = family_column(frame)
    return {
        "sample": sample,
        "rows": len(frame),
        "unique_tickers": int(frame["ticker"].nunique()) if "ticker" in frame.columns else np.nan,
        "unique_sources": int(frame["source"].nunique()) if "source" in frame.columns else np.nan,
        "mean_abs_return_20m_to_1d": float(pd.to_numeric(frame.get("abs_return"), errors="coerce").mean()),
        "abs_return_2pct_rate": float(pd.to_numeric(frame.get("target_abs_return_2pct_any"), errors="coerce").mean()),
        "top_ticker_share": max_group_share(frame, "ticker"),
        "top_source_share": max_group_share(frame, "source"),
        "top_family_share": max_group_share(frame, family),
    }


def max_group_share(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = frame[column].fillna("UNKNOWN").astype(str).value_counts(normalize=True)
    return float(values.iloc[0]) if len(values) else np.nan


def bootstrap_deltas(metrics: pd.DataFrame, baseline_feature_set: str, args: argparse.Namespace) -> pd.DataFrame:
    completed = metrics[metrics["status"].eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(args.seed)
    group_columns = ["task", "mode", "model_type"] if "model_type" in completed.columns else ["task", "mode"]
    for keys, group in completed.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        baseline = group[group["feature_set"].eq(baseline_feature_set)].set_index("fold")
        if baseline.empty:
            continue
        for feature_set, candidate in group.groupby("feature_set", dropna=False):
            candidate = candidate.set_index("fold")
            common_folds = sorted(set(candidate.index) & set(baseline.index))
            if feature_set == baseline_feature_set or not common_folds:
                continue
            cand = candidate.loc[common_folds]
            base = baseline.loc[common_folds]
            for metric, improvement_direction in [
                ("qlike", "lower_is_better"),
                ("mse_realized_variance", "lower_is_better"),
                ("spearman_realized_variance", "higher_is_better"),
                ("top250_realized_variance_lift", "higher_is_better"),
            ]:
                if metric not in cand.columns or metric not in base.columns:
                    continue
                deltas = pd.to_numeric(cand[metric], errors="coerce").to_numpy() - pd.to_numeric(base[metric], errors="coerce").to_numpy()
                deltas = deltas[np.isfinite(deltas)]
                if len(deltas) == 0:
                    continue
                boot = [float(np.mean(rng.choice(deltas, size=len(deltas), replace=True))) for _ in range(args.bootstrap_iterations)]
                rows.append(
                    {
                        "task": keys[0],
                        "mode": keys[1],
                        "model_type": keys[2] if len(keys) > 2 else "unknown",
                        "baseline_feature_set": baseline_feature_set,
                        "feature_set": feature_set,
                        "metric": metric,
                        "improvement_direction": improvement_direction,
                        "folds": len(deltas),
                        "mean_delta": float(np.mean(deltas)),
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_arrival_models(frame: pd.DataFrame, mode: str, args: argparse.Namespace, run_id: str) -> pd.DataFrame:
    rows = []
    target = "future_news_wave_24h"
    feature_sets = [
        FeatureSet("lagged_market_activity", tuple(available_columns(frame, LAGGED_STATE_NUMERIC + BASELINE_NUMERIC)), tuple(available_columns(frame, ["ticker", "derived_event_family"])), False),
        FeatureSet("lagged_activity_plus_headline_context", tuple(available_columns(frame, LAGGED_STATE_NUMERIC + BASELINE_NUMERIC)), tuple(available_columns(frame, BASELINE_CATEGORICAL)), True),
    ]
    splits = chronological_splits(frame, args.walk_forward_folds, args.test_fraction, args.min_train_fraction, args.embargo_minutes)
    for split in splits:
        train = cap_frame(frame.loc[split["train_index"]].copy(), args.max_train_rows, args.seed + int(split["fold"]))
        test = cap_frame(frame.loc[split["test_index"]].copy(), args.max_test_rows, args.seed + int(split["fold"]))
        y_train = pd.to_numeric(train[target], errors="coerce")
        y_test = pd.to_numeric(test[target], errors="coerce")
        train = train.loc[y_train.notna()].copy()
        test = test.loc[y_test.notna()].copy()
        y_train = y_train.loc[train.index].astype(int)
        y_test = y_test.loc[test.index].astype(int)
        for source_feature_set in feature_sets:
            variants = model_feature_variants(source_feature_set, train, args.classifier_model_types)
            for model_type, feature_set in variants:
                base = {
                    "run_id": run_id,
                    "task": "reverse_causality_news_arrival",
                    "mode": mode,
                    "fold": split["fold"],
                    "split_id": split["split_id"],
                    "target": target,
                    "feature_set": source_feature_set.name,
                    "model_feature_set": feature_set.name,
                    "model_type": model_type,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_positive_rate": float(y_test.mean()) if len(y_test) else np.nan,
                }
                if train.empty or test.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
                    rows.append({**base, "status": "skipped_insufficient_class_diversity"})
                    continue
                model = build_classifier(feature_set, args.max_tfidf_features, args.seed, model_type=model_type)
                model.fit(train, y_train.to_numpy())
                scores = model.predict_proba(test)[:, 1]
                rows.append(
                    {
                        **base,
                        "status": "completed",
                        "average_precision": float(average_precision_score(y_test, scores)),
                        "roc_auc": float(roc_auc_score(y_test, scores)),
                        "top250_wave_rate": topk_binary_rate(y_test.to_numpy(), scores, 250),
                    }
                )
    return pd.DataFrame(rows)


def topk_binary_rate(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float))
    take = order[: min(k, len(order))]
    return float(np.mean(y_true[take])) if len(take) else np.nan


def coverage_summary(joined: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {"metric": "joined_baseline_rows", "value": len(joined)},
        {"metric": "options_join_rate", "value": float(joined["has_options_features"].mean()) if "has_options_features" in joined.columns else np.nan},
        {"metric": "sec_join_rate", "value": float(joined["has_sec_features"].mean()) if "has_sec_features" in joined.columns else np.nan},
        {"metric": "macro_join_rate", "value": float(joined["has_macro_features"].mean()) if "has_macro_features" in joined.columns else np.nan},
        {"metric": "sector_reaction_join_rate", "value": float(joined["has_sector_reaction_features"].mean()) if "has_sector_reaction_features" in joined.columns else np.nan},
    ]
    for mode, frame in frames.items():
        rows.extend(
            [
                {"metric": f"{mode}_same_row_volatility_rows", "value": len(frame)},
                {"metric": f"{mode}_unique_tickers", "value": int(frame["ticker"].nunique()) if not frame.empty else 0},
                {"metric": f"{mode}_mean_post_bar_count", "value": float(pd.to_numeric(frame.get("post_bar_count"), errors="coerce").mean()) if not frame.empty else np.nan},
                {"metric": f"{mode}_top_quartile_realized_variance_threshold", "value": frame.attrs.get("top_quartile_realized_variance_threshold", np.nan)},
            ]
        )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    done = metrics[metrics["status"].eq("completed")].copy()
    if done.empty:
        return pd.DataFrame()
    keys = ["task", "mode", "feature_set", "model_type"]
    return done.groupby(keys, dropna=False).agg(
        folds=("fold", "nunique"),
        mean_qlike=("qlike", "mean"),
        mean_mse_realized_variance=("mse_realized_variance", "mean"),
        mean_rmse_realized_volatility=("rmse_realized_volatility", "mean"),
        mean_spearman_realized_variance=("spearman_realized_variance", "mean"),
        mean_top250_realized_variance_lift=("top250_realized_variance_lift", "mean"),
    ).reset_index().sort_values(["task", "mode", "mean_qlike"], na_position="last")


def aggregate_classifier_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    done = metrics[metrics["status"].eq("completed")].copy()
    if done.empty:
        return pd.DataFrame()
    keys = ["task", "mode", "target", "feature_set", "model_type"]
    return done.groupby(keys, dropna=False).agg(
        folds=("fold", "nunique"),
        mean_base_rate=("base_rate", "mean"),
        mean_average_precision=("average_precision", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_top250_hit_rate=("top250_hit_rate", "mean"),
        mean_top250_lift=("top250_lift", "mean"),
    ).reset_index().sort_values(["target", "mean_average_precision"], ascending=[True, False], na_position="last")


def aggregate_family_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    done = metrics[metrics["status"].eq("completed")].copy()
    if done.empty:
        return pd.DataFrame()
    keys = ["task", "mode", "family", "feature_set", "model_type"]
    return done.groupby(keys, dropna=False).agg(
        family_rows=("family_rows", "max"),
        folds=("fold", "nunique"),
        mean_qlike=("qlike", "mean"),
        mean_spearman_realized_variance=("spearman_realized_variance", "mean"),
        mean_top100_realized_variance_lift=("top100_realized_variance_lift", "mean"),
        mean_top250_realized_variance_lift=("top250_realized_variance_lift", "mean"),
    ).reset_index().sort_values(["family", "mean_qlike"], na_position="last")


def aggregate_family_classifier_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    done = metrics[metrics["status"].eq("completed")].copy()
    if done.empty:
        return pd.DataFrame()
    keys = ["task", "mode", "family", "target", "feature_set", "model_type"]
    return done.groupby(keys, dropna=False).agg(
        family_rows=("family_rows", "max"),
        folds=("fold", "nunique"),
        mean_base_rate=("base_rate", "mean"),
        mean_average_precision=("average_precision", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_top100_lift=("top100_lift", "mean"),
        mean_top250_lift=("top250_lift", "mean"),
    ).reset_index().sort_values(["family", "target", "mean_average_precision"], ascending=[True, True, False], na_position="last")


def aggregate_robustness_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    done = metrics[metrics["status"].eq("completed")].copy()
    if done.empty:
        return pd.DataFrame()
    keys = ["task", "mode", "robustness_type", "feature_set", "model_type"]
    extra_keys = [column for column in ["holdout_column", "cap_column"] if column in done.columns]
    keys += extra_keys
    return done.groupby(keys, dropna=False).agg(
        checks=("split_id", "nunique"),
        mean_qlike=("qlike", "mean"),
        mean_spearman_realized_variance=("spearman_realized_variance", "mean"),
        mean_top250_realized_variance_lift=("top250_realized_variance_lift", "mean"),
        min_test_rows=("test_rows", "min"),
        mean_test_rows=("test_rows", "mean"),
    ).reset_index().sort_values(["robustness_type", "mean_qlike"], na_position="last")


def leakage_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mode, frame in frames.items():
        bad_sector = int(mode == "immediate_event" and any(column in feature_sets_for(frame, mode)[-1].numeric for column in SECTOR_EARLY_REACTION))
        rows.append({"mode": mode, "check": "immediate_mode_excludes_early_reaction_features", "violations": bad_sector, "checked_rows": len(frame)})
        rows.append({"mode": mode, "check": "realized_variance_targets_available", "violations": int(frame["realized_variance"].isna().sum()) if "realized_variance" in frame.columns else len(frame), "checked_rows": len(frame)})
    return pd.DataFrame(rows)


def write_report(
    report_dir: Path,
    coverage: pd.DataFrame,
    leakage: pd.DataFrame,
    aggregate: pd.DataFrame,
    classifier_aggregate: pd.DataFrame,
    family_aggregate: pd.DataFrame,
    family_classifier_aggregate: pd.DataFrame,
    robustness_aggregate: pd.DataFrame,
    options_balance: pd.DataFrame,
    deltas: pd.DataFrame,
    arrival: pd.DataFrame,
    messages: list[str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Week 6 Volatility and Reverse-Causality Report", ""]
    lines.extend(["## Data Coverage", "", markdown_table(coverage), ""])
    lines.extend(["## Leakage Checks", "", markdown_table(leakage), ""])
    lines.extend(["## Continuous Volatility Forecasts", ""])
    lines.append(markdown_table(aggregate.head(120)) if not aggregate.empty else "No completed volatility forecast rows.")
    lines.extend(["", "## Extreme-Tail Selection", ""])
    lines.append(markdown_table(classifier_aggregate.head(120)) if not classifier_aggregate.empty else "No completed extreme-tail classifier rows.")
    lines.extend(["", "## Family Specialists", ""])
    lines.append(markdown_table(family_aggregate.head(160)) if not family_aggregate.empty else "No completed family-specialist regression rows.")
    lines.extend(["", "## Family Tail Selection", ""])
    lines.append(markdown_table(family_classifier_aggregate.head(160)) if not family_classifier_aggregate.empty else "No completed family-specialist classifier rows.")
    lines.extend(["", "## Robustness Gauntlet", ""])
    lines.append(markdown_table(robustness_aggregate.head(160)) if not robustness_aggregate.empty else "No completed robustness rows.")
    lines.extend(["", "## Options-Coverage Balance", ""])
    lines.append(markdown_table(options_balance) if not options_balance.empty else "No options-coverage balance rows.")
    lines.extend(["", "## Formal Uncertainty Intervals", ""])
    lines.append(markdown_table(deltas.head(160)) if not deltas.empty else "No bootstrap deltas were produced.")
    lines.extend(["", "## Reverse Causality: Market Activity to News Arrival", ""])
    done_arrival = arrival[arrival["status"].eq("completed")] if not arrival.empty else pd.DataFrame()
    lines.append(markdown_table(done_arrival.head(80)) if not done_arrival.empty else "No completed news-arrival rows.")
    lines.extend(["", "## Research Takeaways", ""])
    lines.extend(recommendations(aggregate, classifier_aggregate, family_aggregate, robustness_aggregate, options_balance, deltas, arrival))
    if messages:
        lines.extend(["", "## Input Notes", ""])
        lines.extend(f"- {message}" for message in messages)
    (report_dir / "week6_volatility_causality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def recommendations(
    aggregate: pd.DataFrame,
    classifier_aggregate: pd.DataFrame,
    family_aggregate: pd.DataFrame,
    robustness_aggregate: pd.DataFrame,
    options_balance: pd.DataFrame,
    deltas: pd.DataFrame,
    arrival: pd.DataFrame,
) -> list[str]:
    lines = []
    if not aggregate.empty:
        for mode, group in aggregate.groupby("mode", dropna=False):
            best = group.sort_values("mean_qlike", na_position="last").head(1)
            if not best.empty:
                row = best.iloc[0]
                lines.append(f"- {mode}: best mean QLIKE came from `{row['feature_set']}` using `{row.get('model_type', 'unknown')}` at {row['mean_qlike']:.6g}.")
    if not classifier_aggregate.empty:
        for target, group in classifier_aggregate.groupby("target", dropna=False):
            best = group.sort_values("mean_average_precision", ascending=False, na_position="last").head(1)
            if not best.empty:
                row = best.iloc[0]
                lines.append(
                    f"- {target}: best AP came from `{row['feature_set']}` using `{row.get('model_type', 'unknown')}` at {row['mean_average_precision']:.6g}, "
                    f"with top-250 lift {row['mean_top250_lift']:.6g}."
                )
    if not family_aggregate.empty:
        best_by_family = family_aggregate.sort_values("mean_qlike", na_position="last").groupby("family", as_index=False).head(1)
        strong = best_by_family.sort_values("mean_top250_realized_variance_lift", ascending=False, na_position="last").head(3)
        if not strong.empty:
            text = ", ".join(f"{row.family} -> {row.feature_set}/{row.model_type}" for row in strong.itertuples(index=False))
            lines.append(f"- Strongest family-specific volatility rankings came from: {text}.")
    if not robustness_aggregate.empty:
        robust_best = robustness_aggregate.sort_values("mean_qlike", na_position="last").head(1).iloc[0]
        lines.append(
            f"- Robustness gauntlet best mean QLIKE row: `{robust_best['feature_set']}` using `{robust_best.get('model_type', 'unknown')}` "
            f"under `{robust_best['robustness_type']}`."
        )
    if not options_balance.empty and {"sample", "rows"}.issubset(options_balance.columns):
        opt = options_balance[options_balance["sample"].eq("options_covered")]
        missing = options_balance[options_balance["sample"].eq("options_missing")]
        if not opt.empty and not missing.empty:
            lines.append(f"- Options coverage remains selective: {int(opt.iloc[0]['rows'])} covered train-eligible rows versus {int(missing.iloc[0]['rows'])} missing rows before matching.")
    improved = deltas[
        ((deltas["metric"].eq("qlike")) & (deltas["ci_high"] < 0))
        | ((deltas["metric"].eq("spearman_realized_variance")) & (deltas["ci_low"] > 0))
    ] if not deltas.empty else pd.DataFrame()
    if improved.empty:
        lines.append("- Bootstrap intervals do not yet show a clearly defended improvement over baseline on QLIKE or Spearman; treat point-estimate gains cautiously.")
    else:
        names = ", ".join(sorted(improved["feature_set"].astype(str).unique())[:6])
        lines.append(f"- Bootstrap intervals support at least one improvement for: {names}.")
    if not arrival.empty and arrival["status"].eq("completed").any():
        best = arrival[arrival["status"].eq("completed")].sort_values("average_precision", ascending=False).head(1).iloc[0]
        lines.append(f"- Reverse-causality check: `{best['feature_set']}` predicts future same-ticker news waves with AP {best['average_precision']:.6g}, so endogenous attention should stay in the paper.")
    lines.append("- The `future_news_negative_control` rows are diagnostic only; strong performance there is shortcut/endogeneity evidence, not a tradable ex-ante feature.")
    return lines


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(format_markdown_value)
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def run_week6(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    run_id = now_run_id()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined, messages = load_joined_frame(args)
    frames = {mode: modeling_frame(joined, args, mode) for mode in args.modes}
    metrics = []
    predictions = []
    classifier_metrics = []
    family_metrics = []
    family_classifier_metrics = []
    robustness_metrics = []
    arrivals = []
    for mode, frame in frames.items():
        if frame.empty:
            continue
        standard_feature_sets = feature_sets_for(frame, mode)
        mode_metrics, mode_predictions = evaluate_regression_models(frame, mode, standard_feature_sets, args, run_id)
        metrics.append(mode_metrics)
        predictions.append(mode_predictions)
        classifier_metrics.append(evaluate_tail_classifiers(frame, mode, standard_feature_sets, args, run_id))
        causal_metrics, causal_predictions = evaluate_regression_models(frame, mode, lagged_causality_sets(frame, mode), args, run_id, task="news_to_future_vol_conditional_on_lagged_state")
        metrics.append(causal_metrics)
        predictions.append(causal_predictions)
        family_regression, family_tail = evaluate_family_specialists(frame, mode, args, run_id)
        family_metrics.append(family_regression)
        family_classifier_metrics.append(family_tail)
        robustness_metrics.append(evaluate_robustness_gauntlet(frame, mode, args, run_id))
        arrivals.append(evaluate_arrival_models(frame, mode, args, run_id))
    metric_table = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    prediction_table = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    classifier_table = pd.concat(classifier_metrics, ignore_index=True) if classifier_metrics else pd.DataFrame()
    family_table = pd.concat(family_metrics, ignore_index=True) if family_metrics else pd.DataFrame()
    family_classifier_table = pd.concat(family_classifier_metrics, ignore_index=True) if family_classifier_metrics else pd.DataFrame()
    robustness_table = pd.concat(robustness_metrics, ignore_index=True) if robustness_metrics else pd.DataFrame()
    arrival_table = pd.concat(arrivals, ignore_index=True) if arrivals else pd.DataFrame()
    aggregate = aggregate_metrics(metric_table)
    classifier_aggregate = aggregate_classifier_metrics(classifier_table)
    family_aggregate = aggregate_family_metrics(family_table)
    family_classifier_aggregate = aggregate_family_classifier_metrics(family_classifier_table)
    robustness_aggregate = aggregate_robustness_metrics(robustness_table)
    deltas = bootstrap_deltas(metric_table, "baseline_only", args)
    causal_deltas = bootstrap_deltas(metric_table[metric_table["task"].eq("news_to_future_vol_conditional_on_lagged_state")], "lagged_market_state_only", args)
    if not causal_deltas.empty:
        deltas = pd.concat([deltas, causal_deltas], ignore_index=True)
    coverage = coverage_summary(joined, frames)
    leakage = leakage_audit(frames)
    options_balance = options_coverage_balance(joined, args)

    outputs = {
        "volatility_regression_metrics": metric_table,
        "volatility_regression_predictions": prediction_table,
        "volatility_regression_aggregate": aggregate,
        "extreme_tail_classifier_metrics": classifier_table,
        "extreme_tail_classifier_aggregate": classifier_aggregate,
        "family_volatility_metrics": family_table,
        "family_volatility_aggregate": family_aggregate,
        "family_tail_classifier_metrics": family_classifier_table,
        "family_tail_classifier_aggregate": family_classifier_aggregate,
        "robustness_regression_metrics": robustness_table,
        "robustness_regression_aggregate": robustness_aggregate,
        "options_coverage_balance": options_balance,
        "bootstrap_metric_deltas": deltas,
        "reverse_causality_arrival_metrics": arrival_table,
        "coverage_summary": coverage,
        "leakage_audit": leakage,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "baseline": str(args.baseline),
            "bar_dir": str(args.bar_dir),
            "options": str(args.options),
            "sec": str(args.sec),
            "macro": str(args.macro),
            "sector": str(args.sector),
        },
        "modes": args.modes,
        "messages": messages,
        "horizon_minutes": args.horizon_minutes,
        "decision_lag_minutes": args.decision_lag_minutes,
        "max_families": args.max_families,
        "min_family_rows": args.min_family_rows,
        "regression_model_types": args.regression_model_types,
        "classifier_model_types": args.classifier_model_types,
    }
    (args.output_dir / "week6_volatility_causality_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(
        args.report_dir,
        coverage,
        leakage,
        aggregate,
        classifier_aggregate,
        family_aggregate,
        family_classifier_aggregate,
        robustness_aggregate,
        options_balance,
        deltas,
        arrival_table,
        messages,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--bar-dir", type=Path, default=DEFAULT_BAR_DIR)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument("--sec", type=Path, default=DEFAULT_SEC)
    parser.add_argument("--macro", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--sector", type=Path, default=DEFAULT_SECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--modes", nargs="*", default=["delayed_20m"], choices=["immediate_event", "delayed_20m"])
    parser.add_argument("--decision-lag-minutes", type=int, default=20)
    parser.add_argument("--horizon-minutes", type=int, default=1440)
    parser.add_argument("--min-post-bars", type=int, default=12)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--min-train-fraction", type=float, default=0.35)
    parser.add_argument("--embargo-minutes", type=int, default=1440)
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-test-rows", type=int, default=15000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-tfidf-features", type=int, default=60000)
    parser.add_argument("--regression-model-types", nargs="*", default=["ridge", "hgb"], choices=["ridge", "hgb"])
    parser.add_argument("--classifier-model-types", nargs="*", default=["logistic", "hgb"], choices=["logistic", "hgb"])
    parser.add_argument("--top-ks", type=int, nargs="*", default=[100, 250, 500])
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--max-families", type=int, default=8)
    parser.add_argument("--min-family-rows", type=int, default=150)
    parser.add_argument("--max-holdout-groups", type=int, default=3)
    parser.add_argument("--min-holdout-rows", type=int, default=150)
    parser.add_argument("--cap-rows-per-group", type=int, default=75)
    parser.add_argument("--max-matched-rows-per-stratum", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_week6(args)
    print(f"Wrote {args.output_dir}")
    print(f"Wrote {args.report_dir / 'week6_volatility_causality.md'}")
