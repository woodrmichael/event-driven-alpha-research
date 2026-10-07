"""Tiny artificial-data workflow using the unchanged historical study primitives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from event_driven_alpha.analysis.final_ranking_core import (
    expanding_walk_forward_splits, purge_training_rows, spearman,
    training_quantile_mapping,
)
from event_driven_alpha.analysis.final_ranking_increment import (
    feature_variants, ridge_pipeline,
)
from event_driven_alpha.analysis.execution_options_engine import StrategyLeg, strategy_cashflows


def synthetic_events(seed: int = 42, rows: int = 120) -> pd.DataFrame:
    """Generate no provider-derived records; relationships are invented for teaching."""
    rng = np.random.default_rng(seed)
    times = pd.date_range('2020-01-02 15:00', periods=rows, freq='B', tz='UTC')
    lagged = rng.lognormal(-3, 0.5, rows)
    reaction = rng.normal(0, 0.02, rows)
    implied = rng.uniform(0.1, 0.8, rows)
    target = np.exp(-8 + 10 * lagged + 12 * np.abs(reaction) + implied + rng.normal(0, 0.4, rows))
    return pd.DataFrame({
        'event_id': [f'synthetic-{i:04}' for i in range(rows)],
        'event_timestamp_utc': times,
        'feature_cutoff_utc': times + pd.Timedelta(minutes=20),
        'label_end_utc': times + pd.Timedelta(hours=24),
        'headline_cluster_key': [f'artificial-cluster-{i}' for i in range(rows)],
        'ticker': [f'SYN{i % 3}' for i in range(rows)],
        'event_hour_utc': times.hour.astype(str),
        'event_day_of_week': times.dayofweek.astype(str),
        'event_month': times.month.astype(str),
        'event_year': times.year.astype(str),
        'pre_event_realized_vol_30m': lagged,
        'stock_return_0m_20m': np.abs(reaction),
        'feature_synthetic_implied_vol': implied,
        'source': ['artificial' for _ in range(rows)],
        'derived_event_family': ['synthetic-update' if i % 2 else 'synthetic-results' for i in range(rows)],
        'headline': ['artificial company update scenario' if i % 2 else 'artificial company results scenario' for i in range(rows)],
        'realized_variance': target,
    })


def straddle_example() -> dict[str, float]:
    """Invented quotes illustrate negative ask-to-bid P&L with per-side fees."""
    legs = [StrategyLeg('SYN_CALL', 'long'), StrategyLeg('SYN_PUT', 'long')]
    entry = pd.DataFrame({'raw_symbol': ['SYN_CALL', 'SYN_PUT'], 'bid': [1.0, 1.0], 'ask': [1.2, 1.2]})
    exit_state = pd.DataFrame({'raw_symbol': ['SYN_CALL', 'SYN_PUT'], 'bid': [1.1, 1.1], 'ask': [1.3, 1.3]})
    return strategy_cashflows(entry, exit_state, legs)


def run_demo(seed: int = 42) -> dict:
    frame = synthetic_events(seed)
    variants = feature_variants(frame)
    predictions = {v.name: [] for v in variants}
    fold_audit = []
    # Three folds keep this a small smoke check, not historical model training.
    for split in expanding_walk_forward_splits(frame, folds=3):
        train, audit = purge_training_rows(frame, split, trading_dates=frame.event_timestamp_utc)
        validation = split.validation_index
        # The 24-hour artificial target window must finish before validation.
        assert (frame.loc[train, 'label_end_utc'] < split.validation_start).all()
        fold_audit.append({'fold': split.fold, 'training_rows': len(train), 'validation_rows': len(validation), 'purged_rows': len(audit)})
        ytrain = frame.loc[train, 'realized_variance'].to_numpy()
        yvalid = frame.loc[validation, 'realized_variance'].to_numpy()
        for variant in variants:
            model = ridge_pipeline(variant)
            model.fit(frame.loc[train], np.log(ytrain))
            train_score = model.predict(frame.loc[train])
            valid_score = model.predict(frame.loc[validation])
            _, q95 = training_quantile_mapping(ytrain, train_score, yvalid, valid_score)
            predictions[variant.name].extend(zip(yvalid.tolist(), valid_score.tolist()))
            assert np.isfinite(q95)
    metrics = []
    for variant in variants:
        pairs = np.asarray(predictions[variant.name])
        metrics.append({'variant': variant.name, 'synthetic_oof_rows': len(pairs), 'synthetic_spearman': spearman(pairs[:, 0], pairs[:, 1])})
    return {
        'notice': 'Artificial workflow demonstration only; these metrics do not reproduce or validate the research findings.',
        'seed': seed, 'synthetic_events': len(frame), 'folds': fold_audit,
        'metrics': metrics, 'invented_straddle_cashflows': straddle_example(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/synthetic_demo'))
    args = parser.parse_args()
    result = run_demo()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['notice'])
    print(f"Wrote {args.output / 'summary.json'}")
