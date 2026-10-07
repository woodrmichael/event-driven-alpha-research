from __future__ import annotations

import pandas as pd

from event_driven_alpha.analysis.execution_options_engine import (
    StrategyLeg,
    common_multileg_feature_state,
    common_multileg_quote_state,
    defined_risk_maximum_loss,
    evidence_based_limit_fill,
    first_eligible_quote,
    strategy_cashflows,
    whole_contract_count,
)


def _quotes() -> pd.DataFrame:
    t = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
    return pd.DataFrame(
        [
            {"raw_symbol": "CALL", "ts_recv": t, "ts_event": t, "sequence": 1, "bid": 1.0, "ask": 1.2},
            {"raw_symbol": "PUT", "ts_recv": t + pd.Timedelta(milliseconds=100), "ts_event": t, "sequence": 1, "bid": 0.0, "ask": 1.1},
            {"raw_symbol": "CALL", "ts_recv": t + pd.Timedelta(seconds=1), "ts_event": t, "sequence": 2, "bid": 1.1, "ask": 1.3},
        ]
    )


def test_first_quote_filters_exact_raw_symbol_and_preserves_zero_bid() -> None:
    t = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
    selected = first_eligible_quote(_quotes(), raw_symbol="PUT", target=t, timeout_seconds=2)
    assert selected is not None
    assert selected["raw_symbol"] == "PUT"
    assert selected["bid"] == 0.0


def test_common_state_requires_distinct_symbols_and_contemporaneous_quotes() -> None:
    t = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
    legs = [StrategyLeg("CALL", "long"), StrategyLeg("PUT", "long")]
    state = common_multileg_quote_state(
        _quotes(), legs, target=t, timeout_seconds=2, maximum_leg_staleness_seconds=0.2
    )
    assert state is not None
    assert state["raw_symbol"].equals(state["manifest_raw_symbol"])
    assert state["common_state_ts_recv"].nunique() == 1


def test_feature_state_never_uses_future_quote() -> None:
    cutoff = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
    quotes = pd.DataFrame(
        [
            {"raw_symbol": "CALL", "ts_recv": cutoff, "ts_event": cutoff, "bid": 1.0, "ask": 1.2},
            {"raw_symbol": "PUT", "ts_recv": cutoff, "ts_event": cutoff, "bid": 0.9, "ask": 1.1},
            {"raw_symbol": "CALL", "ts_recv": cutoff + pd.Timedelta(seconds=1), "ts_event": cutoff, "bid": 9.0, "ask": 9.2},
            {"raw_symbol": "PUT", "ts_recv": cutoff + pd.Timedelta(seconds=1), "ts_event": cutoff, "bid": 8.9, "ask": 9.1},
        ]
    )
    state = common_multileg_feature_state(
        quotes,
        [StrategyLeg("CALL", "long"), StrategyLeg("PUT", "long")],
        cutoff=cutoff,
    )
    assert state is not None
    assert state["ask"].tolist() == [1.2, 1.1]
    assert pd.to_datetime(state["ts_recv"], utc=True).le(cutoff).all()


def test_marketable_multileg_cashflow_uses_asks_then_bids() -> None:
    legs = [StrategyLeg("CALL", "long"), StrategyLeg("PUT", "long")]
    entry = pd.DataFrame([{"raw_symbol": "CALL", "bid": 1.0, "ask": 1.2}, {"raw_symbol": "PUT", "bid": 0.9, "ask": 1.1}])
    exit_frame = pd.DataFrame([{"raw_symbol": "CALL", "bid": 1.5, "ask": 1.7}, {"raw_symbol": "PUT", "bid": 0.0, "ask": 0.2}])
    result = strategy_cashflows(entry, exit_frame, legs, commission_per_contract_per_side=0.65)
    assert result["net_entry_debit"] == 230.0
    assert result["gross_pnl"] == -80.0
    assert result["commissions_and_fees"] == 2.6


def test_limit_fill_requires_subsequent_evidence() -> None:
    t = pd.Timestamp("2025-01-02 15:00:00", tz="UTC")
    events = pd.DataFrame(
        [
            {"raw_symbol": "CALL", "ts_recv": t, "record_type": "quote", "bid": 1.0, "ask": 1.2},
            {"raw_symbol": "CALL", "ts_recv": t + pd.Timedelta(seconds=2), "record_type": "trade", "price": 1.1},
            {"raw_symbol": "CALL", "ts_recv": t + pd.Timedelta(seconds=3), "record_type": "trade", "price": 1.09},
        ]
    )
    assert evidence_based_limit_fill(events, raw_symbol="CALL", placed_at=t, timeout_seconds=1, action="buy", limit=1.1) is None
    assert evidence_based_limit_fill(events, raw_symbol="CALL", placed_at=t, timeout_seconds=2, action="buy", limit=1.1) is None
    filled = evidence_based_limit_fill(events, raw_symbol="CALL", placed_at=t, timeout_seconds=3, action="buy", limit=1.1)
    assert filled is not None and filled["fill_evidence"] == "subsequent_strict_trade_through_limit"


def test_whole_contract_sizing_never_uses_fractional_contracts() -> None:
    assert whole_contract_count(2500, 1200) == 2
    assert whole_contract_count(1000, 1200) == 0


def test_defined_risk_iron_condor_reports_finite_maximum_loss() -> None:
    legs = [
        StrategyLeg("P97", "long"),
        StrategyLeg("P99", "short"),
        StrategyLeg("C101", "short"),
        StrategyLeg("C103", "long"),
    ]
    definitions = pd.DataFrame(
        {
            "raw_symbol": ["P97", "P99", "C101", "C103"],
            "leg": ["put", "put", "call", "call"],
            "strike": [97.0, 99.0, 101.0, 103.0],
            "expiration": [pd.Timestamp("2025-06-20", tz="UTC")] * 4,
        }
    )
    maximum_loss = defined_risk_maximum_loss(
        legs,
        definitions,
        entry_net_cash=100.0,
        conservative_round_trip_fees=5.2,
    )
    assert maximum_loss == 105.2
