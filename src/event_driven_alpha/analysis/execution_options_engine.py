"""Pure execution mechanics for multi-structure historical option research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


Side = Literal["long", "short"]
Action = Literal["buy", "sell"]


@dataclass(frozen=True)
class StrategyLeg:
    raw_symbol: str
    side: Side
    quantity: int = 1
    multiplier: float = 100.0

    def __post_init__(self) -> None:
        if not self.raw_symbol:
            raise ValueError("strategy leg requires a raw symbol")
        if self.side not in {"long", "short"}:
            raise ValueError("strategy leg side must be long or short")
        if self.quantity < 1:
            raise ValueError("strategy leg quantity must be positive")
        if not np.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("strategy leg multiplier must be finite and positive")


def classify_quote(bid: object, ask: object) -> str:
    bid_value = pd.to_numeric(pd.Series([bid]), errors="coerce").iloc[0]
    ask_value = pd.to_numeric(pd.Series([ask]), errors="coerce").iloc[0]
    if pd.isna(bid_value) or pd.isna(ask_value) or bid_value < 0 or ask_value <= 0:
        return "invalid"
    if bid_value > ask_value:
        return "crossed"
    if bid_value == ask_value:
        return "locked"
    if bid_value == 0:
        return "zero_bid"
    return "two_sided"


def prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    required = {"raw_symbol", "ts_recv", "ts_event", "bid", "ask"}
    missing = sorted(required.difference(quotes.columns))
    if missing:
        raise ValueError(f"quote table is missing columns: {missing}")
    if quotes.attrs.get("execution_options_quotes_prepared") is True:
        return quotes
    frame = quotes.copy()
    frame["raw_symbol"] = frame["raw_symbol"].astype(str)
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True, errors="coerce")
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    frame["quote_classification"] = [
        classify_quote(bid, ask) for bid, ask in zip(frame["bid"], frame["ask"])
    ]
    frame["sequence"] = pd.to_numeric(
        frame.get("sequence", pd.Series(np.arange(len(frame)), index=frame.index)),
        errors="coerce",
    ).fillna(-1)
    frame = frame.sort_values(
        ["ts_recv", "sequence", "ts_event"], kind="mergesort"
    ).reset_index(drop=True)
    frame.attrs["execution_options_quotes_prepared"] = True
    return frame


def first_eligible_quote(
    quotes: pd.DataFrame,
    *,
    raw_symbol: str,
    target: object,
    timeout_seconds: int,
) -> pd.Series | None:
    if timeout_seconds < 0:
        raise ValueError("timeout must be nonnegative")
    frame = prepare_quotes(quotes)
    target_ts = pd.Timestamp(target)
    target_ts = target_ts.tz_localize("UTC") if target_ts.tzinfo is None else target_ts.tz_convert("UTC")
    end = target_ts + pd.Timedelta(seconds=timeout_seconds)
    symbol_rows = frame.loc[frame["raw_symbol"].eq(str(raw_symbol))]
    eligible = symbol_rows.loc[
        symbol_rows["ts_recv"].between(target_ts, end, inclusive="both")
        & ~symbol_rows["quote_classification"].isin(["invalid", "crossed"])
    ]
    if eligible.empty:
        return None
    selected = eligible.iloc[0].copy()
    if str(selected["raw_symbol"]) != str(raw_symbol):
        raise RuntimeError("selected quote crossed raw symbols")
    selected["selection_delay_seconds"] = float(
        (selected["ts_recv"] - target_ts).total_seconds()
    )
    return selected


def common_multileg_quote_state(
    quotes: pd.DataFrame,
    legs: Sequence[StrategyLeg],
    *,
    target: object,
    timeout_seconds: int,
    maximum_leg_staleness_seconds: float = 1.0,
) -> pd.DataFrame | None:
    if len({leg.raw_symbol for leg in legs}) != len(legs):
        raise ValueError("multi-leg strategy raw symbols must remain distinct")
    first = {
        leg.raw_symbol: first_eligible_quote(
            quotes,
            raw_symbol=leg.raw_symbol,
            target=target,
            timeout_seconds=timeout_seconds,
        )
        for leg in legs
    }
    if any(value is None for value in first.values()):
        return None
    common_time = max(pd.Timestamp(value["ts_recv"]) for value in first.values() if value is not None)
    frame = prepare_quotes(quotes)
    rows = []
    for leg in legs:
        available = frame.loc[
            frame["raw_symbol"].eq(leg.raw_symbol)
            & frame["ts_recv"].le(common_time)
            & ~frame["quote_classification"].isin(["invalid", "crossed"])
        ]
        if available.empty:
            return None
        selected = available.iloc[-1].copy()
        age = float((common_time - selected["ts_recv"]).total_seconds())
        if age < 0 or age > maximum_leg_staleness_seconds:
            return None
        selected["manifest_raw_symbol"] = leg.raw_symbol
        selected["common_state_ts_recv"] = common_time
        selected["common_state_quote_age_seconds"] = age
        selected["side"] = leg.side
        selected["quantity"] = leg.quantity
        selected["multiplier"] = leg.multiplier
        if str(selected["raw_symbol"]) != leg.raw_symbol:
            raise RuntimeError("common state crossed raw symbols")
        rows.append(selected)
    state = pd.DataFrame(rows)
    if not state["raw_symbol"].eq(state["manifest_raw_symbol"]).all():
        raise RuntimeError("selected_quote_raw_symbol != manifest_raw_symbol")
    return state.sort_values("raw_symbol", kind="mergesort").reset_index(drop=True)


def common_multileg_feature_state(
    quotes: pd.DataFrame,
    legs: Sequence[StrategyLeg],
    *,
    cutoff: object,
    maximum_quote_age_seconds: float = 120.0,
    maximum_cross_leg_timestamp_difference_seconds: float = 1.0,
) -> pd.DataFrame | None:
    """Return the latest common leg state available no later than the cutoff."""

    if len({leg.raw_symbol for leg in legs}) != len(legs):
        raise ValueError("multi-leg strategy raw symbols must remain distinct")
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = (
        cutoff_ts.tz_localize("UTC")
        if cutoff_ts.tzinfo is None
        else cutoff_ts.tz_convert("UTC")
    )
    frame = prepare_quotes(quotes)
    rows = []
    for leg in legs:
        available = frame.loc[
            frame["raw_symbol"].eq(leg.raw_symbol)
            & frame["ts_recv"].le(cutoff_ts)
            & ~frame["quote_classification"].isin(["invalid", "crossed"])
        ]
        if available.empty:
            return None
        selected = available.iloc[-1].copy()
        age = float((cutoff_ts - selected["ts_recv"]).total_seconds())
        if age < 0 or age > maximum_quote_age_seconds:
            return None
        selected["manifest_raw_symbol"] = leg.raw_symbol
        selected["feature_cutoff_utc"] = cutoff_ts
        selected["feature_quote_age_seconds"] = age
        selected["side"] = leg.side
        selected["quantity"] = leg.quantity
        selected["multiplier"] = leg.multiplier
        rows.append(selected)
    state = pd.DataFrame(rows)
    span = float(
        (state["ts_recv"].max() - state["ts_recv"].min()).total_seconds()
    )
    if span > maximum_cross_leg_timestamp_difference_seconds:
        return None
    state["feature_cross_leg_timestamp_span_seconds"] = span
    if not state["raw_symbol"].eq(state["manifest_raw_symbol"]).all():
        raise RuntimeError("selected_feature_quote_raw_symbol != manifest_raw_symbol")
    if not pd.to_datetime(state["ts_recv"], utc=True).le(cutoff_ts).all():
        raise RuntimeError("future quote entered the feature state")
    return state.sort_values("raw_symbol", kind="mergesort").reset_index(drop=True)


def marketable_price(*, side: Side, stage: Literal["entry", "exit"], bid: float, ask: float) -> float:
    if classify_quote(bid, ask) in {"invalid", "crossed"}:
        raise ValueError("cannot fill an invalid or crossed quote")
    if stage == "entry":
        return float(ask if side == "long" else bid)
    if stage == "exit":
        return float(bid if side == "long" else ask)
    raise ValueError("stage must be entry or exit")


def strategy_cashflows(
    entry_state: pd.DataFrame,
    exit_state: pd.DataFrame,
    legs: Sequence[StrategyLeg],
    *,
    commission_per_contract_per_side: float = 0.65,
    regulatory_fee_per_contract_per_side: float = 0.0,
) -> dict[str, float]:
    entry = entry_state.set_index("raw_symbol")
    exit_frame = exit_state.set_index("raw_symbol")
    required_symbols = {leg.raw_symbol for leg in legs}
    if set(entry.index) != required_symbols or set(exit_frame.index) != required_symbols:
        raise ValueError("entry and exit states must contain exactly the strategy legs")
    entry_net_cash = 0.0
    exit_net_cash = 0.0
    gross_long_premium = 0.0
    for leg in legs:
        entry_row = entry.loc[leg.raw_symbol]
        exit_row = exit_frame.loc[leg.raw_symbol]
        entry_price = marketable_price(
            side=leg.side,
            stage="entry",
            bid=float(entry_row["bid"]),
            ask=float(entry_row["ask"]),
        )
        exit_price = marketable_price(
            side=leg.side,
            stage="exit",
            bid=float(exit_row["bid"]),
            ask=float(exit_row["ask"]),
        )
        notional = leg.quantity * leg.multiplier
        direction = -1.0 if leg.side == "long" else 1.0
        entry_net_cash += direction * entry_price * notional
        exit_net_cash -= direction * exit_price * notional
        if leg.side == "long":
            gross_long_premium += entry_price * notional
    side_transactions = 2 * sum(leg.quantity for leg in legs)
    fees = side_transactions * (
        float(commission_per_contract_per_side)
        + float(regulatory_fee_per_contract_per_side)
    )
    gross_pnl = entry_net_cash + exit_net_cash
    net_pnl = gross_pnl - fees
    net_entry_debit = max(0.0, -entry_net_cash)
    denominator = net_entry_debit if net_entry_debit > 0 else gross_long_premium
    return {
        "entry_net_cash": entry_net_cash,
        "exit_net_cash": exit_net_cash,
        "net_entry_debit": net_entry_debit,
        "gross_long_premium": gross_long_premium,
        "gross_pnl": gross_pnl,
        "commissions_and_fees": fees,
        "net_pnl": net_pnl,
        "net_return_on_premium": net_pnl / denominator if denominator > 0 else np.nan,
        "option_side_transactions": float(side_transactions),
    }


def defined_risk_maximum_loss(
    legs: Sequence[StrategyLeg],
    definitions: pd.DataFrame,
    *,
    entry_net_cash: float,
    conservative_round_trip_fees: float,
) -> float:
    """Compute worst expiration P&L over all strike breakpoints."""

    required = {"raw_symbol", "leg", "strike", "expiration"}
    missing = sorted(required.difference(definitions.columns))
    if missing:
        raise ValueError(f"maximum-risk definitions are missing columns: {missing}")
    lookup = definitions.drop_duplicates("raw_symbol").set_index("raw_symbol")
    rows = []
    for leg in legs:
        if leg.raw_symbol not in lookup.index:
            raise ValueError("maximum-risk leg definition is unavailable")
        row = lookup.loc[leg.raw_symbol]
        rows.append(
            {
                "strategy_leg": leg,
                "option_type": str(row["leg"]),
                "strike": float(row["strike"]),
                "expiration": pd.Timestamp(row["expiration"]),
            }
        )
    expirations = {row["expiration"] for row in rows}
    if len(expirations) != 1:
        raise ValueError("maximum-risk calculation requires one common expiration")
    strikes = sorted({row["strike"] for row in rows})
    evaluation_prices = [0.0, *strikes, max(strikes) * 2.0 + 1.0]
    terminal_pnl = []
    for underlying in evaluation_prices:
        payoff = 0.0
        for row in rows:
            leg = row["strategy_leg"]
            intrinsic = (
                max(underlying - row["strike"], 0.0)
                if row["option_type"] == "call"
                else max(row["strike"] - underlying, 0.0)
            )
            direction = 1.0 if leg.side == "long" else -1.0
            payoff += direction * intrinsic * leg.quantity * leg.multiplier
        terminal_pnl.append(
            float(entry_net_cash) + payoff - float(conservative_round_trip_fees)
        )
    maximum_loss = max(0.0, -min(terminal_pnl))
    if not np.isfinite(maximum_loss):
        raise ValueError("maximum loss is not finite")
    return maximum_loss


def limit_price(bid: float, ask: float, *, action: Action, mode: Literal["midpoint", "one_tick_inside"], tick_size: float = 0.01) -> float:
    if classify_quote(bid, ask) in {"invalid", "crossed", "locked"}:
        raise ValueError("limit simulation requires a positive spread")
    midpoint = (float(bid) + float(ask)) / 2.0
    if mode == "midpoint":
        return midpoint
    if mode != "one_tick_inside":
        raise ValueError("unsupported limit mode")
    if action == "buy":
        return min(float(ask), float(bid) + tick_size)
    if action == "sell":
        return max(float(bid), float(ask) - tick_size)
    raise ValueError("action must be buy or sell")


def evidence_based_limit_fill(
    events: pd.DataFrame,
    *,
    raw_symbol: str,
    placed_at: object,
    timeout_seconds: int,
    action: Action,
    limit: float,
) -> pd.Series | None:
    if timeout_seconds < 0:
        raise ValueError("timeout must be nonnegative")
    if action not in {"buy", "sell"}:
        raise ValueError("action must be buy or sell")
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("limit must be finite and nonnegative")
    required = {"raw_symbol", "ts_recv", "record_type"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"limit evidence table is missing columns: {missing}")
    placed = pd.Timestamp(placed_at)
    placed = placed.tz_localize("UTC") if placed.tzinfo is None else placed.tz_convert("UTC")
    end = placed + pd.Timedelta(seconds=timeout_seconds)
    frame = events.copy()
    frame["raw_symbol"] = frame["raw_symbol"].astype(str)
    frame["ts_recv"] = pd.to_datetime(frame["ts_recv"], utc=True, errors="coerce")
    frame = frame.loc[
        frame["raw_symbol"].eq(raw_symbol)
        & frame["ts_recv"].gt(placed)
        & frame["ts_recv"].le(end)
    ].copy()
    frame["sequence"] = pd.to_numeric(
        frame.get("sequence", pd.Series(-1, index=frame.index)), errors="coerce"
    ).fillna(-1)
    frame["ts_event"] = pd.to_datetime(
        frame.get("ts_event", pd.Series(pd.NaT, index=frame.index)),
        utc=True,
        errors="coerce",
    )
    frame = frame.sort_values(
        ["ts_recv", "sequence", "ts_event"], kind="mergesort"
    )
    for _, row in frame.iterrows():
        record_type = str(row["record_type"]).lower()
        evidence = False
        evidence_kind = ""
        if record_type == "trade":
            price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
            # A print exactly at our resting limit does not establish queue
            # priority. Require a strict trade-through unless a later quote
            # makes the order marketable.
            evidence = pd.notna(price) and (price < limit if action == "buy" else price > limit)
            evidence_kind = "subsequent_strict_trade_through_limit"
        elif record_type == "quote":
            bid = pd.to_numeric(pd.Series([row.get("bid")]), errors="coerce").iloc[0]
            ask = pd.to_numeric(pd.Series([row.get("ask")]), errors="coerce").iloc[0]
            if pd.notna(bid) and pd.notna(ask):
                evidence = ask <= limit if action == "buy" else bid >= limit
                evidence_kind = "subsequent_marketable_quote_at_limit"
        if evidence:
            selected = row.copy()
            selected["fill_price"] = float(limit)
            selected["fill_evidence"] = evidence_kind
            selected["fill_delay_seconds"] = float((row["ts_recv"] - placed).total_seconds())
            return selected
    return None


def whole_contract_count(budget_usd: float, premium_per_contract_usd: float) -> int:
    if budget_usd < 0 or premium_per_contract_usd <= 0:
        return 0
    return int(np.floor(float(budget_usd) / float(premium_per_contract_usd)))
