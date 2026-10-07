"""Derived event taxonomy and headline cluster features for Week 5 research."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from event_driven_alpha.analysis.week4_local_data import read_table


DEFAULT_DATASET = Path("data/processed/week4_headline_hpc_modeling_dataset.parquet")
DEFAULT_OUTPUT = Path("data/processed/week5_news_v2_research_features.parquet")
DEFAULT_REPORT_DIR = Path("reports/week5_event_taxonomy")
RETURN_COLUMN = "market_adjusted_return_20m_to_1d"

EVENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("earnings", re.compile(r"\b(earnings?|eps|quarterly results?|q[1-4]|fiscal (quarter|year)|revenue|sales miss|sales beat|profit|loss narrows?|loss widens?)\b", re.I)),
    ("guidance", re.compile(r"\b(guidance|forecast|outlook|expects?|raises? outlook|cuts? outlook|lowers? outlook|preannounc|sees fy|sees q[1-4])\b", re.I)),
    ("analyst_rating", re.compile(r"\b(upgrade[sd]?|downgrade[sd]?|initiates?|reiterate[sd]?|rating|outperform|underperform|buy rating|sell rating|neutral rating)\b", re.I)),
    ("price_target", re.compile(r"\b(price target|pt raised|pt lowered|target price|raises? .*target|cuts? .*target)\b", re.I)),
    ("ma_deal", re.compile(r"\b(acquir(es?|ed|ing)|acquisition|merger|buyout|takeover|deal to buy|to acquire|stake in|divest|sale of|spin[- ]?off)\b", re.I)),
    ("legal_regulatory", re.compile(r"\b(lawsuit|sues?|settlement|sec\b|doj\b|ftc\b|probe|investigation|regulator|approval|antitrust|fine|recall|patent|court|trial)\b", re.I)),
    ("fda_clinical", re.compile(r"\b(fda|phase [123]|clinical trial|drug|therapy|biotech|pdufa|approval|complete response|endpoint|study results?)\b", re.I)),
    ("macro_policy", re.compile(r"\b(fed|fomc|rates?|inflation|cpi|ppi|jobs report|payrolls?|gdp|treasury yields?|tariff|china|oil prices?)\b", re.I)),
    ("financing_capital_return", re.compile(r"\b(offering|secondary|debt offering|notes offering|convertible|buyback|repurchase|dividend|share sale|prices offering)\b", re.I)),
    ("product_strategy", re.compile(r"\b(launch(es?|ed)?|product|partnership|contract|order|wins? contract|agreement|expands?|strategy|ai\b|cloud|chip|semiconductor)\b", re.I)),
    ("labor_layoffs", re.compile(r"\b(layoffs?|job cuts?|strike|union|workers?|hiring freeze|restructuring)\b", re.I)),
    ("executive_board", re.compile(r"\b(ceo|cfo|coo|chairman|board|appoints?|resigns?|steps down|management change)\b", re.I)),
    ("commodity_energy", re.compile(r"\b(oil|gas|crude|natural gas|gold|copper|mining|energy|opec|drilling|pipeline)\b", re.I)),
    ("routine_market_commentary", re.compile(r"\b(stocks? to watch|market update|premarket|midday|after-hours|why .* shares|moving today|top stories|watchlist)\b", re.I)),
]

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "s",
    "shares",
    "stock",
    "stocks",
    "the",
    "to",
    "with",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def classify_event_family(headline: object) -> str:
    text = clean_text(headline)
    if not text:
        return "unknown"
    for family, pattern in EVENT_PATTERNS:
        if pattern.search(text):
            return family
    return "other"


def normalized_headline_tokens(headline: object, max_tokens: int = 12) -> list[str]:
    text = clean_text(headline).lower()
    text = re.sub(r"\$[a-z]{1,5}\b", " ticker ", text)
    text = re.sub(r"\b[A-Z]{1,5}\b", " ticker ", text)
    text = re.sub(r"\b\d+(\.\d+)?%?\b", " number ", text)
    tokens = [token for token in TOKEN_RE.findall(text) if token not in STOP_TOKENS]
    collapsed: list[str] = []
    for token in tokens:
        if collapsed and collapsed[-1] == token:
            continue
        collapsed.append(token)
    return collapsed[:max_tokens]


def headline_cluster_key(headline: object) -> str:
    tokens = normalized_headline_tokens(headline)
    return " ".join(tokens) if tokens else "unknown"


def enrich_event_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "headline" not in out.columns:
        out["headline"] = ""
    out["headline"] = out["headline"].map(clean_text)
    out["derived_event_family"] = out["headline"].map(classify_event_family)
    out["headline_cluster_key"] = out["headline"].map(headline_cluster_key)
    ticker = out.get("ticker", pd.Series("", index=out.index)).fillna("UNKNOWN").astype(str).str.upper()
    out["ticker_headline_cluster_key"] = ticker + "::" + out["headline_cluster_key"]
    if "event_timestamp_utc" in out.columns:
        timestamps = pd.to_datetime(out["event_timestamp_utc"], utc=True, errors="coerce")
        out["event_date_utc"] = timestamps.dt.strftime("%Y-%m-%d").fillna("unknown")
        out["event_hour_utc"] = timestamps.dt.hour.fillna(-1).astype(int)
    return out


def taxonomy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, group in frame.groupby("derived_event_family", dropna=False):
        returns = pd.to_numeric(group.get(RETURN_COLUMN, pd.Series(index=group.index, dtype=float)), errors="coerce")
        rows.append(
            {
                "derived_event_family": family,
                "rows": len(group),
                "unique_tickers": group.get("ticker", pd.Series(index=group.index, dtype=object)).nunique(dropna=True),
                "mean_abs_return": float(returns.abs().mean()) if returns.notna().any() else pd.NA,
                "mean_signed_return": float(returns.mean()) if returns.notna().any() else pd.NA,
                "abs2_rate": _mean_column(group, "target_abs_return_2pct_20m_1d"),
                "volnorm_rate": _mean_column(group, "target_volnorm_abs_2sigma_20m_1d"),
                "positive_2pct_rate": _mean_column(group, "target_positive_2pct_20m_1d"),
                "negative_2pct_rate": _mean_column(group, "target_negative_2pct_20m_1d"),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_abs_return", "rows"], ascending=[False, False], na_position="last")


def cluster_summary(frame: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby("headline_cluster_key", dropna=False):
        if len(group) < min_rows:
            continue
        returns = pd.to_numeric(group.get(RETURN_COLUMN, pd.Series(index=group.index, dtype=float)), errors="coerce")
        rows.append(
            {
                "headline_cluster_key": key,
                "rows": len(group),
                "unique_tickers": group.get("ticker", pd.Series(index=group.index, dtype=object)).nunique(dropna=True),
                "unique_sources": group.get("source", pd.Series(index=group.index, dtype=object)).nunique(dropna=True),
                "mean_abs_return": float(returns.abs().mean()) if returns.notna().any() else pd.NA,
                "abs2_rate": _mean_column(group, "target_abs_return_2pct_20m_1d"),
                "volnorm_rate": _mean_column(group, "target_volnorm_abs_2sigma_20m_1d"),
                "example_headline": group["headline"].iloc[0] if "headline" in group.columns and len(group) else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_abs_return", "rows"], ascending=[False, False], na_position="last")


def _mean_column(frame: pd.DataFrame, column: str) -> float | pd._libs.missing.NAType:
    if column not in frame.columns:
        return pd.NA
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else pd.NA


def write_taxonomy_report(report_dir: Path, frame: pd.DataFrame, family: pd.DataFrame, clusters: pd.DataFrame) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    family.to_csv(report_dir / "event_family_summary.csv", index=False)
    clusters.to_csv(report_dir / "headline_cluster_summary.csv", index=False)
    lines = [
        "# Week 5 Event Taxonomy",
        "",
        f"Rows enriched: {len(frame):,}.",
        f"Derived families: {frame['derived_event_family'].nunique(dropna=True):,}.",
        f"Headline clusters with at least 10 rows: {len(clusters):,}.",
        "",
        "## Family Summary",
        "",
        family.head(25).to_markdown(index=False) if not family.empty else "No family rows.",
        "",
        "## Repeated Headline Clusters",
        "",
        clusters.head(25).to_markdown(index=False) if not clusters.empty else "No repeated cluster rows.",
    ]
    (report_dir / "week5_event_taxonomy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_event_taxonomy(dataset: Path, output: Path, report_dir: Path, max_rows: int = 0, seed: int = 7) -> pd.DataFrame:
    source = read_table(dataset)
    if max_rows > 0 and len(source) > max_rows:
        source = source.sample(max_rows, random_state=seed).sort_index()
    frame = enrich_event_taxonomy(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    family = taxonomy_summary(frame)
    clusters = cluster_summary(frame)
    write_taxonomy_report(report_dir, frame, family, clusters)
    (report_dir / "week5_event_taxonomy_config.json").write_text(
        json.dumps({"dataset": str(dataset), "output": str(output), "max_rows": max_rows, "seed": seed}, indent=2),
        encoding="utf-8",
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_event_taxonomy(args.dataset, args.output, args.report_dir, max_rows=args.max_rows, seed=args.seed)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_dir}")


if __name__ == "__main__":
    main()
