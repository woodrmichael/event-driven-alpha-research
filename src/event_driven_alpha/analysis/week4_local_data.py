"""Local Week 4 data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


REQUIRED_BAR_COLUMNS = ["symbol", "timestamp_utc", "open", "high", "low", "close", "volume"]
DEFAULT_RAW_EVENT_DIR = Path("data/raw/headlines")
DEFAULT_LOCAL_BAR_DIR = Path("data/raw/bars/5min")
DEFAULT_DAILY_BAR_DIR = Path("data/raw/bars/daily")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_symbol(value: object) -> str:
    text = text_value(value).upper()
    if text in {"ES_OR_MES", "ES/MES", "ES_MES"}:
        return "ES_OR_MES"
    return text


def normalize_bar_frame(frame: pd.DataFrame, symbol: str = "", source_file: str = "") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS + ["source_file"])
    out = frame.copy()
    rename_map = {
        "ticker": "symbol",
        "bar_symbol": "symbol",
        "datetime": "timestamp_utc",
        "timestamp": "timestamp_utc",
        "time": "timestamp_utc",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns and value not in out.columns})
    if "symbol" not in out.columns:
        out["symbol"] = symbol or Path(source_file).stem
    for column in REQUIRED_BAR_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out = out[REQUIRED_BAR_COLUMNS].copy()
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce").dt.floor("5min")
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["source_file"] = source_file
    return out.dropna(subset=["symbol", "timestamp_utc"]).drop_duplicates(["symbol", "timestamp_utc"], keep="last")


def load_local_bars(bar_dir: Path = DEFAULT_LOCAL_BAR_DIR, symbols: set[str] | None = None) -> pd.DataFrame:
    if not bar_dir.exists():
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS + ["source_file"])
    normalized_symbols = {normalize_symbol(symbol) for symbol in symbols} if symbols else None
    frames: list[pd.DataFrame] = []
    for path in sorted(bar_dir.glob("*.csv")):
        symbol = normalize_symbol(path.stem)
        if normalized_symbols is not None and symbol not in normalized_symbols:
            continue
        frames.append(normalize_bar_frame(read_table(path), symbol=symbol, source_file=str(path)))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS + ["source_file"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "timestamp_utc"], keep="last")
