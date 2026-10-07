"""Research source preservation and artificial cashflow checks."""
import hashlib
import json
from pathlib import Path

import pytest

from event_driven_alpha.synthetic_demo import straddle_example

ROOT = Path(__file__).resolve().parents[1]


def test_research_logic_matches_original_snapshot():
    hashes = json.loads((ROOT / 'tests/research_source_hashes.json').read_text())
    for name, expected in hashes.items():
        content = (ROOT / name).read_bytes()
        if name.endswith('final_ranking_increment.py'):
            # Only the default run path changes in the public version.
            content = content.replace(
                b'os.environ.get("FINAL_RANKING_RUN_ID", "final_ranking_increment_marketdata_20260809_v1")',
                b'os.environ.get("FINAL_RANKING_RUN_ID", "final_ranking_increment_20260806_v1")',
            )
        assert hashlib.sha256(content).hexdigest() == expected, name


def test_artificial_straddle_pays_spread_and_four_side_commissions():
    result = straddle_example()
    assert result['net_entry_debit'] == pytest.approx(240)
    assert result['gross_pnl'] == pytest.approx(-20)
    assert result['commissions_and_fees'] == pytest.approx(2.6)
    assert result['net_pnl'] == pytest.approx(-22.6)
