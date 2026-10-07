# Contribution guide

Use Python 3.12. Install `requirements.txt` and `pip install -e . --no-deps`.
Run `python -m pytest -q` and `python examples/synthetic_workflow.py`.

Preserve the frozen study definitions and unfavorable results. Fit preprocessing and thresholds on training data only; keep chronological splits, embargoes, and duplicate purges. Do not replace reported historical metrics with synthetic-demo outputs.

Research modules were exported byte-for-byte; changes must disclose deviations from the frozen study. Put reusable new code under `src/event_driven_alpha/` and keep example CLIs thin. Do not add provider records, real headlines, secrets, model vocabularies, or private session material. Do not place orders. See `DATA_ACCESS.md` for reproduction boundaries.
