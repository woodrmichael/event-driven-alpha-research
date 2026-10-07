# Event-Driven Alpha

**Predicting post-news stock volatility with machine learning.**

By Michael Wood · [Paper](paper/final_paper.pdf) · [Presentation](presentation/final_presentation.pdf)

Can a model identify which company news events will be followed by large price movements—and turn that prediction into profitable options trades?

This study combines news, market reaction, historical options, and economic context. The model ranked future volatility well. The options test bought a call and put together (a long straddle) to trade movement in either direction, but the tested strategies did not produce robustly profitable returns after realistic execution costs.

![Extreme-volatility events become more frequent as model scores increase](figures/figure_1_volatility_by_score_decile.png)

Approximately **23%** of events in the highest score decile exceeded the training fold's extreme-volatility threshold, compared with a **5% reference rate**. The shaded band shows variation across folds, not a confidence interval.

## What I built

The project follows a prediction through to its economic consequences:

1. **Event-aligned features:** join news events with prior market conditions, the first 20 minutes of stock and sector reaction, historical options, macroeconomic context, and available SEC filings.
2. **Controlled model comparisons:** evaluate market features, structured event information, and headline text on the same events and chronological folds.
3. **Execution checks:** evaluate option trades using contract-level bid/ask quotes, fees, and conservative fill assumptions rather than treating midpoint prices as executable.

The implementation uses Python, pandas, NumPy, and scikit-learn. The final information comparison uses Ridge regression with numeric and categorical features plus TF-IDF headline text.

## Main findings

| Finding | Result |
| --- | --- |
| The full model ranks future volatility | **0.626 Spearman correlation** across 87,480 out-of-fold predictions |
| The early market reaction supplies most of the added signal | Approximately **+0.109 Spearman** over lagged market features |
| Explicit event information adds little beyond market/options context | **+0.00784 Spearman**, but not statistically established under the prespecified test |
| The two tested straddle policies fail to establish profitability | **−0.558%** over 145 trades; **−3.661%** over 233 trades |

Spearman measures how well predicted and observed rankings agree. Out-of-fold predictions score events the model did not train on.

The absolute ranking result and the incremental news result answer different questions. The full model achieves **0.626** Spearman, while the strong market/options baseline already achieves approximately **0.619**. Adding event information improves the point estimate, but its calendar-month confidence interval crosses zero and its adjusted p-value is **0.251**. The study therefore does not establish a reliable incremental contribution from explicit event content.

### Where the predictive signal comes from

The final comparison holds the estimator, evaluation rows, and folds constant while changing the available information:

| Variant | Information used |
| --- | --- |
| A | Lagged market state and ticker/time controls |
| B | A plus the first 20 minutes of market reaction |
| C | B plus historical options and macroeconomic context |
| D | C plus structured event fields and SEC context |
| E | C plus headline TF-IDF |
| F | C plus both structured fields and headline text |

A–C build on one another; D, E, and F are separate extensions of C. Most improvement comes from the early market reaction. Headline text alone does not improve on C.

![Comparison of feature groups and uncertainty around their incremental contributions](figures/figure_2_signal_sources.png)

### Why the options result matters

A long straddle needs the eventual option value to exceed the premium and trading costs. Option prices already reflect expected movement, and buying at the ask and selling at the bid consumes additional value. Identifying a high-volatility event can therefore be useful for ranking while still producing an unprofitable trade.

The two prespecified straddle policies—H2, a profitability classifier, and H3, a model of continuous return-related outcomes—were evaluated on a separate exact-quote confirmation cohort. Their negative premium-weighted returns are reported above: total net P&L divided by total premium paid. These are conservative historical simulations, not actual broker fills. See the [trading-cost figure](figures/figure_3_prediction_to_trading_gap.png) and the paper for the complete comparison.

## How it was tested

The final matched sample contains **134,584 events across 571 tickers**. Predictions are made **20 minutes after each news event**, using information available by that point. The target sums squared five-minute returns strictly after the decision through 24 hours after the event, using only observed trading bars and requiring at least 12. The model predicts the logarithm of this realized variance.

Models train on earlier events and predict later ones across five expanding chronological folds. A two-trading-day gap and duplicate-event purging separate training from validation. Preprocessing and thresholds are fitted on training data only. All model variants use news to identify the ticker and event time, including the market/options baseline.

Historical option snapshots must precede the event date, and SEC information must be available by the decision cutoff. The three event-information comparisons use paired observations, ticker- and month-clustered uncertainty estimates, and adjustment for multiple tests.

The results depend on historical data coverage and the market periods represented. They are retrospective chronological evidence, not a prospective live-trading test. The negative trading result applies to the policies tested; it does not rule out every possible options strategy.

## Try the workflow

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python examples/synthetic_workflow.py
```

The example uses **entirely synthetic events and option quotes**. It demonstrates chronological evaluation and trading-cost calculations; it does **not** reproduce the paper's results. It writes a small results summary to `outputs/synthetic_demo/summary.json`. A [notebook walkthrough](notebooks/synthetic_walkthrough.ipynb) is also included. Run the tests with `python -m pytest -q`; they cover time alignment, duplicate purging, train-only preprocessing, statistical decisions, and execution mechanics.

## Explore the implementation

- [Feature processing and model comparison](src/event_driven_alpha/analysis/final_ranking_increment.py)
- [Chronological splits, metrics, and statistical tests](src/event_driven_alpha/analysis/final_ranking_core.py)
- [Option quote selection and execution costs](src/event_driven_alpha/analysis/execution_options_engine.py)
- [Research figures](figures/README.md) · [Implementation notes](paper/README.md) · [Data access](DATA_ACCESS.md)

## Data access

Historical data is not included. The example uses synthetic data; reproducing the study requires the original datasets and appropriate provider access. See [data sources and access](DATA_ACCESS.md).
