# Data access

**This repository does not distribute the historical research data.** The runnable example generates artificial events and option quotes without contacting a data provider.

The study used these sources:

| Source | Use |
| --- | --- |
| Benzinga, including Alpaca news routes | News events and headlines |
| Alpaca | Five-minute stock, benchmark, and sector bars |
| MarketData.app | Historical option-chain information |
| FRED | Macroeconomic context |
| SEC EDGAR | Regulatory filing context |
| Databento | Contract-level option quotes and trades |

Raw records, headlines, cached responses, trained model files, and event-level research outputs are excluded. Provider redistribution permissions have not been established; access to a dataset should not be treated as permission to republish it.

The paper and figures report study-level findings. The included code and synthetic example allow readers to inspect the methodology and run the workflow. Reproducing the historical metrics requires the original point-in-time datasets and preprocessing artifacts, plus appropriate access to each source.
