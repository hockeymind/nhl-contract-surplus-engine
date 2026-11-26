# NHL Contract Surplus Engine

Values NHL skater contracts in dollars: goals above replacement (GAR) per player-season, a forecast, a market price curve, and year-by-year surplus against a growing salary cap, with 80% bands.

**Status: research project, not a finished product.** Estimates are model outputs with stated error rates and known biases (see Limitations).

## Headline validation (strict out-of-time test, `src/strict_backtest.py`)
Train: 706 contracts signed 2018-22, with outcomes cut off at the last season before the test signings and an age curve built only from earlier seasons. Test: 350 contracts signed 2023-24. Forecast error 7.05 vs 8.48 goals above replacement per 82 games (2017-19 scale) for guessing the average; 6.74 with the market adjustment. **Last season's GAR/82 alone scores 7.17**, so the forecast adds little over that for point forecasts; the value is in the price curve, aging and bands.
Not part of the test: the aging scale (0.53, fitted in-sample on 2018-23 signings) and the final weights (refit on 2018-24, 1,126 contracts). Small remaining leakage: shrinkage constants and replacement level use all seasons. Price curve uses all 2018-26 signings. 80% bands cover 77% of one-year-ahead outcomes and 72% at five years (variance model fit on 2007-16, tested 2017+).

## Limitations
- GAR undercounts defensive value; defense grades are provisional.
- Forwards at 7%+ of the cap have beaten forecasts (+1.0 GAR/82 average, unstable by period); a pooled correction is applied and is probably still too low.
- 2023-26 signings have no realized outcomes; fair-market-value projections are model estimates.
- Entry-level and sub-$1M contracts are excluded; no goalies.

## Data (not included)
Raw data is not redistributed. Obtain it yourself and place it as below:
- MoneyPuck skater season files -> `data/raw/skaters YYYY.csv` (free for non-commercial use with credit: moneypuck.com)
- PuckPedia signings (saved pages) -> parse with `src/parse_puckpedia.py`; check their terms first
- NHL birth dates -> `python src/fetch_birthdates.py`

## Run order
`src/load_data.py` -> `src/gar.py` -> `src/build_workbook.py` -> `src/price_drift_test.py` -> `src/backtest_rolling.py` -> `src/strict_backtest.py` -> `src/refit_v2.py` -> `src/project_contracts.py` -> `src/project_new.py` -> `src/fmv.py` -> `src/build_workbook2.py` (`grade.py`/`recalibrate.py` are the archived v1 pipeline)

Code written with AI assistance; the model design and checks are the author's.
