# Development Log

## Day 1 — Done
- Package structure + config (Pydantic) + .env + .gitignore. 2 commits pushed.

## Day 2 — Done
- FastAPI /health endpoint (200 OK) + pytest (verified green by making it red first).
- Dockerfile + .dockerignore; build/run OK, /health tested inside container.
- Learning: fixed Docker SOCKS5 proxy issue via WireGuard.

## Day 3 — Done (Week 1, Mon–Tue)

### Dataset + structure
- Chose NYC TLC Yellow Taxi (Jan 2026): real monthly feed, strong daily/weekly seasonality, ride-demand ~ delivery-demand analog for Snapp.
- 3.72M rows, trip-level, 20 cols. No trip-count column → target must be BUILT. Time col = tpep_pickup_datetime.
- Fixed Proxifier hijacking localhost (Direct rule for localhost, above SOCKS5).

### Time-series EDA
- Built hourly target: resample('h').size() → ~744 hrs. Trimmed 7 spillover rows, January only.
- Daily seasonality: trough hour 4, peak hour 18. Weekly: weekend higher (Sat highest) → day-of-week is a real predictor.
- Anomaly ~Jan 25: daily trips −70% (145k→45k), recovered Jan 27–29. Likely snowstorm (verify). Best anomaly example for Week 4 + post.
- Concept: seasonal → NON-stationary. Seasonality ≠ stationarity.

## Day 4 — Done (Week 1, Wed)

### Walk-forward backtest
- Saved hourly target to data/processed/, loaded in new 02_backtest.ipynb. One source of truth: backtest only evaluates, doesn't rebuild.
- Wrote walk_forward_splits() from scratch (generator). Leakage guard: assert train.max() < test.min() every fold.
- Params: initial_train=24*14 (2 weeks → learns weekly seasonality), horizon=24, step=24 → 17 folds, no gaps/overlap.
- expanding vs sliding in one function (only train start differs). Verified visually.
- Experiment: step=48 + horizon=24 → visible gaps in coverage. Made step≠horizon concrete.
- Validated vs sklearn TimeSeriesSplit: with test_size=24, n_splits=17 → EXACT boundary match. Confirms my function is correct.
- Interview point: chose hand-built for direct control over horizon + minimum train aligned to weekly seasonality — generic tool lacks domain knowledge.
- Note (from my algo-trading background): this is the same idea as walk-forward optimization used to backtest trading strategies. A random shuffle split here would be the ML equivalent of look-ahead bias — using future data the model wouldn't have had at prediction time.

### Env hygiene (carry-over cleared)
- Created + activated .venv (fixed PowerShell ExecutionPolicy). Installed full deps, regenerated requirements.txt.
- Learning: pip freeze OVERWRITES — install everything first, then freeze.
- Switched Jupyter kernel to .venv; imports verified.

## Day 5 — Done (Week 1, Thu)

### Seasonal-naive baseline
- Prediction for hour t = actual at t − 168h (same hour, previous week). No model fit.
- 168 = 24×7 → locks daily + weekly seasonality at once. Zero params, nothing to overfit.
- Evaluated on the same walk-forward folds (apples-to-apples with Week 2 model).

### Results (17 folds)
| metric | mean | median | std |
|---|---|---|---|
| MAE  | 760.6 | 516.2 | 665.0 |
| MAPE | 29.5% | 9.9%  | 53.8% |

- Official baseline = median MAPE 9.9% (typical day). Mean/median gap driven by 2 outlier folds, not general instability.

### Anomaly confirmed — 3 independent sources
- EDA: −70% trips ~Jan 25.
- Baseline: 2 folds near Jan 25–26 spike to 214% / 112% MAPE (~10× rest).
- Real event: largest NYC snowstorm since 2021 (Central Park 11.4", daily record; precip ended Jan 26, 8:30 AM).
- → Keep as eval case-study for Week 4 anomaly detector.

## Day 6 — Done (Week 1, Fri)

### Feature engineering
- Built make_features(y: pd.Series) -> pd.DataFrame in app/features.py. Single responsibility: only BUILDS features, no split, no dropna.
- Read y independently from data/processed/ (not from notebook 02) → one source of truth, notebooks stay decoupled.

### Features + leakage reasoning
- Calendar (hour, dayofweek, is_low_demand): safest family — derived from index, no aggregation, zero leak risk.
- is_low_demand = (hour >= 23) | (hour <= 5), thresholds read from EDA (trough at 4–5, drop-off from 23). Encodes domain knowledge → saves the tree several nested splits.
- lag_168 = y.shift(168): inherently backward-looking (safe). Positive shift = past; negative would be look-ahead leak. Creates a 168-row NaN warm-up tail.
- ma_24 = y.rolling(24).mean().shift(1): rolling includes row t itself → leak. shift(1) drops y_t from the window (closed candle, no repaint). Smallest shift that closes the leak = best.

### Decisions (separation of concerns)
- Features built in make_features. dropna handled AFTER the call, before split. walk-forward split stays separate.
- Dropping the leading NaN rows is safe: trims the START of the series (warm-up), doesn't break row continuity → walk-forward unaffected.

### Leakage guard
- assert_series_equal(y.shift(1).rolling(24).mean(), feat["ma_24"], check_names=False).
- Same value computed two independent ways; passes silently → shift verified correct.

## Carry to Week 2
- Availability caveat: with horizon=24, shift(1) is optimistic — real last-known point is t-24, not t-1. Must address when building the model.
- Week 2 target: LightGBM must beat median MAPE 9.9% AND beat baseline on the 2 storm folds.