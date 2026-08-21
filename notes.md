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

## Next
- Thu: dumb baseline ("tomorrow = same day last week") + base metric (MAE/MAPE) on the folds.
- Open: verify Jan 25 dip (snowstorm?).
- Later: split requirements prod vs dev; watch Docker dep bloat.