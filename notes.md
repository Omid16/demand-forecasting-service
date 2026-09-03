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


## Day 7 — Done (Week 1→2 bridge)

### Availability correction (the carry-over from Week 1)
- ma_24: shift(1) → shift(25). Reason = 1 (no-repaint, window must not see current row) + 24 (availability: worst hour of the 24h block sits 24h ahead of last closed point t-1).
- Rule derived: a feature with shift >= horizon is inherently safe (passes availability). Only shorter-than-horizon windows need manual push-back.
  - lag_168: 168 >= 24 → safe (worst hour still reaches only t-145).
  - ma_24: short window → had to be pushed to shift(25).
- Verified the correction is NOT a no-op: (shift(1) != shift(25)).sum() = 744 rows differ; max abs diff ~4820. head(3) hid it (looked identical) → lesson: use (a!=b).sum() / .abs().max() to diff, not head.

### 03_features.ipynb — see & validate X (no model yet)
- Notebook only calls make_features from app/features.py (no feature logic in notebook → single source of truth).
- X = feat.dropna() drops warm-up tail. y aligned by LABEL: y1 = y.loc[X.index] (not positional slicing — positional breaks silently if a mid-series NaN ever appears).
- Guard: assert (X.index == y1.index).all() → true alignment, not just equal length.
- warm-up: lag_168 makes the longest NaN tail → first 7 days (168 rows) consumed (not lost — they feed lag_168 for the rest). Series 744 → 576h. X starts 2026-01-08.

### Walk-forward wired to the honest X
- walk_forward_splits(X, 24*14, 24, 24, "expanding") → 10 folds (was 17). Reason: (576-336)/24 ≈ 10; 7 daily steps dropped with the 7-day warm-up. Predicted by hand before running, confirmed by len(folds).
- expanding confirmed visually: train.min fixed at Jan 8, train.max slides forward each fold. Leakage guard (train.max < test.min) green.
- Storm folds (Jan 25–26) not in first 3 → likely fold 3–4. Watch these when scoring the model.

### Refactor / hygiene
- walk_forward_splits moved from notebook 02 into app/backtest.py. Notebook 02 now imports it → single source of truth (killed the duplicate-definition trap).
- Tooling lesson: stale-kernel trap in Jupyter — editing features.py doesn't update an already-imported function. Fix = Restart Kernel (clean) or importlib.reload (quick). MQL5 analog: edit without recompile.



## Day 8 — Done (Week 2, Mon — baseline re-score)

### Apples-to-apples fix
- Baseline was scored on 17 folds; honest X now yields 10. Re-scored seasonal-naive on the SAME 10 folds before any model comparison.
- Read baseline straight from X: y_pred = X.loc[te, "lag_168"] (the column IS y.shift(168)) → bit-for-bit identical to what the model sees. No fresh shift.
- Same score fns as Day 5 (sklearn MAE/MAPE), untouched.

### New official baseline (10 folds)
| metric | mean | median |
|---|---|---|
| MAPE | 40.5% | 10.5% |
- Official baseline = median MAPE 10.5% (was 9.9% on 17 folds). This 10.5% is now THE number to beat.
- Predicted by hand, confirmed: median barely moved (9.9→10.5), mean jumped (29.5→40.5). Storm folds now 2-of-10 (was 2-of-17) → heavier weight. mean is sum-based (sensitive to tail), median is rank-based (robust) → two-regime data confirmed (Week 4 fuel).

## Day 9 — Done (Week 2, first .fit())

### Leakage caught + closed
- First LightGBM gave median MAPE 1.9% — too good → suspected look-ahead (algo-trading instinct: dream Sharpe = cheat first, genius never).
- feature_importances_: top feature was `y` itself → make_features leaks target into X (used features: 6, should be 5).
- Fix: X = X.drop(columns="y") in notebook. pandas trap: drop returns a copy, must reassign to X (bare X.drop is a no-op).
- Confirmed closed: used features 6→5, MAPE jumped to honest number.

### First honest model (10 folds, default params, random_state=42)
| metric | mean | median |
|---|---|---|
| MAPE | 48.0% | 15.2% |
- Model LOST to baseline: 15.2% vs 10.5%. Not a failure — expected.
- Cause: only 336 rows/fold → default GBM (100 trees, unbounded depth) overfits. `No further splits` warnings = too little data/structure. Zero-param seasonal-naive has nothing to overfit → beats an untamed GBM on regular taxi data.



## Day 10 — Done (Week 2, root-fix leak + regularization)

### Leak closed at root
- make_features: `return df` → `return df.drop(columns="y")`. y still BUILT inside (lag_168, ma_24 need it) but no longer LEAVES. Working column ≠ output column.
- Verified X.columns == 5. Notebook workaround removed → root is single source of truth (FastAPI Week 3 gets clean X automatically).

### Regularization
- num_leaves=5, min_child_samples=7, n_estimators=50, random_state=42.
- Why these: attack tree-internal complexity first (answers the `No further splits` symptom); n_estimators is the blunt lever. Light trees = weak learners → need MORE of them, not fewer.
- `No further splits` warnings GONE → worked mechanically. But median MAPE ~15.6% → still loses to baseline 10.5%.

### Per-fold (the story median hid)
- 8 normal folds ~10–26% → level with baseline. median over normal ≈ 14%.
- 2 storm folds (3,4): 245% / 131% → WORSE than baseline (214% / 112%). Model is storm-fragile, not storm-robust.

### Lessons
- median immune to VALUE of outliers, not their COUNT (15.6% on 10 vs 14% on 8 normal). → separate regimes before aggregating.
- Model fit on normal regime = worst in regime shift (confidently continues old pattern). Storm = regime shift = algo-trading crash/gap analog.
- Not overfit anymore — no feature tells the model a storm exists. Forecasting a storm ≠ forecaster's job → Week 4 anomaly detector.
- On regular data, seasonal-naive is a genuinely hard baseline. Beating it needs BETTER FEATURES, not more regularization.

## Carry to Day 11
- DECISION (eyes open): one more fold on a NEW feature to try to beat baseline. Ahead of roadmap so justified; may take up to a day (each new lag/rolling needs its own leakage/availability analysis + re-run).
- Feature = HOLIDAYS. Known in advance → no leakage. Hypothesis: lag_168 misfires when last week normal but this week is a holiday.
- Storm feature REJECTED = look-ahead leakage (saw the storm in data already). → Week 4 anomaly detector.
- ⚠️ ANSWER FIRST: data is January 2026 only. How many real holidays are in it? If 1–2, too few samples to learn → revisit decision before building.



## Day 11 — Done (Week 2, Tue — experiment tracking)

### HOLIDAYS feature — REJECTED (answered the "answer first" question)
- Counted real holidays inside honest X (starts Jan 8): New Year (Jan 1) falls in warm-up → OUT. MLK (Jan 19) → IN. So n=1 real holiday in the eval window.
- n=1 → a tree can't learn a feature that fires once in the whole series. Feature dropped. Honest result, not a failure.
- Weekend ≠ holiday: model already knows weekends via dayofweek + lag_168. For taxi, weekend is PEAK demand (Sat highest, EDA Day 3), not low → flagging it would teach the reverse. MQL5 analog: weekend = market closed, hardcoded in calendar, not a news-event flag.
- Spike handling = NOT forecaster's job. Known-in-advance (holiday) could be a feature; surprise spike (storm) = look-ahead if flagged from data already seen. → Week 4 anomaly detector on residuals.
- DECISION: model is level with seasonal-naive on normal data. Accepted as the honest result (as sworn Day 10). Modeling of Project 1 is CLOSED.

### MLflow set up (SQLite backend)
- `pip install mlflow` (3.15.2). Side effect: pandas 3.0.5 → 2.3.3 (mlflow needs pandas<3). Re-ran notebooks 02+03, leakage guards still green → downgrade safe. Froze requirements.txt.
- Backend = sqlite:///mlflow.db (not default file store). Reason: model registry (Week 3/4) needs a DB backend, file store can't do it. Ahead-choice.
- .gitignore += mlflow.db, mlruns/, mlartifacts/. Three DIFFERENT reasons for ignore: .venv = reproducible-local, .env = secret, mlflow.db = local state of another tool (NOT reproducible, but not git's job either). MLflow complements git (versions results), doesn't replace it (versions code).
- UI: `mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db` — must point at the SAME db or UI shows empty.

### num-leaves bug (caught by MLflow)
- LightGBM warned `Unknown parameter: num-leaves` → I'd passed it with a HYPHEN. LGBM silently ignored it and ran default num_leaves=31, NOT 5. So the "regularized" run wasn't regularized at all.
- Worse: log_params logged num_leaves=5 (correct dict) while model ran 31 → MLflow would've recorded a LIE. Fixed the key → warnings gone, median MAPE back to 15.6% (matches Day 10).
- Lesson: this is exactly why single-source-of-truth matters. One dict feeds BOTH model (**params) AND MLflow → no room to diverge. For defaults, read from the model itself: model.get_params(), never hand-type.

### 3-model comparison logged (roadmap Tue: "compare several models")
- All 3 scored on the SAME 10 folds (apples-to-apples, Day 8 rule). Eval params (initial_train=336, horizon=24, step=24, expanding) logged on every run = proof of fair comparison.

| run | model_type | mape_median | mape_mean |
|---|---|---|---|
| seasonal_naive | baseline | 10.5% | 40.5% |
| lgbm_default | num_leaves=31 | 15.2% | 48.0% |
| lgbm_regularized | num_leaves=5 | 15.6% | 49.5% |

- Baseline read straight from X.loc[te, "lag_168"] → no fresh shift, bit-for-bit Day 8. Duration 60ms vs model 1.5s → cheapest baseline still beats costliest model on normal data. Bottleneck is FEATURES, not compute/algorithm.
- Every run: mape_mean logged alongside mape_median on purpose. The mean/median gap (≈50% vs ≈15%) shows the two-regime (storm) story without opening per-fold.

### Why no other algorithms (XGBoost/CatBoost etc.)
- Bottleneck is features, not model. Any GBM hits the seasonal-naive ceiling on normal data + goes blind in storm folds (no storm feature). Swapping algorithm doesn't move that ceiling. Roadmap's sacred rule: wrapper hires you, not model accuracy. Adding algorithms = tuning the thing that isn't the problem.

## Carry to Day 12
- joblib: save the model + interpret results (roadmap Fri). OPEN QUESTION first: WHICH model to serve — regularized, default, or baseline? Baseline wins on median (10.5%) but isn't a real model. Decide before saving.
- Still open for Week 2: LinkedIn post #3 (plan = video series, not text).