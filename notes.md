# Development Log

## Day 1 — Done
- Package structure + config with Pydantic + .env + .gitignore
- Two commits pushed to GitHub

## Day 2 — Done
- main.py with FastAPI and /health endpoint (returns 200 OK)
- pytest set up + test for /health (green; verified by intentionally making it red)
- Dockerfile + .dockerignore + requirements.txt
- Docker build and run successful; /health tested from inside the container
- Learning: solved SOCKS5 proxy issue for Docker using WireGuard

## Day 3 — Done (Project 1, Week 1)

### Part A — Dataset + structure (Week 1 Monday)
- Chose NYC TLC Yellow Taxi (Jan 2026) as the demand dataset. Rationale: real operational feed (published monthly), strong daily/weekly seasonality, ride-demand ~ delivery-demand analog for Snapp. Target companies' real order data is not public.
- Project structure: data/{raw,processed}, notebooks/, src/. raw kept untouched; data/ gitignored.
- Loaded parquet (needs pyarrow): 3.72M rows, trip-level (1 row = 1 trip), 20 cols.
- Time col = tpep_pickup_datetime (datetime64). No trip-count column exists → target must be BUILT.
- Env: VS Code + Jupyter. Fixed Proxifier hijacking localhost (Unexpected peer connection) via a Direct rule for localhost/127.0.0.1/::1, ordered above the SOCKS5 rule.
- TODO: still on system Python — create a venv before requirements work.

### Part B — Time-series EDA (Week 1 Tuesday)
- Built hourly target: df_jan.set_index('tpep_pickup_datetime').resample('h').size() → ~744 hrs.
- Trimmed 7 spillover rows (Dec-2025 / Feb-2026) before aggregating; filtered to January only.
- Daily seasonality confirmed: trough at hour 4, peak at hour 18.
- Weekly seasonality confirmed: weekend higher (Saturday highest). day-of-week is a real predictor.
- Anomaly found ~Jan 25: daily trips dropped ~70% (145k → 45k), recovered by Jan 27-29. Likely snowstorm (verify). Best real anomaly example for Week 4 + LinkedIn post.
- Concept: series is seasonal → NON-stationary (mean shifts with hour/day). Seasonality ≠ stationarity.

## Next
- Week 1 Wednesday: implement time-based walk-forward backtest (NOT random split). The hardest, most important day of the week.
- Carry-over: verify the Jan 25 dip (quick web check) — snowstorm?
- Carry-over: create venv + start requirements hygiene.