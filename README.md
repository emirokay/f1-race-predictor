# F1 Miami Grand Prix 2026 — Race Prediction Model

Predicts the finishing order of the 2026 Miami Grand Prix using an XGBRanker model trained on historical F1 data (2022-2025) from four street circuits.

## Approach

Finishing order in Formula 1 is a **ranking problem**, not a regression problem. This model uses **XGBRanker** (pairwise ranking objective) to get the order right within each race, rather than predicting absolute finishing positions.

### Training Data

- **Source:** FastF1 API — historical sessions from 2022-2025
- **Circuits:** Miami, Monaco, Singapore, Las Vegas (4 street / low-overtaking circuits)
- **Validation:** Leave-One-Year-Out (LOYO) cross-validation on Miami only — strict temporal integrity

### Features (13 total)

| Feature | Description |
|---|---|
| GridPosition | Qualifying position |
| QualyGapToPole | Continuous gap from pole (seconds) — breaks ties in top 4 |
| SprintPosition | Sprint finishing order (best single pre-race predictor) |
| CarStrength | `log(TopSpeed x TeamRank) x RegStability` |
| PaceDelta | Normalized pace gap to fastest driver |
| DegradationRate | Tire deg slope from long runs |
| DriverCircuitAvgFinish | Rolling circuit-specific history (strict no-leakage) |
| Consistency | Lap time coefficient of variation |
| SectorVariance | Std dev of sector deltas |
| Sector1/2/3_Delta | Per-sector time gaps |
| HasSprint | Sprint weekend flag |

### 2026 Prediction

For the 2026 race, all data comes from **pre-race sessions only** (Qualifying, Sprint, FP1) via the [TracingInsights](https://github.com/TracingInsights/2026) public dataset. Race results are **never loaded** — data safety is explicit.

## Results

| Metric | Average (LOYO CV) |
|---|---|
| MAE | 3.03 positions |
| Kendall's Tau | 0.57 |
| Spearman ρ | 0.76 |
| Top 5 Overlap | 80% |
| Podium Overlap | 50% |

### Predicted Top 5 — Miami 2026

1. Lando Norris (McLaren)
2. Kimi Antonelli (Mercedes)
3. Charles Leclerc (Ferrari)
4. Max Verstappen (Red Bull Racing)
5. George Russell (Mercedes)

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook f1_predictor.ipynb
```

Dependencies: `fastf1`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `scipy`, `requests`, `jupyter`

## Notes

- The FastF1 API cache (`f1_cache/`) is re-generated on first run — requires internet
- Some historical races may fail to load due to FastF1 API rate limits (the code handles this gracefully)
- 2026 data relies on the TracingInsights GitHub repo (public, maintained by the F1 community)
 