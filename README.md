# Pitch Sequencing: Does the Previous Pitch Predict a Whiff?

A pitch-level machine learning project on MLB Statcast data. It asks whether knowing what pitch came *before* helps predict a swing-and-miss, beyond what the current pitch already tells you (speed, movement, location, count).

## The question

> How much does pitch sequencing add to predicting a whiff, beyond the characteristics of the current pitch itself?

The intuition is that a hitter who just saw a fastball should be more vulnerable to an offspeed pitch. This project tests that, rather than assuming it.

## Data

- **Source:** MLB Statcast pitch-by-pitch data, pulled with [`pybaseball`](https://github.com/jldbc/pybaseball).
- **Window:** June 1–30, 2026 regular season (about 116k pitches, 119 columns).
- **Unit of analysis:** each *swing* (55,399 after dropping rows with missing pitch measurements).
- **Target (`whiff`):** 1 if the swing was a miss, 0 otherwise. Following Baseball Savant's convention, whiffs are `swinging_strike`, `swinging_strike_blocked`, `missed_bunt` and `foul_tip`. Swings also include fouls, balls in play and foul bunts. About 25% of swings are whiffs.

## Method

1. **Sort** pitches chronologically (`game_pk`, `at_bat_number`, `pitch_number`).
2. **Build sequence features** on the full, sorted data (before filtering to swings) with `groupby(...).shift(1)` within each plate appearance: previous pitch type, previous speed, speed change, same-type-as-previous, vertical and horizontal location change, and whether the previous pitch was a whiff.
3. **Filter to swings** and drop rows missing any of the baseline features.
4. **Split by date, not randomly.** Pitches within the same game and pitcher are highly correlated, so a random split leaks information. Train: June 1–23 (41,626 swings). Test: June 24–30 (13,773 swings).
5. **Fit two models with identical settings** (`HistGradientBoostingClassifier`, `categorical_features="from_dtype"`, `random_state=42`) and compare them on the same test swings:
   - **Model A:** current-pitch features only (pitch type, velocity, spin rate, movement, plate location, balls, strikes, batter and pitcher handedness).
   - **Model B:** Model A plus the sequence features.
6. **Evaluate** with ROC AUC and log loss against a constant-probability baseline.

## Results so far

| Model | ROC AUC | Log loss |
|---|---|---|
| Baseline (always predict the training whiff rate) | 0.500 | 0.565 |
| Model A (current pitch only) | 0.768 | 0.466 |
| Model B (plus sequence features) | 0.766 | 0.467 |

- Pitch characteristics carry real signal: Model A beats the baseline by a wide margin.
- The sequence features added nothing measurable in aggregate. The differences between A and B (about 0.002) are within what a different test week could produce.
- **Preliminary, still being investigated:** offspeed pitches thrown after a non-fastball whiff *more* (33.5%) than offspeed pitches thrown after a fastball (28.6%). The gap points the same direction for all six offspeed pitch types and is not explained by count. Model A explains part of it, but not all. It may reflect pitcher quality (pitchers with strong offspeed pitches throw them back to back) rather than a sequencing effect, which has not been ruled out yet.

## Project structure

```
pitch-sequencing/
├── data.py                  # pulls Statcast data and saves it as parquet
├── data/
│   └── statcast_june.parquet   # created by data.py (not committed)
├── <your-notebook>.ipynb    # exploration, features, models, evaluation
└── README.md
```

## Getting started (Windows)

These steps use PowerShell.

### Prerequisites

- **Python 3.10 or newer** from [python.org](https://www.python.org/downloads/). During install, check **"Add python.exe to PATH"**.
- **Git** (optional, only needed to clone the repo).

### 1. Get the code

```powershell
git clone <repo-url>
cd pitch-sequencing
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it

```powershell
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your prompt. If PowerShell says running scripts is disabled, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

(In Command Prompt instead of PowerShell, use `.venv\Scripts\activate.bat`.)

### 4. Install dependencies

```powershell
pip install pybaseball pandas numpy scikit-learn jupyter pyarrow
```

### 5. Download the data

```powershell
mkdir data
python data.py
```

This pulls June 2026 Statcast data and saves it to `data\statcast_june.parquet`. It can take several minutes the first time. `pybaseball`'s cache is enabled, so repeat runs are faster.

### 6. Run the notebook

```powershell
jupyter notebook
```

Your browser opens the Jupyter file list. Open the project notebook and choose **Kernel → Restart Kernel and Run All Cells**. Run it from the project root so the relative path `data/statcast_june.parquet` resolves.

If everything is working, you should see numbers close to the results table above. They can shift slightly if Statcast revises historical data.

## Troubleshooting

- **`ImportError: DLL load failed ... An Application Control policy has blocked this file`:** Windows is blocking one of pyarrow's files (often Smart App Control). First try `pip install --force-reinstall pyarrow`. To test outside the notebook: `python -c "import pyarrow; print(pyarrow.__version__)"`. If it still fails, check Smart App Control in Windows Security, and reboot after changing it.
- **`SystemError: ... Could not get source` after a failed import:** the notebook kernel is in a half-loaded state. Use **Kernel → Restart Kernel** (not just re-running the cell).
- **`FileNotFoundError: data/statcast_june.parquet`:** run `python data.py` first, and launch Jupyter from the project root.
- **`jupyter` is not recognized:** the virtual environment isn't active. Repeat step 3, then confirm step 4 finished.
- **Only the last line of a cell shows output:** that is normal in Jupyter. Wrap earlier lines in `print(...)` or put the one you want to see last.

## Roadmap

- Compare Model A and Model B on offspeed swings only, where the effect would live, plus a bootstrap confidence interval for the log loss difference.
- Check whether the raw "after a fastball" gap survives a within-pitcher comparison.
- Measure **tunneling** directly: compute where consecutive pitches are at the batter's decision point from Statcast release position, velocity and acceleration, and test whether a smaller "tunnel gap" goes with more whiffs.
- Pull the full season for enough data to detect small effects.
- Wrap the model in an API (FastAPI), containerize it with Docker, add CI, and deploy to AWS.

## Limitations

- One month of data. Small effects need much more.
- The models have no pitcher or batter identity, so pitcher quality can masquerade as other effects.
- The whiff target counts `foul_tip` as a whiff to match Baseball Savant. Other definitions would change the rates slightly.
