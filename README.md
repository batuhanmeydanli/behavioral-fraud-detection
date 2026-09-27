# Behavioral Fraud Detection

Based on a behavioral fraud detection project developed during an internship using real transaction data. This public version uses synthetic data and a cleaned implementation for demonstration. The workflow combines **19 engineered features**, **Isolation Forest**, **One-Class SVM**, **Z-score analysis** and **dynamic thresholds**, supported by detailed reports, visualizations and an interactive transaction simulator.

**For confidentiality, this repository includes fully synthetic example data instead of company transaction data.**

## Project overview

The project examines how transactions differ from cardholders' observed behavior across amounts, transaction frequency, IP usage, response codes, currencies and card/customer relationships. Complementary detection methods make it possible to inspect which methods identify each transaction and where their results overlap.

The workflow covers:

1. Loading and validating transaction data.
2. Engineering behavioral, temporal and contextual features.
3. Training Isolation Forest and One-Class SVM and calculating statistical flags.
4. Comparing method outputs and generating card-level analysis.
5. Exploring new transaction scenarios through an interactive simulator.

**Stack:** Python, Pandas, NumPy, scikit-learn and Matplotlib.

## Feature engineering

| Category | Features |
|---|---|
| Amount behavior | Five-transaction rolling average, rolling amount Z-score, daily amount Z-score, daily mean and daily total |
| Transaction activity | Ten-minute count, one-hour count, daily count, time since previous transaction and rolling mean interarrival time |
| IP behavior | New-IP indicator and maximum distinct-IP count within a one-hour window |
| Response behavior | Consecutive declined transactions |
| Currency behavior | Daily currency variety and deviation from the card's most frequent currency |
| Card/customer relationships | Daily cards per customer and distinct customers per card |
| Repetition and timing | Same-amount transactions within one hour and most frequent transaction hour |

The model feature list is defined in `MODEL_FEATURES`. Rolling amount Z-score uses a ten-transaction window; the dynamic rule uses rolling mean ± 1.5 standard deviations. The separate Z-score flag uses an absolute threshold of 1.0, with additional thresholds included in the analysis report.

## Detection and analysis

| Method | Configuration |
|---|---|
| Isolation Forest | 150 estimators, contamination 0.01, random state 42 |
| One-Class SVM | RBF kernel, gamma `auto`, nu 0.05 |
| Rolling Z-score | Card-level amount deviation over a rolling window |
| Dynamic threshold | Card-level rolling mean and standard deviation |

The methods are reported separately and through a combined review flag. The default combination is **any method flags the transaction (OR)**; the simulator also reports separate session rules for rapid activity and repeated amounts. Each result identifies the methods responsible for the flag.

Model decision scores use `decision_function`: lower values indicate greater anomaly, with zero as the decision boundary. A flagged transaction is a candidate for review, not confirmed fraud.

## Quick start

Python 3.10 or newer is required. From the project folder:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and run the project:

```bash
python -m pip install -r requirements.txt
python fraud_detection_pipeline.py
```

The default run uses **480 synthetic transactions across eight fictional cards** and writes results to `outputs/`.

Additional commands:

```bash
# Interactive transaction scenarios
python fraud_detection_pipeline.py --interactive

# Select a card for detailed charts
python fraud_detection_pipeline.py --card SYNTHETIC_CARD_000

# Use a compatible local CSV
python fraud_detection_pipeline.py --input data/transactions.csv --output-dir outputs/custom

# Run without generating charts
python fraud_detection_pipeline.py --skip-plots
```

## Transaction data format

| Column | Description |
|---|---|
| `CardToken` | Card identifier |
| `CardHolderIp` | IP identifier |
| `Amount` | Finite numeric transaction amount |
| `CurrencyCode` | Currency category |
| `ResponseCode` | `00` denotes success; other codes count as declines in this mapping |
| `TransactionDate` | Transaction timestamp; ISO 8601 recommended |
| `externalCustomerId` | Optional customer identifier for card-sharing features |

Identifiers are read as strings. Timestamps are normalized to UTC; timestamps without an offset are interpreted as UTC. Adapt the response-code mapping to the source dataset when necessary.

**Schema note:** The public demo uses a normalized analysis schema and does not reproduce the original raw company log schema. Column names and mappings in this repository are designed for the public demonstration pipeline.

## Outputs

- **Feature and scored CSVs:** transaction data, engineered features, model scores, per-method flags and their union.
- **Text report:** flag distributions, threshold comparisons, overlapping detections, card rankings and detailed transaction examples.
- **JSON summary:** aggregate analysis results.
- **Six figures:** a six-panel dashboard, IF and OCSVM card transaction views, a rolling mean/Z-score view, and separate model score views.

Figures are saved under `outputs/plots/` without opening GUI windows. Rerunning with the same output directory replaces generated files.

## Interactive simulator

Start with `--interactive`, then enter a known card identifier, a positive amount and optionally an IP address. Enter `q` at the card prompt to exit.

The simulator displays feature values, both model results, statistical flags, contributing methods and session-rule results. Simulated transactions remain in memory and become part of the session history for subsequent scenarios. Batch and simulator model features use the same extraction function.

Session rules cover consecutive same-amount/same-currency transactions within ten minutes, at least three session transactions within ten minutes, and inter-session gaps shorter than thirty seconds. The interactive currency defaults to `TRY`, and the candidate response defaults to `00`.

## Tests and examples

```bash
# Automated regression and decision-integration tests
python -m unittest discover -s tests -v

# Reproducible interactive scenario checks
python examples/check_scenarios.py

# Regenerate the synthetic dataset
python examples/generate_synthetic_data.py
```

The test suite covers feature values, row alignment, input validation, model outputs, plotting, session handling and the contribution of each detection method. The scenario script exercises 160 synthetic transactions and exports per-method and combined results.

## Analysis context

This is a retrospective transaction-analysis workflow: both models fit and score the supplied dataset, and some features use full-day or full-history aggregates. Raw amount aggregates are not FX-normalized. Synthetic examples demonstrate execution and method behavior; they are not a measurement of performance on the original company data.

Only fictional data is included. Outputs retain values from the input dataset, so use synthetic examples when publishing CSVs, reports or screenshots.
