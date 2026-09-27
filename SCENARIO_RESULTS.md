# Complementary-method scenario analysis

The analysis combines multiple complementary detectors. An individual detector's
non-trigger does not describe the outcome of the overall approach. The actual
interactive UI was exercised on eight synthetic cards across eight scenarios,
160 candidates in total. No confirmed fraud labels are supplied.

Run `python examples/check_scenarios.py` to reproduce all outputs.

## Observed contributions

| Scenario | Transactions | Isolation Forest | OCSVM | Z-score | Dynamic threshold | Method union | Session rules | Combined review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_amount | 8 | 0 | 8 | 8 | 8 | 8 | 0 | 8 |
| low_amount | 8 | 0 | 8 | 8 | 8 | 8 | 0 | 8 |
| new_ip_only | 8 | 0 | 8 | 0 | 0 | 8 | 0 | 8 |
| normal_single | 8 | 0 | 8 | 0 | 0 | 8 | 0 | 8 |
| normal_spaced_different | 32 | 0 | 32 | 3 | 1 | 32 | 0 | 32 |
| rapid_different_amounts | 32 | 0 | 32 | 3 | 1 | 32 | 24 | 32 |
| same_amount_rapid | 32 | 0 | 32 | 1 | 0 | 32 | 24 | 32 |
| same_amount_spaced | 32 | 0 | 32 | 1 | 0 | 32 | 0 | 32 |

## Interpretation

- High-amount candidates (5,000) were flagged by Z-score and dynamic threshold
  in all eight card cases, as well as by OCSVM. Isolation Forest did not flag
  those candidates. Thus those candidates **were flagged by the complementary
  method union**; they were not missed by the overall analysis.
- Low-amount candidates (1) were also flagged by the statistical methods and OCSVM.
- OCSVM flagged all 160 candidates, including normal-amount control scenarios.
  Consequently the default OR union also flagged every candidate on this sample.
  This is broad alerting, not evidence of perfect detection or useful selectivity.
  It identifies an OCSVM/combination-policy calibration question for representative
  evaluation; the results are retained without tuning to this small sample.
- Session repetition rules are separate: twenty-minute same-amount repeats did
  not trigger those rules; ten-second sequences did from the second transaction.
  Other detectors may still flag the same transactions, as the table shows.
- A model's flag count and union count are diagnostic measurements, not precision,
  recall or fraud accuracy. Statistical flags are correlated and should not be
  counted as independent votes or calibrated confidence.

## Explicit combination policy

This release uses a review union of IF, OCSVM, rolling Z-score and dynamic threshold,
plus separately identified session rules. OR is an explicit public-demo default,
not an assertion about the original project's historical business decision policy.
No temporary median/amount-ratio detector participates. The statistical flags come
from the original batch feature implementation, using its existing thresholds.

Each scenario starts from a fresh session. Subsequent transaction gaps are ten
seconds or twenty minutes. The first candidate is placed thirty seconds after
the latest global historical event, following simulator behavior. The controls
are synthetic behavioral examples rather than ground-truth fraud/non-fraud labels.
