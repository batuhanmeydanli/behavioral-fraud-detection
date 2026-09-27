# Validation record

Checked locally on 2026-09-27 with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0 and Matplotlib 3.10.8. Dependency ranges are not a claim
that every version in those ranges was tested.

## Executed checks

`python -m unittest discover -s tests -v`: **17 tests passed**.

The tests verify:

1. Exact transaction counts, repeated amounts, new IP flags, decline runs and time gaps.
2. Retention of the original current-inclusive rolling mean, full-day totals, all-history maximum IP count and 19-feature list.
3. Correct feature assignment when CSV rows are shuffled.
4. Single-card and single-row feature extraction.
5. Response-code preservation, schema validation and rejection of non-finite amounts.
6. Original final model parameters, binary predictions and continuous decision scores.
7. All six plot functions on zero-anomaly groups, plus zero-overlap reporting.
8. Card date/score alignment for shuffled rows.
9. Agreement between simulator and batch features and exact session candidate commitment.
10. True consecutive session amount matches and retained descriptive rule output.
11. A complete interactive transaction entry followed by clean exit using simulated input.
12. CLI result/report export and suppression of all plotting when requested.

The full default CLI was also run on 480 wholly synthetic transactions across eight
cards. It produced the feature CSV, scored CSV, detailed text report, JSON summary
and all six PNG figures. The six-panel dashboard was visually inspected.

## Observed synthetic run

| Output | Flagged transactions |
|---|---:|
| Isolation Forest | 5 / 480 |
| One-Class SVM | 111 / 480 |
| Dynamic amount rule | 39 / 480 |
| Rolling Z-score, absolute value > 1.0 | 168 / 480 |
| Intersection of IF, OCSVM and dynamic rule | 0 / 480 |

These are observed outputs, not accuracy metrics. Models were fitted and scored
on the same dataset, preserving the original analysis design. The synthetic
sample is a reproducible software demonstration, not a fraud benchmark. No
performance conclusion or production claim is drawn from these counts.

No original company data was used for these checks.

## Complementary-method simulator checks

Five integration tests verify that each of IF, OCSVM, Z-score and dynamic threshold
can independently trigger the method union; no flags means no review; session
rules are separately attributable; short-gap boundaries are respected; and the
temporary amount-ratio detector is absent. Together with the twelve regression
tests, **17 tests passed**.

The full interactive scenario runner passed UI/decision consistency checks on
160 candidates with both trained models supplied. See `SCENARIO_RESULTS.md` for
per-method and union counts. OR flagged all candidates because OCSVM flagged all
of them; this is not a performance success claim. High-amount candidates were also
flagged by the original Z-score and dynamic-threshold methods. No detector was
retuned to these examples.
