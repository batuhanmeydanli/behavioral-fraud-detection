"""Reproduce simulator decisions on 160 synthetic transactions.

Tests exercise the actual interactive UI. Expected outcomes specify demo rule
behavior, not ground-truth fraud labels or model accuracy.
"""
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
import contextlib
import io
import json
import re
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fraud_detection_pipeline as p

SCENARIOS = {
    'normal_single': ([90], 1200, False),
    'high_amount': ([5000], 1200, False),
    'low_amount': ([1], 1200, False),
    'new_ip_only': ([90], 1200, True),
    'rapid_different_amounts': ([85, 95, 105, 115], 10, False),
    'same_amount_rapid': ([90, 90, 90, 90], 10, False),
    'normal_spaced_different': ([85, 95, 105, 115], 1200, False),
    'same_amount_spaced': ([90, 90, 90, 90], 1200, False),
}


class Clock:
    value = datetime(2026, 1, 1)

    @classmethod
    def now(cls):
        return cls.value


class ObservedTester(p.FraudDetectionTester):
    def __init__(self, *args):
        super().__init__(*args)
        self.observed = []

    def evaluate_candidate(self, *args, **kwargs):
        decision = super().evaluate_candidate(*args, **kwargs)
        self.observed.append(decision)
        return decision


def run():
    root = Path(__file__).resolve().parents[1]
    with contextlib.redirect_stdout(io.StringIO()):
        df = p.engineer_features(p.load_data(root/'examples/synthetic_transactions.csv'))
        X = df[p.MODEL_FEATURES].fillna(0).copy()
        X, iso = p.train_isolation_forest(X, p.MODEL_FEATURES)
        X, svm = p.train_oneclass_svm(X, p.MODEL_FEATURES)
    records = []
    for name, (amounts, gap, new_ip) in SCENARIOS.items():
        for card in sorted(df.CardToken.unique()):
            known_ip = df[df.CardToken == card].CardHolderIp.mode().iloc[0]
            answers = []
            for amount in amounts:
                answers.extend([card, str(amount), '192.0.2.250' if new_ip else known_ip])
            answers.append('q')
            iterator = iter(answers)

            def answer(prompt):
                if 'Kart Token' in prompt:
                    Clock.value += timedelta(seconds=gap)
                return next(iterator)

            tester = ObservedTester(df, p.MODEL_FEATURES, iso, svm)
            log = io.StringIO()
            with patch.object(p, 'datetime', Clock), patch('builtins.input', side_effect=answer), contextlib.redirect_stdout(log):
                tester.run_interactive_test()
            assert 'Hata oluştu' not in log.getvalue(), log.getvalue()
            displayed = re.findall(r'└── Karar: (ANOMALİ|NORMAL)', log.getvalue())
            assert len(tester.observed) == len(amounts) == len(displayed)
            for step, (decision, printed) in enumerate(zip(tester.observed, displayed), 1):
                expected_rule = (
                    name in ('rapid_different_amounts', 'same_amount_rapid') and step >= 2)
                assert decision['rule_flag'] == expected_rule, (name, card, step, decision)
                assert decision['review_flag'] == (printed == 'ANOMALİ')
                assert decision['method_union_flag'] == any([decision['model_flag'], decision['ocsvm_flag'], decision['zscore_flag'], decision['dynamic_flag']])
                records.append(dict(scenario=name, card=card, step=step, amount=amounts[step-1], **decision))
    result = pd.DataFrame(records)
    output = root/'outputs'
    output.mkdir(exist_ok=True)
    result.to_csv(output/'scenario_results.csv', index=False)
    summary = result.groupby('scenario').agg(
        transactions=('review_flag', 'size'),
        isolation_forest=('model_flag', 'sum'),
        one_class_svm=('ocsvm_flag', 'sum'),
        zscore=('zscore_flag', 'sum'),
        dynamic_threshold=('dynamic_flag', 'sum'),
        method_union=('method_union_flag', 'sum'),
        rule_flags=('rule_flag', 'sum'),
        review_flags=('review_flag', 'sum'),
    )
    summary.to_csv(output/'scenario_summary.csv')
    print(summary.to_string())
    print(f'All rule/UI assertions passed on {len(result)} synthetic transactions.')
    print('Model outcomes are reported separately; these are not fraud accuracy metrics.')
    return result, summary


if __name__ == '__main__':
    run()
