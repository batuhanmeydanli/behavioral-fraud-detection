import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import fraud_detection_pipeline as p


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = pd.DataFrame({
            'CardToken': ['DEMO_A']*6 + ['DEMO_B']*6,
            'CardHolderIp': ['ip_1','ip_1','ip_2','ip_2','ip_3','ip_3']*2,
            'Amount': [10,20,10,40,50,600]*2,
            'CurrencyCode': ['TRY']*12,
            'ResponseCode': ['00','05','05','00','05','00']*2,
            'TransactionDate': [f'2025-01-01T00:{i*2:02d}:00Z' for i in range(6)] +
                               [f'2025-01-01T00:{i*2+1:02d}:00Z' for i in range(6)],
        })
        self.silent = contextlib.redirect_stdout(io.StringIO())
        self.silent.__enter__()
        self.addCleanup(self.silent.__exit__, None, None, None)

    def features(self, raw=None):
        return p.engineer_features(self.raw if raw is None else raw)

    def models(self):
        df = self.features()
        x = df[p.MODEL_FEATURES].fillna(0).copy()
        x, iso = p.train_isolation_forest(x, p.MODEL_FEATURES)
        x, svm = p.train_oneclass_svm(x, p.MODEL_FEATURES)
        for col in x.columns:
            if col not in p.MODEL_FEATURES:
                df[col] = x[col]
        return df, x, iso, svm

    def test_per_transaction_features(self):
        df = self.features().iloc[:6]
        self.assertEqual(df.last_10min_tx_count.tolist(), [0,1,2,3,4,5])
        self.assertEqual(df.last_1h_tx_count.tolist(), [0,1,2,3,4,5])
        self.assertEqual(df.same_amount_tx_count_last_1h.tolist(), [0,0,1,0,0,0])
        self.assertEqual(df.is_new_ip.tolist(), [1,0,1,0,1,0])
        self.assertEqual(df.consecutive_decline_count.tolist(), [0,1,2,0,1,0])
        self.assertEqual(df.time_since_last_tx.tolist(), [0,120,120,120,120,120])

    def test_retained_feature_definitions(self):
        df = self.features().iloc[:6]
        self.assertAlmostEqual(df.iloc[2].last_5_amount_avg, 40/3)
        self.assertEqual(df.daily_tx_count.tolist(), [6]*6)
        self.assertEqual(df.daily_amount_sum.tolist(), [730]*6)
        self.assertEqual(df.max_unique_ip_count_1h.tolist(), [3]*6)
        self.assertEqual(len(p.MODEL_FEATURES), 19)

    def test_unsorted_rows_keep_alignment(self):
        a = self.features()[p.MODEL_FEATURES]
        b = self.features(self.raw.sample(frac=1, random_state=7))[p.MODEL_FEATURES]
        pd.testing.assert_frame_equal(a, b)

    def test_single_card_single_row(self):
        for count in [1,6]:
            df = self.features(self.raw.iloc[:count])
            self.assertEqual(len(df), count)
            self.assertEqual(df.iloc[0].time_since_last_tx, 0)

    def test_csv_success_codes_and_validation(self):
        source = self.root/'raw.csv'
        self.raw.to_csv(source,index=False)
        self.assertEqual(p.load_data(source).iloc[0].ResponseCode,'00')
        with self.assertRaises(ValueError):
            self.features(self.raw.drop(columns='Amount'))
        raw=self.raw.copy()
        raw['Amount']=raw.Amount.astype(float)
        raw.loc[0,'Amount']=np.inf
        with self.assertRaises(ValueError):self.features(raw)

    def test_original_model_parameters_and_continuous_scores(self):
        df,x,iso,svm=self.models()
        self.assertEqual(iso.get_params()['bootstrap'],False)
        self.assertEqual(iso.get_params()['max_features'],1.0)
        self.assertEqual(iso.get_params()['n_estimators'],150)
        self.assertEqual(svm.get_params()['gamma'],'auto')
        np.testing.assert_allclose(x.isoforest_score,iso.decision_function(x[p.MODEL_FEATURES]))
        self.assertGreater(x.isoforest_score.nunique(),2)
        np.testing.assert_array_equal(x.ocsvm_flag,(svm.predict(x[p.MODEL_FEATURES])==-1).astype(int))

    def test_six_graphs_and_empty_anomalies(self):
        df,x,iso,svm=self.models()
        for col in ['isoforest_flag','ocsvm_flag']:
            df[col]=0;x[col]=0
        df['dynamic_amount_flag']=0
        p.enhanced_results_analysis(df,x,p.Config)
        functions=[
            lambda path:p.create_comprehensive_visualizations(df,x,p.Config,path),
            lambda path:p.visualize_isolation_forest_results(df,x,'DEMO_A',path),
            lambda path:p.visualize_one_class_svm_results(df,x,'DEMO_A',path),
            lambda path:p.visualize_card_transactions(df,'DEMO_A',path),
            lambda path:p.visualize_card_transactions_svm(df,'DEMO_A',path),
            lambda path:p.visualize_rolling_zscore(df,'DEMO_A',output_path=path),
        ]
        for i,fn in enumerate(functions):
            dest=self.root/f'plot_{i}.png';fn(dest)
            self.assertGreater(dest.stat().st_size,1000)

    def test_card_plot_alignment(self):
        df,x,_,_=self.models()
        order=[3,1,4,0,2,5,7,6,8,11,10,9]
        shuffled=df.iloc[order];model=x.iloc[order]
        fig=p.visualize_isolation_forest_results(shuffled,model,'DEMO_A')
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(),x.iloc[:6].isoforest_score)

    def test_simulator_matches_batch_features_and_commits_exact_candidate(self):
        df,x,iso,_=self.models()
        tester=p.FraudDetectionTester(df,p.MODEL_FEATURES,iso)
        features,history=tester.calculate_test_features('DEMO_A',99,new_ip='ip_new')
        candidate=tester._pending['DEMO_A'].copy()
        batch=p.engineer_features(pd.concat([df,pd.DataFrame([candidate])],ignore_index=True))
        actual=batch[(batch.CardToken=='DEMO_A') & (batch.TransactionDate==candidate['TransactionDate'])].iloc[0]
        np.testing.assert_allclose([features[c] for c in p.MODEL_FEATURES],actual[p.MODEL_FEATURES].astype(float).fillna(0))
        tester.predict_anomaly(features)
        tester.add_session_transaction('DEMO_A',99,'ip_new')
        self.assertEqual(tester.session_transactions['DEMO_A'][0],candidate)
        second,_=tester.calculate_test_features('DEMO_A',99,new_ip='ip_new')
        self.assertEqual(second['consecutive_same_amount'],2)
        self.assertGreater(second['time_since_last_tx'],0)
        self.assertLess(second['time_since_last_tx'],30)

    def test_simulator_consecutive_suffix_and_rules(self):
        df,x,iso,_=self.models()
        tester=p.FraudDetectionTester(df,p.MODEL_FEATURES,iso)
        for amount in [99,80,99]:
            f,h=tester.calculate_test_features('DEMO_A',amount)
            tester.add_session_transaction('DEMO_A',amount)
        self.assertEqual(f['consecutive_same_amount'],1)
        reasons,points=tester.analyze_anomaly_reasons(f,h,99)
        self.assertIsInstance(reasons,list)
        self.assertGreaterEqual(points,0)

    def test_interactive_session_and_exit(self):
        df,x,iso,_=self.models()
        tester=p.FraudDetectionTester(df,p.MODEL_FEATURES,iso)
        with patch('builtins.input',side_effect=['DEMO_A','100','ip_1','q']):
            tester.run_interactive_test()
        self.assertEqual(len(tester.session_transactions['DEMO_A']),1)

    def test_cli_persists_reports_and_skips_all_plots(self):
        source=self.root/'input.csv';self.raw.to_csv(source,index=False)
        dest=self.root/'result'
        with patch.object(p,'_finish_plot') as plots:
            p.main(['--input',str(source),'--output-dir',str(dest),'--skip-plots'])
            plots.assert_not_called()
        result=pd.read_csv(dest/'transactions_scored.csv')
        self.assertIn('ocsvm_score',result.columns)
        self.assertTrue((dest/'analysis_report.txt').exists())
        self.assertTrue((dest/'summary.json').exists())
        self.assertFalse((dest/'plots').exists())


if __name__=='__main__':
    unittest.main()
