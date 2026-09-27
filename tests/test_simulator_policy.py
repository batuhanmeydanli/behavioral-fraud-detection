"""Decision integration: complementary methods plus separate session rules."""
import unittest
from unittest.mock import Mock
import pandas as pd
import fraud_detection_pipeline as p


class SimulatorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame({
            'CardToken':['demo']*10,'CardHolderIp':['known']*10,'Amount':[100]*10,
            'CurrencyCode':['TRY']*10,'ResponseCode':['00']*10,
            'TransactionDate':pd.date_range('2025-01-01',periods=10,freq='20min',tz='UTC')})
        self.tester=p.FraudDetectionTester(self.history,p.MODEL_FEATURES,Mock())
        self.features=dict.fromkeys(p.MODEL_FEATURES,0.0)
        self.features.update(consecutive_same_amount=1,rapid_session_transactions=1,seconds_since_last_test=0,
                             zscore_anomaly_flag=0,dynamic_amount_flag=0)
        self.matrix=pd.DataFrame([self.features])
        self.tester.predict_anomaly=Mock(return_value=(False,0.1,self.matrix))

    def test_each_method_can_independently_trigger_review(self):
        for method in ['if','svm','zscore','dynamic']:
            with self.subTest(method=method):
                f=self.features.copy()
                self.tester.ocsvm_model=Mock()
                self.tester.ocsvm_model.predict.return_value=[-1 if method=='svm' else 1]
                self.tester.ocsvm_model.decision_function.return_value=[-0.1 if method=='svm' else 0.1]
                self.tester.predict_anomaly.return_value=(method=='if',0.1,self.matrix)
                f['zscore_anomaly_flag']=int(method=='zscore')
                f['dynamic_amount_flag']=int(method=='dynamic')
                result=self.tester.evaluate_candidate(f,self.history,100)
                self.assertTrue(result['method_union_flag']);self.assertTrue(result['review_flag'])
                self.assertFalse(result['rule_flag']);self.assertEqual(len(result['triggered_methods']),1)

    def test_no_method_or_rule_means_no_review(self):
        result=self.tester.evaluate_candidate(self.features,self.history,100)
        self.assertFalse(result['review_flag'])
        self.assertIsNone(result['ocsvm_flag'])
        self.assertEqual(result['triggered_methods'],[])

    def test_session_flags_remain_separate(self):
        result=self.tester.evaluate_candidate(dict(self.features,rapid_session_transactions=3),self.history,100)
        self.assertFalse(result['method_union_flag']);self.assertTrue(result['rule_flag']);self.assertTrue(result['review_flag'])

    def test_time_boundaries(self):
        for gap,expected in [(0,False),(29,True),(30,False),(1200,False)]:
            f=dict(self.features,seconds_since_last_test=gap)
            self.assertEqual(self.tester.evaluate_rules(f,self.history,100)[0],expected)

    def test_no_unrequested_amount_ratio_rule(self):
        self.assertFalse(self.tester.evaluate_rules(self.features,self.history,10000)[0])
        result=self.tester.evaluate_candidate(dict(self.features,dynamic_amount_flag=1),self.history,10000)
        self.assertTrue(result['review_flag']);self.assertEqual(result['triggered_methods'],['dynamic_threshold'])


if __name__=='__main__':unittest.main()
