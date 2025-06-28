import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

train_data = pd.read_csv('data/train_data.csv')
inference_data = pd.read_csv('data/inference_data.csv')

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=inference_data)
report.save_html("drift_report.html")

test_suite = TestSuite(tests=[TestNumberOfDriftedColumns()])
test_suite.run(reference_data=train_data, current_data=inference_data)

if test_suite.as_dict()['summary']['failed_tests'] > 0:
    print("⚠️ Data Drift Detected!")
else:
    print("✅ No Data Drift.")