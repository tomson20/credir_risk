import pandas as pd
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset

reference_data = pd.read_csv('data/train_data.csv')
current_data = pd.read_csv('data/inference_data.csv')

report = Report(metrics=[TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("concept_drift_report.html")