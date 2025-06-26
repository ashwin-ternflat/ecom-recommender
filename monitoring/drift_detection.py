import os
os.makedirs("../reports", exist_ok=True)



import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

baseline = pd.read_csv('../data/sessionized_events.csv')
current = pd.read_csv('../data/sessionized_events.csv')  # Replace with new data
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=baseline, current_data=current)
report.save_html("../reports/drift_report.html")
print("Drift report saved.")
