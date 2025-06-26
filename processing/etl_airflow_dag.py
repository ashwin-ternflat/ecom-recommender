from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess

DEFAULT_ARGS = {'owner': 'airflow', 'retries': 1, 'retry_delay': timedelta(minutes=5)}

def run_loader(): subprocess.run(['python3', 'data_ingestion/loader.py'])
def run_sessionizer(): subprocess.run(['python3', 'processing/sessionization.py'])
def run_feature_engineering(): subprocess.run(['python3', 'features/feature_engineering.py'])
def run_model_training(): subprocess.run(['python3', 'model/train_model.py'])

with DAG(
    dag_id='ecommerce_etl_pipeline',
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:
    PythonOperator(task_id='load_user_events', python_callable=run_loader) >> \
    PythonOperator(task_id='sessionize_logs', python_callable=run_sessionizer) >> \
    PythonOperator(task_id='generate_features', python_callable=run_feature_engineering) >> \
    PythonOperator(task_id='train_model', python_callable=run_model_training)
