"""
Airflow DAG: file_watcher_dag

Runs every minute. Checks data/incoming/ for new CSV files.
For each new file: triggers the Kafka producer to stream it.

Schedule: every 1 minute
"""

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty  import EmptyOperator

PROJECT_ROOT = Path(os.environ.get("RAILWAY_PROJECT_ROOT",
                                    "/home/sairaman/Desktop/BIg_Data_Project"))
INCOMING_DIR  = PROJECT_ROOT / "data" / "incoming"
DONE_DIR      = PROJECT_ROOT / "data" / "processed_incoming"
TRACKER_FILE  = PROJECT_ROOT / "data" / "processed_incoming" / ".processed_files.txt"

default_args = {
    "owner":            "railway",
    "retries":          1,
    "retry_delay":      timedelta(minutes=1),
    "start_date":       datetime(2024, 1, 1),
}


# ── Task functions ───────────────────────────────────────────────────

def check_for_new_files(**context):
    """Scan incoming folder. Push new file paths to XCom."""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-processed filenames
    processed = set()
    if TRACKER_FILE.exists():
        processed = set(TRACKER_FILE.read_text().splitlines())

    new_files = [
        str(f) for f in sorted(INCOMING_DIR.glob("*.csv"))
        if f.name not in processed
    ]

    context["ti"].xcom_push(key="new_files", value=new_files)
    print(f"[FileWatcher] Found {len(new_files)} new file(s): {new_files}")
    return "stream_files" if new_files else "no_new_files"


def stream_files_to_kafka(**context):
    """Stream each new file through the Kafka producer."""
    new_files = context["ti"].xcom_pull(key="new_files", task_ids="check_files")
    if not new_files:
        print("[FileWatcher] No files to stream.")
        return

    for fpath in new_files:
        print(f"[FileWatcher] Streaming: {fpath}")
        result = subprocess.run(
            ["python", "-m", "src.kafka.producer",
             "--mode", "stream",
             "--csv",  fpath,
             "--speed", "0"],     # 0 = max speed (Airflow handles scheduling)
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"[FileWatcher] ✅ {Path(fpath).name} streamed successfully.")
            print(result.stdout)
        else:
            print(f"[FileWatcher] ❌ Error streaming {fpath}:")
            print(result.stderr)
            raise RuntimeError(f"Producer failed for {fpath}")

    # Mark files as processed
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = set()
    if TRACKER_FILE.exists():
        processed = set(TRACKER_FILE.read_text().splitlines())
    for fpath in new_files:
        processed.add(Path(fpath).name)
    TRACKER_FILE.write_text("\n".join(sorted(processed)))
    print(f"[FileWatcher] Marked {len(new_files)} file(s) as processed.")


def log_no_files(**context):
    print("[FileWatcher] No new files in data/incoming/ — nothing to do.")


# ── DAG ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="file_watcher_dag",
    description="Watch data/incoming/ and stream new CSVs to Kafka",
    schedule_interval="* * * * *",    # every 1 minute
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["railway", "streaming", "phase3"],
) as dag:

    check_task = BranchPythonOperator(
        task_id="check_files",
        python_callable=check_for_new_files,
        provide_context=True,
    )

    stream_task = PythonOperator(
        task_id="stream_files",
        python_callable=stream_files_to_kafka,
        provide_context=True,
    )

    no_files_task = PythonOperator(
        task_id="no_new_files",
        python_callable=log_no_files,
        provide_context=True,
    )

    done_task = EmptyOperator(
        task_id="done",
        trigger_rule="none_failed_min_one_success",
    )

    check_task >> [stream_task, no_files_task] >> done_task