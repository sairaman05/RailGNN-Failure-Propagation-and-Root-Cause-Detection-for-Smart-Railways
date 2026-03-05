"""
Airflow DAG: retrain_dag

Runs weekly. Checks if model performance has degraded.
If yes (or forced), retrains TGNN on all available data.

Schedule: every Sunday at 2 AM
"""

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty  import EmptyOperator

PROJECT_ROOT  = Path(os.environ.get("RAILWAY_PROJECT_ROOT",
                                     "/home/sairaman/Desktop/BIg_Data_Project"))
EVAL_PATH     = PROJECT_ROOT / "data" / "processed" / "evaluation_results.json"
CKPT_PATH     = PROJECT_ROOT / "checkpoints" / "best_model.pt"
RETRAIN_LOG   = PROJECT_ROOT / "data" / "processed" / "retrain_log.json"

# Retrain if accuracy drops below this threshold
ACC_THRESHOLD = 0.75

default_args = {
    "owner":        "railway",
    "retries":      1,
    "retry_delay":  timedelta(minutes=5),
    "start_date":   datetime(2024, 1, 1),
}


# ── Task functions ───────────────────────────────────────────────────

def check_model_performance(**context):
    """
    Read last evaluation results.
    Trigger retraining if accuracy < threshold or no checkpoint exists.
    """
    if not CKPT_PATH.exists():
        print("[Retrain] No checkpoint found — training from scratch.")
        context["ti"].xcom_push(key="retrain_reason", value="no_checkpoint")
        return "run_retrain"

    if not EVAL_PATH.exists():
        print("[Retrain] No evaluation results — running retrain to establish baseline.")
        context["ti"].xcom_push(key="retrain_reason", value="no_eval")
        return "run_retrain"

    with open(EVAL_PATH) as fh:
        results = json.load(fh)

    acc = results.get("overall", {}).get("accuracy", 0.0)
    auc = results.get("overall", {}).get("auc_roc",  0.0)

    print(f"[Retrain] Current model — accuracy: {acc:.4f}  AUC: {auc:.4f}")
    print(f"[Retrain] Threshold     — accuracy: {ACC_THRESHOLD}")

    if acc < ACC_THRESHOLD:
        reason = f"accuracy {acc:.4f} below threshold {ACC_THRESHOLD}"
        print(f"[Retrain] ⚠ Performance degraded: {reason}. Retraining.")
        context["ti"].xcom_push(key="retrain_reason", value=reason)
        return "run_retrain"

    print("[Retrain] ✅ Model performance OK — skipping retrain.")
    return "skip_retrain"


def run_retrain(**context):
    """Run the TGNN trainer."""
    reason = context["ti"].xcom_pull(key="retrain_reason",
                                      task_ids="check_performance")
    print(f"[Retrain] Starting retraining. Reason: {reason}")

    result = subprocess.run(
        ["python", "-m", "src.model.trainer",
         "--source", "json",
         "--epochs", "50",
         "--patience", "8"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,    # 1 hour max
    )

    if result.returncode == 0:
        print("[Retrain] ✅ Retraining complete.")
        print(result.stdout[-2000:])   # last 2000 chars
    else:
        print("[Retrain] ❌ Retraining failed:")
        print(result.stderr)
        raise RuntimeError("Model retraining failed.")


def run_evaluation(**context):
    """Evaluate the freshly trained model and save metrics."""
    print("[Retrain] Running evaluation on test set...")
    result = subprocess.run(
        ["python", "-m", "src.model.evaluator",
         "--checkpoint", str(CKPT_PATH),
         "--source", "json"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode == 0:
        print("[Retrain] ✅ Evaluation complete.")
        print(result.stdout)
    else:
        print("[Retrain] ❌ Evaluation failed:")
        print(result.stderr)
        raise RuntimeError("Evaluation failed.")


def log_retrain_event(**context):
    """Append retraining event to retrain_log.json."""
    reason = context["ti"].xcom_pull(key="retrain_reason",
                                      task_ids="check_performance") or "scheduled"
    # Load current eval results if available
    metrics = {}
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as fh:
            data = json.load(fh)
        metrics = data.get("overall", {})

    entry = {
        "timestamp":   datetime.utcnow().isoformat(),
        "reason":      reason,
        "metrics":     metrics,
        "checkpoint":  str(CKPT_PATH),
    }

    log = []
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG) as fh:
            log = json.load(fh)
    log.append(entry)

    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG, "w") as fh:
        json.dump(log, fh, indent=2)

    print(f"[Retrain] Logged retraining event → {RETRAIN_LOG}")
    print(f"[Retrain] Metrics: {metrics}")


def skip_retrain(**context):
    print("[Retrain] Model is healthy — no retraining needed this week.")


# ── DAG ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="retrain_dag",
    description="Weekly TGNN retraining if performance degrades",
    schedule_interval="0 2 * * 0",    # every Sunday at 2 AM
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["railway", "training", "phase5"],
) as dag:

    check_task = BranchPythonOperator(
        task_id="check_performance",
        python_callable=check_model_performance,
        provide_context=True,
    )

    retrain_task = PythonOperator(
        task_id="run_retrain",
        python_callable=run_retrain,
        provide_context=True,
    )

    eval_task = PythonOperator(
        task_id="run_evaluation",
        python_callable=run_evaluation,
        provide_context=True,
    )

    log_task = PythonOperator(
        task_id="log_retrain_event",
        python_callable=log_retrain_event,
        provide_context=True,
    )

    skip_task = PythonOperator(
        task_id="skip_retrain",
        python_callable=skip_retrain,
        provide_context=True,
    )

    done_task = EmptyOperator(
        task_id="done",
        trigger_rule="none_failed_min_one_success",
    )

    check_task >> retrain_task >> eval_task >> log_task >> done_task
    check_task >> skip_task >> done_task