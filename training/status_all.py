"""Current queue status with PID identity checks; no ML model is loaded."""
import json
from pathlib import Path

from status_ast import inspect_status

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'artifacts/all_tasks/status.json'
    if not path.exists():
        print(json.dumps({'stage': 'not_started'}))
        return
    result = inspect_status(path, 'pipeline_all.py')
    terminal = {'complete', 'failed', 'local_checks_complete_visual_review_pending'}
    if result['stage'] not in terminal and result['process_identity_matches'] is False:
        result['effective_state'] = 'interrupted'
    if result['stage'].endswith('ESC50_AST'):
        job = ROOT / 'artifacts/report_esc50/ast_job_20261103_resumable'
        result['esc50'] = inspect_status(job / 'training_status.json', 'train_ast.py')
    elif result['stage'] == 'training_pump_ID_and_LOF':
        metrics = ROOT / 'artifacts/anomaly_pump/training_metrics.json'
        if metrics.exists():
            result['pump_training'] = json.loads(metrics.read_text())
    elif result['stage'] == 'training_project_AST':
        job = ROOT / 'artifacts/ast_project/job_status.json'
        if job.exists():
            result['project'] = inspect_status(job, 'train_project.py')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
