"""Sequential remaining work: A -> C -> B -> gallery/report -> full local checks.

Training programs run in separate processes so GPU/RAM are released between tasks.
The currently running AST job is watched, never launched a second time.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

from pipeline_ast import atomic_json
from status_ast import inspect_status

ROOT = Path(__file__).resolve().parents[1]


def update_documents(payload):
    for name in ['AGENTS.md', 'CODEX.md']:
        path = ROOT / name
        if not path.exists():
            continue
        start, end = '<!-- ALL_TASKS_STATUS_BEGIN -->', '<!-- ALL_TASKS_STATUS_END -->'
        stamp = datetime.now(timezone(timedelta(hours=3))).isoformat(timespec='seconds')
        block = '\n'.join([start, '## Общий цикл всех задач', '', f'Событие: {stamp}.',
            f"Этап: **{payload['stage']}**.", '',
            f"PID управляющего процесса: {payload['pid']}; проверять команду pipeline_all.py.",
            'Статус: artifacts/all_tasks/status.json; журнал: artifacts/all_tasks/stdout.log.',
            'Собственные записи не требуются. Статус отражает событие, а не гарантирует живой процесс.',
            '', '```json', json.dumps(payload, ensure_ascii=False, indent=2), '```', end])
        text = path.read_text(encoding='utf-8')
        if start in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            text = before + block + after
        else:
            text += '\n\n' + block + '\n'
        pending = path.with_suffix('.md.tmp')
        pending.write_text(text, encoding='utf-8')
        pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'artifacts/all_tasks'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'runner.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        parser.error('The all-tasks pipeline is already running')
    context = {}
    def status(stage, **extra):
        context.update(extra)
        payload = dict(stage=stage, pid=os.getpid(), heartbeat_unix=time.time(), **context)
        atomic_json(directory / 'status.json', payload)
        update_documents(payload)
        print(json.dumps(payload), flush=True)
    def run(script, *arguments, root_python=False, env=None):
        python = ROOT / '.venv/Scripts/python.exe' if root_python else Path(sys.executable)
        subprocess.run([str(python), '-u', str(ROOT / script), *arguments], cwd=ROOT, env=env, check=True)
    try:
        ast_job = ROOT / 'artifacts/report_esc50/ast_job_20261103_resumable'
        status('waiting_for_ESC50_AST')
        while True:
            current = inspect_status(ast_job / 'pipeline_status.json', 'pipeline_ast.py')
            if current['recorded_state'] == 'complete' and not current['process_identity_matches']:
                break
            if current['effective_state'] in ['failed', 'interrupted']:
                child = inspect_status(ast_job / 'training_status.json', 'train_ast.py')
                if child['process_identity_matches']:
                    time.sleep(30)
                    continue
                status('resuming_ESC50_AST')
                run('training/pipeline_ast.py', '--job-dir', str(ast_job), '--resume')
                continue
            atomic_json(directory / 'status.json', dict(stage='waiting_for_ESC50_AST', pid=os.getpid(),
                heartbeat_unix=time.time(), observed_ast_state=current['effective_state']))
            time.sleep(30)
        status('preparing_pump_data')
        run('training/prepare_pump.py')
        status('training_pump_ID_and_LOF', pump_config='training/configs/pump.yaml')
        run('training/train_anomaly.py')
        deployment = json.loads((ROOT / 'artifacts/anomaly_pump/deployment.json').read_text())
        status('preparing_project_dataset', pump_result=deployment)
        if not (ROOT / 'data/project_manifest.json').exists():
            run('training/prepare_project.py')
        benchmark_path = ROOT / 'artifacts/ast_project/benchmark.json'
        if not benchmark_path.exists():
            status('benchmarking_project_AST', project_config='training/configs/ast_project.yaml')
            run('training/train_project.py', '--mode', 'benchmark')
        benchmark = json.loads(benchmark_path.read_text())
        status('project_AST_estimate_before_training',
               project_estimated_seconds=benchmark['estimated_seconds_per_fold'],
               project_benchmark_run=benchmark['output'])
        status('training_project_AST')
        previous = ROOT / 'artifacts/ast_project/job_status.json'
        run('training/train_project.py', '--mode', 'train', *(['--resume'] if previous.exists() else []))
        status('finalizing_gallery_and_reports')
        run('training/finalize_project.py')
        status('validating_local_application')
        environment = dict(os.environ, FSL_DEVICE='cpu', HF_HUB_OFFLINE='1', FSL_SKIP_MODELS='0')
        run('manage.py', 'check', root_python=True, env=environment)
        root_python = ROOT / '.venv/Scripts/python.exe'
        subprocess.run([str(root_python), '-m', 'pytest', '-q', f'--junitxml={directory / "application_tests.xml"}'],
                       cwd=ROOT, env=environment, check=True)
        run('scripts/verify_artifacts.py', root_python=True, env=environment)
        status('building_presentation_from_final_metrics')
        node = Path(os.environ['USERPROFILE']) / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
        subprocess.run([str(node), str(ROOT / 'training/build_presentation.mjs')], cwd=ROOT, check=True)
        status('local_checks_complete_visual_review_pending', note='Models, real gallery, report and application checks completed. PPTX is structurally validated and rendered; individual final slides still require visual review. HF publication awaits separate explicit decision.')
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
