"""Run five-fold AST training, then final evaluation; resume from epoch checkpoints."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def update_codex(stage, job, extra):
    document = ROOT / 'CODEX.md'
    if not document.exists():
        return
    start, end = '<!-- CODEX_AST_STATUS_BEGIN -->', '<!-- CODEX_AST_STATUS_END -->'
    labels = {'training': 'Обучение запущено; актуальность процесса проверять status_ast.py',
              'evaluation': 'Все 5 фолдов обучены; выполняется финальная оценка',
              'complete': 'Все 5 фолдов обучены, финальная оценка завершена',
              'failed': 'Цикл остановлен с ошибкой; проверить статус и checkpoint'}
    timestamp = datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d %H:%M:%S MSK')
    lines = [start, '## Последнее событие цикла AST (автоматически)', '',
             f'Обновлено: {timestamp}. **{labels[stage]}.**', '',
             f'Каталог задания: `{job.relative_to(ROOT).as_posix()}`.',
             'Состояние `running` само по себе не подтверждает, что процесс ещё жив.', '']
    if stage == 'complete':
        import csv
        report = Path(extra['evaluation'])
        with (report / 'summary.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        lines += ['| Модель | Accuracy, mean ± SD | Macro-F1, mean ± SD |', '|---|---:|---:|']
        for row in rows:
            lines.append(f"| {row['model']} | {float(row['accuracy_mean']):.4f} ± {float(row['accuracy_std']):.4f} | {float(row['macro_f1_mean']):.4f} ± {float(row['macro_f1_std']):.4f} |")
        lines += ['', f'Сводка: `{report.relative_to(ROOT).as_posix()}/summary.csv`.',
                  'Задачи C и B этим циклом не запускались.', '']
    if stage == 'failed':
        lines += [f"Ошибка: `{extra.get('error_type')}: {extra.get('error')}`.", '']
    lines.append(end)
    original = document.read_text(encoding='utf-8')
    if start in original and end in original:
        before, remaining = original.split(start, 1)
        _, after = remaining.split(end, 1)
        updated = before + '\n'.join(lines) + after
    else:
        updated = original + '\n\n' + '\n'.join(lines) + '\n'
    pending = document.with_suffix('.md.tmp')
    pending.write_text(updated, encoding='utf-8')
    pending.replace(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    job = (ROOT / args.job_dir).resolve()
    if not job.is_relative_to((ROOT / 'artifacts').resolve()):
        parser.error('job-dir must be inside artifacts/')
    job.mkdir(parents=True, exist_ok=True)
    status_file = job / 'pipeline_status.json'
    training_status = job / 'training_status.json'
    if status_file.exists() and not args.resume:
        parser.error('Existing job requires --resume')
    if args.resume and not training_status.exists():
        parser.error('No training status to resume')
    # OS releases this lock on process exit, including an abrupt termination.
    lock = (job / 'runner.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        parser.error('This job already has a live pipeline process')
    def status(stage, **extra):
        atomic_json(status_file, dict(stage=stage, pid=os.getpid(), heartbeat_unix=time.time(), **extra))
        update_codex(stage, job, extra)
    try:
        status('training')
        old = json.loads(training_status.read_text()) if training_status.exists() else {}
        if old.get('status') != 'complete':
            command = [sys.executable, '-u', str(ROOT / 'training/train_ast.py'), '--config',
                'training/configs/ast_esc50.yaml', '--mode', 'train', '--job-status', str(training_status)]
            if args.resume:
                command.append('--resume-job')
            subprocess.run(command, cwd=ROOT, check=True)
        trained = json.loads(training_status.read_text())
        status('evaluation', training_index=trained['index'])
        subprocess.run([sys.executable, '-u', str(ROOT / 'training/evaluate_ast.py'),
            '--training-index', trained['index'], '--output-dir', str(job.parent / f'{job.name}_evaluation')],
            cwd=ROOT, check=True)
        status('complete', evaluation=str(job.parent / f'{job.name}_evaluation'))
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
