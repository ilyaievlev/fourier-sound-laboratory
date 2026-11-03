"""Read AST job status and verify process identity instead of trusting a stale PID."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def inspect_status(path, expected_script):
    status = json.loads(path.read_text(encoding='utf-8'))
    pid = int(status.get('pid', 0))
    command = f'Get-CimInstance Win32_Process -Filter "ProcessId = {pid}" | Select-Object Name,CommandLine | ConvertTo-Json -Compress'
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        alive = None
    else:
        process = json.loads(result.stdout) if result.stdout.strip() else {}
        alive = bool(process and process.get('Name', '').lower() in {'python.exe', 'pythonw.exe'}
                     and expected_script in process.get('CommandLine', ''))
    recorded = status.get('status', status.get('stage'))
    status['recorded_state'] = recorded
    status['process_identity_matches'] = alive
    status['effective_state'] = ('interrupted' if alive is False else 'unknown') if recorded in {
        'running', 'training', 'evaluation'} and alive is not True else recorded
    status['heartbeat_age_seconds'] = time.time() - status.get('heartbeat_unix', time.time())
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', required=True)
    args = parser.parse_args()
    job = ROOT / args.job_dir
    output = {}
    for filename, script in [('pipeline_status.json', 'pipeline_ast.py'), ('training_status.json', 'train_ast.py')]:
        if (job / filename).exists():
            output[filename] = inspect_status(job / filename, script)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
