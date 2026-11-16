"""User-requested two-round diagnostic, adopting the existing candidate service.

Leave the official script unchanged. Preserve its first two completed4K rounds,
stop its outer four-run command, then call its original run_once twice for16K.
No source/mode/service changes between cases. NOT an official four-run summary.
"""
import csv
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time

root=Path(sys.argv[1])
state={}
old=json.loads((root/'performance-only-state.json').read_text())
manager=old['manager_pid'];client=old['client_pid'];server=old['server_pid']


def update(**kw):
    state.update(kw,at=time.time())
    p=root/'two-round-state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(root/'two-round-state.json')


def owned(pid,substring):
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert substring in command,(pid,command)
    return command


owned(manager,'run_static_kv_013_model.py')
owned(client,'benchmark_throughput_serve.py')
owned(server,'/usr/local/bin/vllm')
(root/'two-round-user-protocol.json').write_text(json.dumps(dict(
    reason='User requested two rounds per case after official benchmark already started',
    runs_per_case=2,official_runs_unchanged=4,official_skip_first_unchanged=1,
    summary='raw two rounds only; no fabricated official average',
    manager=manager,client=client,server=server,
    original_command=json.loads((root/'performance-client-command.json').read_text())),indent=2))

# Suspend only our waiting outer controller so terminating its CLI does not
# tear down the healthy model between4K/16K. Its service remains unchanged.
os.kill(manager,signal.SIGSTOP)
update(status='waiting-for-two-complete-4k-rounds',manager_paused=manager)
old_cli_stopped=False
try:
    while True:
        eligible=[]
        for p in Path('/workspace/benchmark_results').glob('raw_runs_*.csv'):
            if p.stat().st_mtime<old['updated']:continue
            with p.open() as f:rows=list(csv.DictReader(f))
            if len(rows)>=2 and all(int(x['Prefill'])==4096 for x in rows[:2]):eligible.append((p,rows))
        if eligible:
            assert len(eligible)==1,eligible
            raw,rows=eligible[0];break
        if Path(f'/proc/{client}/stat').read_text().split()[2]=='Z':
            raise RuntimeError('Official client exited before two4K rounds')
        time.sleep(.1)
    assert all(x['Run Status']=='SUCCESS' and float(x['Successful Requests'])==256 for x in rows[:2]),rows[:2]
    owned(client,'benchmark_throughput_serve.py')
    assert os.getpgid(client)==client
    os.killpg(client,signal.SIGTERM);old_cli_stopped=True
    (root/'four-run-partial-original.csv').write_bytes(raw.read_bytes())
    text=(root/'performance-client.log').read_text()
    update(status='4k-two-rounds-complete',original_raw=str(raw),
        next_round_banner_seen='Run 3/4' in text,completed_4k=rows[:2])
    env=json.loads((root/'performance-command.json').read_text())['env']
    os.environ.update(env)
    os.environ['VLLM_PLUGINS']=''
    os.environ['PATH']='/workspace/tianshu-mixed-env-20261109/bin:'+os.environ['PATH']
    bench=runpy.run_path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
    assert bench['RUNS']==4 and bench['SKIP_FIRST']==1
    output=root/'two-round-diagnostic-raw.csv'
    for row in rows[:2]:bench['append_csv'](row,str(output),bench['RAW_CSV_COLUMNS'])
    common=bench['build_common_args']('/workspace/MiniCPM5-2B',9031,'minicpm')
    for round_id in (1,2):
        update(status='running-16k',round=round_id)
        case=(16384,1024,64,128)
        metrics=bench['run_once'](case,round_id,common)
        assert metrics['successful_requests']==128,metrics
        row=bench['format_result'](case,metrics);row['Run Status']='SUCCESS'
        bench['append_csv'](row,str(output),bench['RAW_CSV_COLUMNS'])
        (root/f'two-round-16k-{round_id}-metrics.json').write_text(json.dumps(metrics,indent=2))
    update(status='measurements-complete',raw_csv=str(output))
except BaseException as e:
    update(status='failed',error=repr(e));raise
finally:
    # Pending SIGTERM is delivered on resume; the original owner cleans its
    # registered API/engine children. Do not kill unrelated GPU processes.
    if not old_cli_stopped:
        try:os.killpg(client,signal.SIGTERM)
        except ProcessLookupError:pass
    try:
        os.kill(manager,signal.SIGTERM);os.kill(manager,signal.SIGCONT)
    except ProcessLookupError:pass
    for _ in range(70):
        gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if 'No running processes found' in gpu:break
        time.sleep(1)
    (root/'two-round-final-gpu.txt').write_text(gpu)
    if state['status']=='measurements-complete':
        assert 'No running processes found' in gpu,'Resources not released'
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0
        update(status='complete')
