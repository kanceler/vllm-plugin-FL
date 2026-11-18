"""Official Level 3 accuracy controller for K top4 v2; no performance run."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/workspace/experiments/tianshu-k-top4-accuracy-20261115-01')
PY='/workspace/tianshu-mixed-env-20261109/bin/python'
BASE=Path('/workspace/tianshu-s2-track2-final-20261109')
SOURCE=ROOT/'source'
BENCH=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
server=None
client=None
state={}


def update(**kw):
    state.update(kw,updated=time.time())
    tmp=ROOT/'state.tmp'; tmp.write_text(json.dumps(state,indent=2));tmp.replace(ROOT/'state.json')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stop(p):
    if p is not None and p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=50)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=20)


def env_for(candidate,stage):
    env=dict(os.environ)
    for k in list(env):
        if k.startswith(('VLLM_FL_','MIXED_SMALL_','GEMM_DIAG_')) or k in ('PYTHONPATH','FLAGGEMS_DB_URL','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
            env.pop(k)
    env.update(PYTHONPATH=str(SOURCE if candidate else BASE),VLLM_PLUGINS='fl',
        VLLM_FL_V_ONLY='0', VLLM_FL_K_TOP4='1',
        VLLM_FL_INT8_KV_REPAIR='1' if candidate else '0',
        TORCHINDUCTOR_COMPILE_THREADS='1',PYTHONUNBUFFERED='1')
    # Preserve persistent compiler caches; use a private copy of the same tuning
    # database for each mode so prior-task experiments cannot contaminate it.
    db=ROOT/(stage+'-flaggems.db')
    shutil.copy2(ROOT/'initial-flaggems.db',db)
    env['FLAGGEMS_DB_URL']='sqlite:///'+str(db)
    return env


def preflight(stage):
    gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    (ROOT/(stage+'-gpu-before.txt')).write_text(gpu)
    (ROOT/(stage+'-processes-before.txt')).write_text(subprocess.check_output(['ps','-eo','pid,etime,args'],text=True))
    if 'No running processes found' not in gpu:
        raise RuntimeError('GPU occupied; refusing to start '+stage)
    with socket.socket() as s:s.bind(('127.0.0.1',9031))


def serve(candidate,stage):
    global server
    preflight(stage)
    env=env_for(candidate,stage)
    command=[PY,'/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
        '--compilation-config','{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    (ROOT/(stage+'-command.json')).write_text(json.dumps(dict(command=command,
        env={k:env.get(k) for k in ('PYTHONPATH','VLLM_PLUGINS','VLLM_FL_V_ONLY','VLLM_FL_K_TOP4','VLLM_FL_INT8_KV_REPAIR','PATH','LD_LIBRARY_PATH','FLAGGEMS_DB_URL')}),indent=2))
    server=subprocess.Popen(command,env=env,stdout=(ROOT/(stage+'-serve.log')).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    update(phase=stage+'-starting',server_pid=server.pid)
    deadline=time.time()+1800
    while time.time()<deadline:
        if server.poll() is not None:raise RuntimeError('Server exited: '+stage)
        try:
            with urllib.request.urlopen('http://127.0.0.1:9031/health',timeout=5) as r:
                if r.status==200:break
        except Exception:pass
        time.sleep(5)
    else:raise RuntimeError('Server startup timed out')
    text=(ROOT/(stage+'-serve.log')).read_text()
    if candidate:
        count=text.count('K top4 v2 initialized: K=INT8+BF16 top4, V=INT8 scale-on-V, FP32 scales, writer=load_w1')
        expected=json.loads(Path('/workspace/MiniCPM5-2B/config.json').read_text())['num_hidden_layers']
        if count!=expected:raise RuntimeError('Unexpected K top4 layer coverage: '+str(count))
        (ROOT/'coverage.json').write_text(json.dumps(dict(initialized_layers=count,expected_layers=expected,health_status=200,cache_bytes_per_token_head=276),indent=2))
    update(phase=stage+'-ready',healthy=True)
    return env


def run_client(command,env,stage,cwd='/workspace'):
    global client
    (ROOT/(stage+'-client-command.json')).write_text(json.dumps(command,indent=2))
    client=subprocess.Popen(command,env=env,cwd=cwd,stdout=(ROOT/(stage+'-client.log')).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    update(phase=stage+'-running',client_pid=client.pid)
    code=client.wait();(ROOT/(stage+'-client.exit')).write_text(str(code))
    if code:raise RuntimeError(stage+' client failed: '+str(code))


def release(stage):
    global server
    stop(server);server=None
    for _ in range(30):
        gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if 'No running processes found' in gpu:break
        time.sleep(2)
    (ROOT/(stage+'-gpu-after.txt')).write_text(gpu)


def math():
    env=serve(True,'k-top4-math')
    work=Path('/workspace/evalscope-datasets/level3')
    before=set(work.glob('*'))
    command=['/usr/local/bin/evalscope','eval','--model','minicpm',
        '--api-url','http://127.0.0.1:9031/v1/chat/completions','--api-key','EMPTY',
        '--eval-type','openai_api','--datasets','math_500','--dataset-args',
        '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}',
        '--eval-batch-size','8','--timeout','3600','--generation-config',
        '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}',
        '--work-dir',str(work),'--ignore-errors']
    env=dict(env,VLLM_PLUGINS='')
    run_client(command,env,'k-top4-math')
    new=[p for p in work.glob('*') if p not in before and p.is_dir()]
    reports=[p for d in new for p in d.glob('reports/minicpm/math_500.json')]
    if len(reports)!=1:raise RuntimeError('Missing or ambiguous new accuracy report')
    report=json.loads(reports[0].read_text())
    e=report['execution_summary']
    if e['requested']!=105 or e['succeeded']!=105 or e['errored'] or e['incomplete']:
        raise RuntimeError('Incomplete model accuracy: '+str(e))
    metrics=[m for m in report['metrics'] if m['identity']['name']=='accuracy']
    accuracy=float(metrics[0]['score'])
    for d in new:shutil.copytree(d,ROOT/'math-results'/d.name)
    update(accuracy=accuracy,accuracy_report=str(reports[0]),phase='math-finished')
    release('k-top4-math')
    if accuracy<.95:raise RuntimeError('Model accuracy below 0.95')



def interrupted(*args): raise KeyboardInterrupt('stop requested')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if (ROOT/'state.json').exists(): raise RuntimeError('Use a new result directory')
    try:
        assert (ROOT/'operator-02/COMPLETE').exists(), 'Operator gate not complete'
        shutil.copy2('/workspace/experiments/tianshu-vonly-scalev-20261115-01/initial-flaggems.db',ROOT/'initial-flaggems.db')
        files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*.py')}
        (ROOT/'source-sha256.json').write_text(json.dumps(files,indent=2))
        update(phase='preflight',manager_pid=os.getpid(),source_sha=sha(SOURCE/'vllm_fl/attention/k_top4_service.py'))
        math()
        update(phase='complete')
    except BaseException as exc:
        update(phase='failed',error=repr(exc));raise
    finally:
        stop(client);stop(server)
        (ROOT/'final-gpu.txt').write_text(subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True))
