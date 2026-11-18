"""Serial diagnostic controller: operator gate -> math -> 16K BF16/V-only.

Uses original benchmark implementation and records every command. Not an
official two-scenario result. Never stops processes it did not start.
"""
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

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/workspace/experiments/tianshu-vonly-20261113-01')
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
        VLLM_FL_V_ONLY='1' if candidate else '0',
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
        env={k:env.get(k) for k in ('PYTHONPATH','VLLM_PLUGINS','VLLM_FL_V_ONLY','VLLM_FL_INT8_KV_REPAIR','PATH','LD_LIBRARY_PATH','FLAGGEMS_DB_URL')}),indent=2))
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
        count=text.count('V-only implementation initialized: K=BF16 V=INT8 scale=FP32')
        if count!=42:raise RuntimeError('Unexpected V-only layer coverage: '+str(count))
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
    env=serve(True,'vonly-math')
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
    run_client(command,env,'vonly-math')
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
    release('vonly-math')
    if accuracy<.95:raise RuntimeError('Model accuracy below 0.95')


def performance(candidate):
    stage='vonly-16k' if candidate else 'bf16-16k'
    env=serve(candidate,stage)
    output=Path('/workspace/benchmark_results')
    before=set(output.glob('*.csv'))
    run_client([PY,str(BENCH),'--model','/workspace/MiniCPM5-2B',
        '--served-model-name','minicpm','--port','9031','--test-cases','[[16384,1024,64,128]]'],env,stage)
    dst=ROOT/stage;dst.mkdir()
    for p in output.glob('*.csv'):
        if p not in before:shutil.copy2(p,dst/p.name)
    summaries=list(dst.glob('summary_*.csv'));raw=list(dst.glob('raw_runs_*.csv'))
    if len(summaries)!=1 or len(raw)!=1:raise RuntimeError('Missing benchmark CSV')
    if len(list(csv.DictReader(raw[0].open())))!=4:raise RuntimeError('Incomplete 4-round benchmark')
    release(stage)
    update(**{stage+'_summary':str(summaries[0])})


def interrupted(*args):raise KeyboardInterrupt('stop requested')


if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    candidate_only='--candidate-only' in sys.argv[2:]
    if (ROOT/'state.json').exists():
        if not candidate_only:raise RuntimeError('Prior state exists; use a new run')
        state.update(json.loads((ROOT/'state.json').read_text()))
        if state.get('accuracy',0)<.95:
            raise RuntimeError('Cannot resume performance without passed accuracy')
        if (ROOT/'vonly-16k-command.json').exists():
            raise RuntimeError('Candidate already started; use a new performance stage')
        state.pop('error',None)
    try:
        assert (ROOT/'operator-01/OPERATOR_PASS').exists()
        assert subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip()=='f82fa1670509f081aa13b58464cb6e5627fc10c4'
        assert not subprocess.check_output(['git','-C',str(BASE),'status','--porcelain'],text=True).strip()
        assert sha(BENCH)==sha(BASE/'benchmarks/benchmark_throughput_serve.py')
        original=Path('/root/.flaggems/config_cache/TunedConfig_iluvatar_triton_3_6.db')
        if not candidate_only:
            shutil.copy2(original,ROOT/'initial-flaggems.db')
        update(phase='preflight',manager_pid=os.getpid(),benchmark_sha256=sha(BENCH),db_sha256=sha(original))
        if not candidate_only:
            math()
            performance(False)
        performance(True)
        update(phase='complete')
    except BaseException as exc:
        update(phase='failed',error=repr(exc));raise
    finally:
        stop(client);stop(server)
