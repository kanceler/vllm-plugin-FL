"""Official Level 3 accuracy controller for static K fold service; no performance run."""
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

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/workspace/experiments/tianshu-k-static-service-20261116-01')
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
        VLLM_FL_V_ONLY='0', VLLM_FL_K_TOP4='0', VLLM_FL_K_TOP4_JOINT='0', VLLM_FL_K_STATIC_FOLD='1',
        PATH='/usr/local/corex/bin:/usr/local/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH='/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib',
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
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(('127.0.0.1',9031))==0:
            raise RuntimeError('Active listener still owns9031')
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(('0.0.0.0',9031))


def serve(candidate,stage):
    global server
    preflight(stage)
    env=env_for(candidate,stage)
    command=[PY,'/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
        '--compilation-config','{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    (ROOT/(stage+'-command.json')).write_text(json.dumps(dict(command=command,
        env={k:env.get(k) for k in ('PYTHONPATH','VLLM_PLUGINS','VLLM_FL_V_ONLY','VLLM_FL_K_TOP4','VLLM_FL_K_TOP4_JOINT','VLLM_FL_K_STATIC_FOLD','VLLM_FL_INT8_KV_REPAIR','PATH','LD_LIBRARY_PATH','FLAGGEMS_DB_URL')}),indent=2))
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
        count=text.count('Static K layer initialized: layer=')
        expected=json.loads(Path('/workspace/MiniCPM5-2B/config.json').read_text())['num_hidden_layers']
        if count!=expected:raise RuntimeError('Unexpected static K layer coverage: '+str(count))
        if '35/35' not in text:raise RuntimeError('Expected 35 actual FULL Decode captures not confirmed')
        if 'Static K workspace initialized: page32 threshold80 segments32' not in text:
            raise RuntimeError('Static MetadataBuilder was not used')
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
    env=serve(True,'k-static-math')
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
    run_client(command,env,'k-static-math')
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
    release('k-static-math')
    if accuracy<.95:raise RuntimeError('Model accuracy below 0.95')



def smoke():
    from concurrent.futures import ThreadPoolExecutor
    env=serve(True,'k-static-smoke');results=[]
    def request_one(item):
        size,i,prompt=item
        body=dict(model='minicpm',messages=[dict(role='user',content=prompt)],temperature=0.,max_tokens=128)
        (ROOT/f'smoke-chat-request-{size}-{i}.json').write_text(json.dumps(body,indent=2))
        request=urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=600) as response:payload=json.load(response)
        (ROOT/f'smoke-chat-response-{size}-{i}.json').write_text(json.dumps(payload,indent=2))
        choices=payload.get('choices',[])
        if len(choices)!=1:raise RuntimeError('Static chat smoke missing choice')
        message=choices[0].get('message',{})
        if not (message.get('content') or message.get('reasoning_content')):
            raise RuntimeError('Static chat smoke empty output')
        return dict(index=i,usage=payload.get('usage'),finish_reason=choices[0].get('finish_reason'))
    for size in (1,2,7):
        prompts=[f'请解释怎样整理图书馆的分类目录，并说明一种容易维护的记录方法。记录编号{i+size*10}。'+('背景：社区图书馆每天记录借阅与维修情况。'*(i*5+1)) for i in range(size)]
        if size==1:prompts[0]+='公共阅览室需要保存照明、通风和设备检查的记录。'*120
        with ThreadPoolExecutor(max_workers=size) as pool:
            output=list(pool.map(request_one,[(size,i,p) for i,p in enumerate(prompts)]))
        results.append(dict(prompts=size,responses=output))
        update(phase='smoke-running',smoke_results=results)
    (ROOT/'MODEL_SMOKE_COMPLETE').write_text('42 calibrated layers, native joint QKV backend, 35 actual FULL graphs, chat API concurrency1/2/7; not official accuracy\n')
    release('k-static-smoke')



def interrupted(*args): raise KeyboardInterrupt('stop requested')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if (ROOT/'state.json').exists(): raise RuntimeError('Use a new result directory')
    try:
        assert (ROOT/'validation-01/COMPLETE').exists(), 'Operator gate not complete'
        shutil.copy2('/workspace/experiments/tianshu-vonly-scalev-20261115-01/initial-flaggems.db',ROOT/'initial-flaggems.db')
        files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*.py')}
        (ROOT/'source-sha256.json').write_text(json.dumps(files,indent=2))
        update(phase='preflight',manager_pid=os.getpid(),source_sha=sha(SOURCE/'vllm_fl/attention/k_static_fold_service.py'))
        proof=ROOT/'MODEL_SMOKE_PROVENANCE.json'
        if proof.exists():
            record=json.loads(proof.read_text());prior=Path(record['source_experiment'])
            assert (prior/'MODEL_SMOKE_COMPLETE').exists(),'Prior model smoke incomplete'
            relative='vllm_fl/attention/configs/k_static_fold_biv150_service_v1.json'
            assert sha(SOURCE/relative)==record['service_manifest_sha256']==sha(prior/'source'/relative)
            for relative in json.loads((SOURCE/relative).read_text())['source_bindings']:
                assert sha(SOURCE/relative)==sha(prior/'source'/relative),relative
            update(phase='verified-reused-model-smoke',model_smoke_source=str(prior))
        else:
            smoke()
        math()
        update(phase='complete')
    except BaseException as exc:
        update(phase='failed',error=repr(exc));raise
    finally:
        stop(client);stop(server)
        (ROOT/'final-gpu.txt').write_text(subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True))
