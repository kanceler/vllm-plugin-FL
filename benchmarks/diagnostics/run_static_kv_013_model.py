"""Run candidate only, gated: smoke -> official accuracy -> official performance.

No baseline rerun. Never changes official commands, RUNS, data or model parameters.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from run_static_kv_013_gates import environment, free_resources, PYTHON

server=None
client=None
root=None
state={}
state_name="model-state.json"
skip_accuracy=False


def update(**kw):
    state.update(kw,updated=time.time())
    temp=root/(state_name+'.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(root/state_name)


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=50)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=20)


def release(stage):
    global server
    stop(server);server=None
    for _ in range(30):
        gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if 'No running processes found' in gpu:break
        time.sleep(1)
    (root/(stage+'-gpu-after.txt')).write_text(gpu)
    if 'No running processes found' not in gpu:raise RuntimeError('GPU still occupied')


def serve(stage):
    global server
    free_resources(root,stage)
    env=environment(root,stage,True)
    command=[PYTHON,'/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B',
        '--port','9031','--compilation-config','{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    record=dict(command=command,env={k:v for k,v in env.items() if k.startswith(('VLLM','PYTHONPATH','FLAGGEMS')) or k in ('PATH','LD_LIBRARY_PATH')})
    (root/(stage+'-command.json')).write_text(json.dumps(record,indent=2))
    server=subprocess.Popen(command,env=env,stdout=(root/(stage+'-serve.log')).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    update(stage=stage,status='starting',server_pid=server.pid)
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if server.poll() is not None:raise RuntimeError('Server exited '+stage)
        try:
            with urllib.request.urlopen('http://127.0.0.1:9031/health',timeout=5) as r:
                if r.status==200:break
        except Exception:pass
        time.sleep(3)
    else:raise TimeoutError('Startup deadline')
    text=(root/(stage+'-serve.log')).read_text()
    coverage=dict(initialized_layers=text.count('Static K layer initialized: layer='),
        full_decode_capture='35/35' in text,
        static_workspace='Static K workspace initialized: page32 threshold80 segments32' in text,
        compact_warmup_log='Static KV compact Decode active:' in text,
        sampling_install_log='BI-V150 row-parallel sampling installed' in text)
    (root/(stage+'-coverage.json')).write_text(json.dumps(coverage,indent=2))
    assert coverage['initialized_layers']==42,coverage
    assert coverage['full_decode_capture'] and coverage['static_workspace'],coverage
    assert coverage['compact_warmup_log'],'No compact path capture/eager evidence'
    update(status='healthy',coverage=coverage)
    return env


def run_client(command,env,stage):
    global client
    env=dict(env,VLLM_PLUGINS='')
    # Use the existing mixed Python environment; no dependency modifications.
    env['PATH']=str(Path(PYTHON).parent)+':'+env['PATH']
    (root/(stage+'-client-command.json')).write_text(json.dumps(command,indent=2))
    client=subprocess.Popen(command,env=env,cwd='/workspace',stdout=(root/(stage+'-client.log')).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    update(status='running',client_pid=client.pid)
    code=client.wait();(root/(stage+'-client.exit')).write_text(str(code))
    if code:raise RuntimeError(stage+' client exit '+str(code))


def smoke():
    serve('smoke');responses=[]
    def one(item):
        size,index=item
        body=dict(model='minicpm',messages=[dict(role='user',content=(
            f'请解释图书馆借阅记录应该包括哪些信息，并用两句话说明理由。编号{size}-{index}。'
            +'背景：社区图书馆记录借阅和维修。'*(1+index*5)))],temperature=0.,max_tokens=128)
        request=urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=600) as r:answer=json.load(r)
        message=answer['choices'][0]['message']
        assert message.get('content') or message.get('reasoning_content'),'Empty chat response'
        (root/f'smoke-{size}-{index}.json').write_text(json.dumps(dict(request=body,response=answer),indent=2))
        return dict(size=size,index=index,finish=answer['choices'][0].get('finish_reason'))
    for size in (1,2,7):
        with ThreadPoolExecutor(max_workers=size) as pool:
            responses.extend(pool.map(one,[(size,i) for i in range(size)]))
    (root/'SMOKE_COMPLETE').write_text(json.dumps(responses,indent=2))
    release('smoke')


def accuracy():
    assert (root/'SMOKE_COMPLETE').exists()
    env=serve('accuracy');work=Path('/workspace/evalscope-datasets/level3')
    before=set(work.iterdir())
    command=['/usr/local/bin/evalscope','eval','--model','minicpm',
        '--api-url','http://127.0.0.1:9031/v1/chat/completions','--api-key','EMPTY',
        '--eval-type','openai_api','--datasets','math_500','--dataset-args',
        '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}',
        '--eval-batch-size','8','--timeout','3600','--generation-config',
        '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}',
        '--work-dir',str(work),'--ignore-errors']
    run_client(command,env,'accuracy')
    dirs=[p for p in work.iterdir() if p not in before and p.is_dir()]
    reports=[p for d in dirs for p in d.glob('reports/minicpm/math_500.json')]
    assert len(reports)==1,reports
    data=json.loads(reports[0].read_text());summary=data['execution_summary']
    assert summary['requested']==105 and summary['succeeded']==105 and not summary['errored'] and not summary['incomplete'],summary
    value=float(next(m for m in data['metrics'] if m['identity']['name']=='accuracy')['score'])
    for d in dirs:shutil.copytree(d,root/'math-results'/d.name)
    (root/'accuracy-result.json').write_text(json.dumps(dict(score=value,report=str(reports[0]),execution_summary=summary),indent=2))
    release('accuracy')
    assert value>=.950,('Accuracy below gate',value)
    (root/'ACCURACY_COMPLETE').write_text(str(value)+'\n')
    update(accuracy=value,status='accuracy-complete')


def performance():
    if not skip_accuracy:
        assert (root/'ACCURACY_COMPLETE').exists()
    else:
        assert (root/'accuracy-user-cancelled.json').exists()
        assert (root/'SMOKE_COMPLETE').exists()
    bench=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
    constants={}
    for node in ast.parse(bench.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('RUNS','SKIP_FIRST'):
                    constants[target.id]=ast.literal_eval(node.value)
    assert constants==dict(RUNS=4,SKIP_FIRST=1),constants
    original=root/'official-benchmark-original.py'
    shutil.copy2(bench,original)
    before={p for p in Path('/workspace/benchmark_results').glob('*') if p.is_file()}
    env=serve('performance')
    command=['python3',str(bench),'--model','/workspace/MiniCPM5-2B',
        '--served-model-name','minicpm','--port','9031',
        '--test-cases','[[4096,1024,64,256],[16384,1024,64,128]]']
    run_client(command,env,'performance')
    assert hashlib.sha256(bench.read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest()
    results=root/'benchmark-results';results.mkdir()
    new=[p for p in Path('/workspace/benchmark_results').glob('*') if p.is_file() and p not in before]
    for p in new:shutil.copy2(p,results/p.name)
    import csv
    summaries=list(results.glob('summary_*.csv'));raw=list(results.glob('raw_runs_*.csv'))
    assert len(summaries)==1 and len(raw)==1,(summaries,raw)
    rows=list(csv.DictReader(summaries[0].open()));rounds=list(csv.DictReader(raw[0].open()))
    assert len(rows)==2 and len(rounds)==8,(len(rows),len(rounds))
    assert all(row['Run Status']=='SUCCESS' and float(row['Successful Requests'])==float(row['Num Prompts']) for row in rounds)
    release('performance')
    (root/'PERFORMANCE_COMPLETE').write_text('Official original script, 4 rounds/case, skip1; both summary rows and8 raw retained\n')


def main():
    global root, state_name, skip_accuracy
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--performance-only',action='store_true')
    a=p.parse_args();root=a.root;skip_accuracy=a.performance_only
    if skip_accuracy:state_name='performance-only-state.json'
    assert (root/'CPU_COMPLETE').exists() and (root/'GPU_COMPLETE').exists() and (root/'MICRO_COMPLETE').exists()
    assert (root/'MICRO_REVIEW.json').exists(), 'Review both positive and negative micro results first'
    assert not (root/state_name).exists(),'New model attempt directory required'
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Interrupted')))
    try:
        update(status='preflight',manager_pid=os.getpid())
        if skip_accuracy:
            assert (root/'accuracy-user-cancelled.json').exists()
            assert (root/'SMOKE_COMPLETE').exists()
            # Reuse only completed smoke from identical application sources.
            prior=json.loads((root/'source-before-model.json').read_text())
            for name,expected in prior.items():
                if name.startswith('vllm_fl/'):
                    assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==expected,name
            update(accuracy='cancelled_incomplete_by_user',smoke='reused_identical_source')
            performance()
        else:
            smoke();accuracy();performance()
        update(status='complete')
    except BaseException as e:
        update(status='failed',error=repr(e));raise
    finally:
        stop(client);stop(server)
        (root/('performance-only-final-gpu.txt' if skip_accuracy else 'model-final-gpu.txt')).write_text(subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True))


if __name__=='__main__':main()
