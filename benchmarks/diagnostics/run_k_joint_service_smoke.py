"""Official-parameter server boot and independent API smoke; not an official evaluation."""
import hashlib,json,os,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path


def main():
    root=Path(sys.argv[1]);stage=root/'service-smoke-01';stage.mkdir(exist_ok=False)
    assert (root/'validation-01/COMPLETE').exists(),'Frozen callpath gate not complete'
    source=root/'source';server=None
    def update(**kw):(stage/'state.json').write_text(json.dumps(dict(updated=time.time(),**kw),indent=2))
    try:
        gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True);(stage/'gpu-before.txt').write_text(gpu)
        (stage/'processes-before.txt').write_text(subprocess.check_output(['ps','-eo','pid,etime,args'],text=True))
        if 'No running processes found' not in gpu:raise RuntimeError('GPU occupied; no server started')
        with socket.socket() as sock:sock.bind(('127.0.0.1',9031))
        env=dict(os.environ)
        for k in list(env):
            if k.startswith(('VLLM_FL_','MIXED_SMALL_','GEMM_DIAG_')) or k=='PYTHONPATH':env.pop(k)
        env.update(PYTHONPATH=str(source),VLLM_PLUGINS='fl',VLLM_FL_INT8_KV_REPAIR='1',VLLM_FL_K_TOP4_JOINT='1',VLLM_FL_K_TOP4='0',VLLM_FL_V_ONLY='0',PATH='/usr/local/corex/bin:/usr/local/bin:/usr/bin:/bin',LD_LIBRARY_PATH='/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib',PYTHONUNBUFFERED='1',TORCHINDUCTOR_COMPILE_THREADS='1')
        import shutil
        shutil.copy2('/workspace/experiments/tianshu-k-top4-accuracy-20261115-02/initial-flaggems.db',stage/'flaggems.db')
        env['FLAGGEMS_DB_URL']='sqlite:///'+str(stage/'flaggems.db')
        cmd=['/workspace/tianshu-mixed-env-20261109/bin/python','/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode": "FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
        (stage/'command.json').write_text(json.dumps(dict(command=cmd,env={k:env[k] for k in ('PYTHONPATH','VLLM_PLUGINS','VLLM_FL_INT8_KV_REPAIR','VLLM_FL_K_TOP4_JOINT','VLLM_FL_K_TOP4','VLLM_FL_V_ONLY','LD_LIBRARY_PATH','FLAGGEMS_DB_URL')}),indent=2))
        (stage/'source-sha256.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')},indent=2))
        server=subprocess.Popen(cmd,env=env,stdout=(stage/'serve.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
        update(phase='starting',server_pid=server.pid)
        deadline=time.time()+1800
        while time.time()<deadline:
            if server.poll() is not None:raise RuntimeError('Joint server exited during initialization')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/health',timeout=5) as response:
                    if response.status==200:break
            except Exception:pass
            time.sleep(5)
        else:raise RuntimeError('Joint service startup timeout')
        log=(stage/'serve.log').read_text();marker='K joint v1 initialized: dynamic top4 + V INT8 scale-on-V; page32 separate_meta token_major'
        count=log.count(marker);expected=json.loads(Path('/workspace/MiniCPM5-2B/config.json').read_text())['num_hidden_layers']
        if count!=expected:raise RuntimeError(f'Joint layer coverage mismatch {count}/{expected}')
        if log.count('page32 separate_meta, 276 bytes/token/head')!=expected:raise RuntimeError('Page32 actual allocation coverage missing')
        update(phase='healthy',server_pid=server.pid,layers=count,page_size=32)
        results=[]
        for size in (1,2,7):
            prompts=[f'Request {i+size*100}: Explain one way a public library can organize a quiet reading room.' for i in range(size)]
            if size==1:prompts[0]+=(' A library keeps public records of reading rooms and asks visitors about lighting and ventilation.'*180)
            body=dict(model='minicpm',prompt=prompts if size>1 else prompts[0],temperature=0.,max_tokens=96 if size==1 else 32)
            (stage/f'request-{size}.json').write_text(json.dumps(body,indent=2))
            request=urllib.request.Request('http://127.0.0.1:9031/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=600) as response:payload=json.load(response)
            (stage/f'response-{size}.json').write_text(json.dumps(payload,indent=2))
            choices=payload.get('choices',[])
            if len(choices)!=size or any(not c.get('text','') for c in choices):raise RuntimeError('Empty or incomplete smoke response')
            results.append(dict(prompts=size,choices=len(choices),finish_reasons=[c.get('finish_reason') for c in choices],usage=payload.get('usage')))
            update(phase='smoke-running',server_pid=server.pid,results=results)
        (stage/'COMPLETE').write_text('42 layers, fixed page32, health200, API batches1/2/7; not official accuracy/performance\n')
        update(phase='complete',server_pid=server.pid,layers=count,page_size=32,results=results)
    except BaseException as exc:update(phase='failed',error=repr(exc));raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=50)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=20)
        for _ in range(25):
            gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
            if 'No running processes found' in gpu:break
            time.sleep(2)
        (stage/'gpu-after.txt').write_text(gpu)
        (stage/'processes-after.txt').write_text(subprocess.check_output(['ps','-eo','pid,etime,args'],text=True))

if __name__=='__main__':main()
