import importlib.util,json,os,sys,shutil,subprocess,hashlib,tarfile
from pathlib import Path
root=Path('/workspace/experiments/tianshu-k-static-down-perf-20261116-03')
prior=Path('/workspace/experiments/tianshu-k-static-service-20261116-04')
root.mkdir(exist_ok=False)
shutil.copy2(__file__,root/'controller.py')
shutil.copytree(prior/'source',root/'source')

quant=Path('/workspace/experiments/tianshu-large-m-runtime-20261113-01/source/vllm_fl/quantization/w8a16')
shutil.copytree(quant,root/'source/vllm_fl/quantization/w8a16')
p=root/'source/vllm_fl/__init__.py'
text=p.read_text(); anchor='    register_quant_linear()'
assert text.count(anchor)==1
p.write_text(text.replace(anchor,'    from vllm_fl.quantization.w8a16.linear import install_tianshu_separate\n    install_tianshu_separate()\n'+anchor))
manifest=root/'source/vllm_fl/attention/configs/k_static_fold_biv150_service_v1.json'
data=json.loads(manifest.read_text());original=dict(data['source_bindings'])
for name in data['source_bindings']:
    data['source_bindings'][name]=hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()
changed=[name for name in original if original[name]!=data['source_bindings'][name]]
assert set(changed)<= {'vllm_fl/__init__.py'},changed
manifest.write_text(json.dumps(data,indent=2)+'\n')
(root/'integration.json').write_text(json.dumps(dict(quant_source=str(quant),changed_bindings=changed,reason='install verified GEMM hook; static arithmetic and calibration unchanged'),indent=2))

shutil.copy2(prior/'initial-flaggems.db',root/'initial-flaggems.db')
sys.argv=[sys.argv[0],str(root)]
spec=importlib.util.spec_from_file_location('control',root/'source/benchmarks/diagnostics/run_k_static_fold_accuracy.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
bench=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
spec=importlib.util.spec_from_file_location('bench',bench);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
shutil.copy2(bench,root/'official-benchmark-original.py')
(root/'protocol.json').write_text(json.dumps(dict(type='two-round diagnostic, not official summary',order=[16384,4096],rounds=2,skip_first=False,benchmark_sha256=c.sha(bench),prior_accuracy=json.loads((prior/'state.json').read_text()),source_hashes={str(p.relative_to(root/'source')):c.sha(p) for p in (root/'source').rglob('*.py')}),indent=2))
try:
    original_env=c.env_for
    def combined_env(candidate,stage):
        env=original_env(candidate,stage)
        env['PYTHONPATH']=str(root/'source')+':/workspace/experiments/tianshu-boundaries-20261113-01/gems'
        for group in ('DOWN','GATE_UP','QKV','O'):env['VLLM_FL_TIANSHU_W8A16_'+group]='1' if group=='DOWN' else '0'
        (root/(stage+'-full-env.json')).write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('VLLM_FL_','VLLM_PLUGINS','PYTHONPATH','FLAGGEMS_DB_URL'))},indent=2))
        return env
    c.env_for=combined_env
    c.preflight('policy-check')
    check_env=combined_env(True,'policy-check')
    check=Path(__file__).with_name('check-k-static-down-policy.py')
    c.run_client([c.PY,str(check),str(root/'policy-check.json')],check_env,'policy-check',cwd=str(root))
    if os.environ.get('VALIDATE_ONLY')=='1':
        c.update(phase='validation-complete-performance-paused')
        sys.exit(0)
    env=c.serve(True,'static-perf')
    log=(root/'static-perf-serve.log').read_text()
    assert log.count('W8A16 enabled projection=down_proj ')==42,'Down coverage missing'
    assert all(('W8A16 enabled projection='+name+' ') not in log for name in ('qkv_proj','o_proj','gate_up_proj'))
    (root/'down-coverage.json').write_text(json.dumps(dict(down=42,qkv=0,o=0,gate_up=0)))
    env['VLLM_PLUGINS']=''
    env['PATH']='/workspace/tianshu-mixed-env-20261109/bin:'+env['PATH']
    common=b.build_common_args('/workspace/MiniCPM5-2B',9031,'minicpm')
    for length,prompts in [(16384,128),(4096,256)]:
        case=(length,1024,64,prompts)
        for round_id in (1,2):
            stage=f'{length}-round{round_id}'
            command=common+['--random-input-len',str(length),'--random-output-len','1024','--max-concurrency','64','--num-prompts',str(prompts)]
            c.run_client(command,env,stage)
            metrics=b.extract_metrics((root/(stage+'-client.log')).read_text())
            (root/(stage+'-metrics.json')).write_text(json.dumps(metrics,indent=2))
            if metrics['successful_requests']!=prompts:raise RuntimeError('Incomplete requests '+stage)
            row=b.format_result(case,metrics);row['Run Status']='SUCCESS'
            b.append_csv(row,str(root/'diagnostic-raw.csv'),b.RAW_CSV_COLUMNS)
            c.update(phase='round-complete',case=length,round=round_id)
            with tarfile.open(str(root)+'-'+stage+'.tar.gz','w:gz') as t:
                for p in root.iterdir():
                    if p.is_file():t.add(p,arcname=p.name)
    c.release('static-perf');c.update(phase='complete')
except Exception as e:
    c.update(phase='failed',error=repr(e));raise
finally:
    c.stop(c.client);c.stop(c.server)
    (root/'final-gpu.txt').write_text(subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True))
    with tarfile.open(str(root)+'.tar.gz','w:gz') as t:t.add(root,arcname=root.name)
