"""User-requested 16K-only diagnostic; model accuracy is deliberately pending."""
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess

ROOT = Path('/workspace/experiments/tianshu-vonly-stage1-16k-20261115-01')
SOURCE = Path('/workspace/experiments/tianshu-vonly-stage1-20261113-01/source')
MODULE = SOURCE / 'benchmarks/diagnostics/run_tianshu_vonly_validation.py'

if __name__ == '__main__':
    ROOT.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('vonly_controller', MODULE)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    control.ROOT = ROOT
    control.SOURCE = SOURCE
    signal.signal(signal.SIGTERM, control.interrupted)
    signal.signal(signal.SIGINT, control.interrupted)
    try:
        assert (SOURCE.parent / 'operator-01/OPERATOR_PASS').exists()
        assert subprocess.check_output(['git','-C',str(control.BASE),'rev-parse','HEAD'],text=True).strip() == 'f82fa1670509f081aa13b58464cb6e5627fc10c4'
        assert not subprocess.check_output(['git','-C',str(control.BASE),'status','--porcelain'],text=True).strip()
        assert control.sha(control.BENCH) == control.sha(control.BASE/'benchmarks/benchmark_throughput_serve.py')
        original = Path('/root/.flaggems/config_cache/TunedConfig_iluvatar_triton_3_6.db')
        shutil.copy2(original, ROOT/'initial-flaggems.db')
        files = ['vllm_fl/attention/v_only.py','vllm_fl/attention/int8_kv_repair.py',
                 'vllm_fl/attention/tile_policy.py','vllm_fl/worker/model_runner.py',
                 'vllm_fl/dispatch/backends/flaggems/flaggems.py',
                 'vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py']
        (ROOT/'source-hashes.json').write_text(json.dumps({f:control.sha(SOURCE/f) for f in files},indent=2))
        control.update(phase='preflight',manager_pid=__import__('os').getpid(),
            accuracy=None,accuracy_status='not tested for this version',
            baseline_status='not run',benchmark_sha256=control.sha(control.BENCH),
            db_sha256=control.sha(original))
        control.performance(True)
        control.update(phase='complete')
    except BaseException as exc:
        control.update(phase='failed',error=repr(exc))
        raise
    finally:
        control.stop(control.client)
        control.stop(control.server)
