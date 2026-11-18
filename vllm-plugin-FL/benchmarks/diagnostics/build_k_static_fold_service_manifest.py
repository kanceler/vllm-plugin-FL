"""Bind the model service to its actual implementation and independent calibration."""
import hashlib,json
from pathlib import Path


def main():
    root=Path(__file__).parents[2]
    frozen=json.loads((root/'benchmarks/diagnostics/frozen/k_static_fold_biv150_v1/manifest.json').read_text())
    prior=json.loads((root/'benchmarks/diagnostics/configs/k_top4_joint_service_v1.json').read_text())
    paths=set(prior['source_bindings'])
    paths.update(str(p.relative_to(root)) for p in (root/'vllm_fl/attention').glob('k_static_fold*.py'))
    cal=root/'vllm_fl/attention/configs/k_static_fold_minicpm5_calibration_v1.json'
    data=dict(version=1,status='model_callpath_validation_pending',identity=frozen['identity'],
        layout=frozen['layout'],records=[r for r in frozen['records'] if r['input_threshold']==64],
        source_bindings={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(paths)},
        provider_bindings=frozen['provider_bindings'],calibration_sha256=hashlib.sha256(cal.read_bytes()).hexdigest(),
        switch='VLLM_FL_K_STATIC_FOLD',global_seq_threshold=80,max_segments=32,
        reference_snapshot='k_static_fold_biv150_v1',official_accuracy='not_run')
    (root/'vllm_fl/attention/configs/k_static_fold_biv150_service_v1.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
