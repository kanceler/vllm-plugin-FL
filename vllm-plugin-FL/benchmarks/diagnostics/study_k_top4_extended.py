"""CPU-only synthetic screening; thresholds are unchanged, not model acceptance."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import torch
from study_int8_kv_granularity import quant


def error(actual, reference):
    delta = actual - reference
    return {'max_abs': delta.abs().max().item(),
            'nrmse': (delta.square().mean().sqrt() /
                      reference.square().mean().sqrt().clamp_min(1e-8)).item()}


def stages(q, k, v):
    logits = q.transpose(0, 1) @ k.repeat_interleave(8, 1).permute(1, 2, 0) / 128**.5
    prob = logits.softmax(-1)
    output = prob @ v.repeat_interleave(8, 1).transpose(0, 1)
    return logits, prob, output


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=False)
    cases = [('normal', 0, 0, 'none'), ('heavy_tail', 0, 0, 'none')]
    cases += [('spikes', n, a, placement) for n in (4, 5, 8, 16)
              for a in (10, 30, 100) for placement in ('single', 'distributed')]
    rows = []
    # Identical inputs for every candidate; no eval dataset or device calls.
    with (target / 'raw.jsonl').open('w') as stream:
        for seed in range(8):
            for length in (17, 257, 4096):
                for distribution, count, amplitude, placement in cases:
                    torch.manual_seed(20261113 + seed)
                    k = torch.randn(length, 2, 128)
                    v = torch.randn_like(k)
                    q = torch.randn(4, 16, 128)
                    if distribution == 'heavy_tail':
                        # Student t, df=3, rescaled to unit theoretical variance.
                        for x in (k, v):
                            denominator = torch.randn(*x.shape, 3).square().sum(-1).sqrt()
                            x.div_(denominator.clamp_min(1e-6))
                    elif count:
                        tokens = [1] if placement == 'single' else range(0, length, 7)
                        for token in tokens:
                            for x in (k, v):
                                for head in range(2):
                                    dims = torch.randperm(128)[:count]
                                    signs = torch.randint(0, 2, (count,)) * 2 - 1
                                    x[token, head, dims] = signs.float() * amplitude
                    q, k, v = [x.bfloat16().float() for x in (q, k, v)]
                    base = stages(q, k, v)
                    k4, v0, v4 = quant(k, 128, 4), quant(v, 128), quant(v, 128, 4)
                    candidates = [('v_only', k, v0), ('k4_v0', k4, v0),
                                  ('k4_v4', k4, v4), ('k4_only', k4, v)]
                    for name, dk, dv in candidates:
                        actual = stages(q, dk, dv)
                        metrics = {label: error(a, b) for label, a, b in
                                   zip(('logits', 'probability', 'output'), actual, base)}
                        metrics['k'] = error(dk, k)
                        metrics['v'] = error(dv, v)
                        metrics['probability_tv_max'] = ((actual[1] - base[1]).abs().sum(-1) * .5).max().item()
                        # Exact output difference decomposition, not additive error norms.
                        expanded_v = dv.repeat_interleave(8, 1).transpose(0, 1)
                        expanded_base_v = v.repeat_interleave(8, 1).transpose(0, 1)
                        from_k = (actual[1] - base[1]) @ expanded_base_v
                        from_v = actual[1] @ (expanded_v - expanded_base_v)
                        metrics['decomposition_residual_max'] = (actual[2] - base[2] - from_k - from_v).abs().max().item()
                        metrics['k_output_component_max'] = from_k.abs().max().item()
                        metrics['v_output_component_max'] = from_v.abs().max().item()
                        row = dict(seed=seed, length=length, distribution=distribution,
                                   count=count, amplitude=amplitude, placement=placement,
                                   method=name, metrics=metrics,
                                   passed=metrics['output']['max_abs'] <= .20 and metrics['output']['nrmse'] <= .05)
                        rows.append(row)
                        stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(f'completed seed={seed} length={length}', flush=True)
    summary = []
    for name, _, _ in candidates:
        for category in ('all', 'normal', 'heavy_tail', 'spikes'):
            subset = [r for r in rows if r['method'] == name and
                      (category == 'all' or r['distribution'] == category)]
            worst = max(subset, key=lambda r: r['metrics']['output']['max_abs'])
            summary.append(dict(method=name, category=category, total=len(subset),
                                passed=sum(r['passed'] for r in subset),
                                max_abs=worst['metrics']['output']['max_abs'],
                                max_nrmse=max(r['metrics']['output']['nrmse'] for r in subset),
                                worst_case=worst))
    (target / 'summary.json').write_text(json.dumps(summary, indent=2))
    metadata = dict(torch=torch.__version__, python=platform.python_version(), device='cpu',
                    threads=2, thresholds=dict(max_abs=.20, nrmse=.05), rows=len(rows),
                    scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__), Path(__file__).with_name('study_int8_kv_granularity.py'))})
    (target / 'metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
