"""CPU-only integrity audit; does not import FlagGems or touch runtime caches."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / 'flaggems-source-manifest.json').read_text())
snapshot = root / 'FlagGems'
errors = []
for name, expected in manifest['files'].items():
    path = snapshot / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(name)
actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file()}
extra = actual - manifest['files'].keys()
# Generated cache files must not silently enter a distributable source snapshot.
errors.extend(sorted(extra))
for name in manifest['files']:
    if name.endswith('.py'):
        ast.parse((snapshot / name).read_bytes(), filename=name)
provider = 'src/flag_gems/runtime/backend/_iluvatar/ops/mm.py'
expected_provider = 'fd120991ed1c5cab988d02cbdb1d97ccde1e9e2eedad55affe90321d875f32ed'
assert manifest['files'][provider] == expected_provider, 'submission BF16 fixed-config provider identity changed'
if errors:
    raise SystemExit('Integrity mismatch: ' + ', '.join(errors))
print(json.dumps({'files_verified': len(actual), 'provider_identity_matches': True,
                  'gpu_validation': 'not_run', 'runtime_caches_touched': False}))
