"""CPU-only integration source identity validation."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
data = json.loads((root / "quantization-integration-manifest.json").read_text())
for name, expected in data["files"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Integration binding mismatch: " + name)
print(json.dumps({"verified": len(data["files"]), "gpu_validation": "not_run"}))
