"""Test integration identity and Iluvatar routing without GPU dependencies."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'vllm_fl/quantization/integration.py'


class IdentityTests(unittest.TestCase):
    def load(self, folder):
        root = Path(folder)
        target = root / 'vllm_fl/quantization/integration.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(SOURCE.read_bytes())
        dep = root / 'bound.py'
        dep.write_text('version = 1\n')
        (root / 'quantization-integration-manifest.json').write_text(json.dumps({
            'files': {'bound.py': hashlib.sha256(dep.read_bytes()).hexdigest()}}))
        spec = importlib.util.spec_from_file_location('vllm_fl.quantization.integration', target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, dep

    def stubs(self):
        config = types.ModuleType('vllm.config')
        compilation = types.ModuleType('vllm.config.compilation')
        class Model:
            def compute_hash(self):
                return 'original-model'
        class Compile:
            def compute_hash(self):
                return 'original-compile'
        config.VllmConfig = Model
        compilation.CompilationConfig = Compile
        return {'vllm.config': config, 'vllm.config.compilation': compilation,
                'torch': types.SimpleNamespace(__version__='test-torch'),
                'triton': types.SimpleNamespace(__version__='test-triton')}, Model, Compile

    def test_identity_installs_once_and_preserves_original_contribution(self):
        with tempfile.TemporaryDirectory() as folder:
            module, _ = self.load(folder)
            stubs, model, compile_cls = self.stubs()
            with patch.dict(sys.modules, stubs), patch.dict(os.environ, {}, clear=True):
                module._install_identity()
                first = model().compute_hash()
                module._install_identity()
                self.assertEqual(first, model().compute_hash())
                self.assertNotEqual(first, 'original-model')
                self.assertNotEqual(first, compile_cls().compute_hash())

    def test_flags_change_identity_in_new_process_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            module, _ = self.load(folder)
            hashes = []
            for value in ('0', '1'):
                stubs, model, _ = self.stubs()
                with patch.dict(sys.modules, stubs), patch.dict(os.environ, {
                        'VLLM_FL_TIANSHU_W8A16_DOWN': value}, clear=True):
                    module._install_identity()
                    hashes.append(model().compute_hash())
            self.assertNotEqual(*hashes)

    def test_modified_source_is_rejected_before_hash_install(self):
        with tempfile.TemporaryDirectory() as folder:
            module, dep = self.load(folder)
            dep.write_text('version = 2\n')
            stubs, model, _ = self.stubs()
            with patch.dict(sys.modules, stubs):
                with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                    module._install_identity()
                self.assertEqual(model().compute_hash(), 'original-model')

    def test_iluvatar_only_calls_tianshu_installer(self):
        with tempfile.TemporaryDirectory() as folder:
            module, _ = self.load(folder)
            calls = []
            installer = types.ModuleType('vllm_fl.quantization.w8a16.linear')
            installer.install_tianshu_separate = lambda: calls.append('tianshu') or True
            gems = types.SimpleNamespace(runtime=types.SimpleNamespace(
                device=types.SimpleNamespace(vendor_name='iluvatar')))
            with patch.dict(sys.modules, {'flag_gems': gems,
                    'vllm_fl.quantization.w8a16.linear': installer}), \
                    patch.object(module, '_install_identity') as identity:
                self.assertTrue(module.install_quantization())
                identity.assert_called_once()
            self.assertEqual(calls, ['tianshu'])


if __name__ == '__main__':
    unittest.main()
