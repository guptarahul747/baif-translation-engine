import ssl
import sys
import types
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

onnx_stub = types.ModuleType("onnx")
onnx_stub.load = lambda *args, **kwargs: None
onnx_stub.save = lambda *args, **kwargs: None
sys.modules.setdefault("onnx", onnx_stub)

huggingface_hub = types.ModuleType("huggingface_hub")
huggingface_hub.snapshot_download = lambda *args, **kwargs: None
huggingface_hub.hf_hub_download = lambda *args, **kwargs: None
sys.modules.setdefault("huggingface_hub", huggingface_hub)

import download_models


class DownloadModelsSSLTests(unittest.TestCase):
    def test_build_ssl_context_returns_ssl_context(self):
        ctx = download_models.build_ssl_context()
        self.assertIsInstance(ctx, ssl.SSLContext)

    def test_https_url_opens_with_default_ca_bundle(self):
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2"
        ctx = download_models.build_ssl_context()
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, context=ctx, timeout=30) as response:
            self.assertIn(response.status, (200, 302, 303, 307, 308))


if __name__ == "__main__":
    unittest.main()
