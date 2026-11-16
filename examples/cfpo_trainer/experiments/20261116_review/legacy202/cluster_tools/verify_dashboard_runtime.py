"""Verify only newly introduced dashboard conflicts, without loading a model."""
from importlib import metadata
import json
from pathlib import Path
import sys

expected = {
    'swanlab':'0.10.3', 'swanboard':'0.1.10b2', 'fastapi':'0.115.12',
    'starlette':'0.46.2', 'setuptools':'80.9.0', 'opentelemetry-api':'1.40.0',
}
actual={name:metadata.version(name) for name in expected}
assert actual==expected,actual
import swanlab
import swanboard
import fastapi
import starlette
import opentelemetry.sdk.trace
import prometheus_fastapi_instrumentator

report={'checked_versions':actual,'python':sys.version,'scope':'Dashboard imports and the known newly introduced container dependency conflicts only; no vLLM/model execution',
        'transformers':metadata.version('transformers'),'vllm':metadata.version('vllm')}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
print('DASHBOARD_IMPORT_CHECK_OK')
