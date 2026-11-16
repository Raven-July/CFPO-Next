"""Import/backend smoke only. Does not load weights or certify visible UI contents."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import transformers
    import vllm
    import swanlab
    import swanboard
    from prometheus_fastapi_instrumentator import Instrumentator
    assert transformers.__version__=='5.3.0.dev0',transformers.__version__
    report={'scope':'Imports and native dashboard HTTP backend only; no model loaded, no training, browser review still required',
            'versions':{k:metadata.version(k) for k in ['torch','transformers','vllm','swanlab','swanboard','fastapi','starlette','setuptools','opentelemetry-api']}}
    report['cuda_available']=torch.cuda.is_available()
    report['cuda_device_count']=torch.cuda.device_count()
    if not report['cuda_available'] or report['cuda_device_count']<1:
        raise RuntimeError('CUDA unavailable inside allocated GPU container; inspect driver libraries')
    probe=torch.ones(1,device='cuda')+1
    torch.cuda.synchronize()
    if probe.item()!=2: raise RuntimeError('CUDA tensor probe failed')
    report['cuda_tensor_probe']='passed'
    print('CUDA_TENSOR_PROBE_OK',flush=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    log=args.run/'native-board-backend.log'
    with log.open('w',encoding='utf-8') as out:
        proc=subprocess.Popen([sys.executable,'-m','swanlab','watch',str(args.run/'board-smoke'),
                               '--host','127.0.0.1','--port',str(port)],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for _ in range(60):
                if proc.poll() is not None: raise RuntimeError('Native board exited; inspect '+str(log))
                try:
                    with urlopen('http://127.0.0.1:'+str(port)+'/',timeout=1) as response:
                        if response.status==200:
                            report['native_board_http_status']=200
                            break
                except OSError:
                    time.sleep(1)
            else: raise TimeoutError('Native board did not respond within the probe window')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait()
    (args.run/'runtime-and-board-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RUNTIME_IMPORTS_AND_BOARD_BACKEND_OK; browser contents not yet verified')


if __name__=='__main__': main()
