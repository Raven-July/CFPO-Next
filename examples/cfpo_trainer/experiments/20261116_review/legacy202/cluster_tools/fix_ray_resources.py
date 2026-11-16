"""Amend Ray's CPU count and preserve the resource-blocked smoke; no submission."""
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime


def main(pack=None, base=None):
    pack=Path(pack) if pack is not None else Path(__file__).resolve().parent
    base=Path(base) if base is not None else Path('/groups/xitucheng213/home/share/u2025210309')
    session=Path((pack/'session-path.txt').read_text().strip())
    if session.parent!=base:
        raise ValueError('Unexpected session path')
    smoke=session/'smoke'
    job=(smoke/'job-id.txt').read_text().strip()
    if job!='222256': raise ValueError('Expected resource-blocked smoke job 222256')
    result=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%A|%T'],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    if result.returncode: raise RuntimeError('Cannot query scheduler: '+result.stderr)
    if any(line.split('|')[0].strip()==job for line in result.stdout.splitlines()):
        raise RuntimeError('Job still in live queue. Run scancel 222256, wait for it to leave, then retry.')
    record=subprocess.check_output(['sacct','-n','-X','-P','-j',job,'--format=JobIDRaw,State']).decode()
    rows=[line.strip().split('|') for line in record.splitlines() if line.strip()]
    rows=[r for r in rows if len(r)>=2 and r[0]==job]
    if len(rows)!=1 or rows[0][1].split()[0].rstrip('+') not in ('CANCELLED','FAILED','TIMEOUT'):
        raise RuntimeError('Need terminal cancelled/failed accounting record; retry after accounting updates')
    text=(smoke/'train.out').read_text(errors='replace')
    if 'No available node types can fulfill resource request' not in text:
        raise ValueError('Resource failure log does not match')
    if list((smoke/'checkpoints').glob('global_step_*')) or (session/'train').exists():
        raise ValueError('Training artifacts exist; inspect before changing source')
    if smoke.resolve().parent!=session.resolve(): raise ValueError('Unexpected smoke path')
    launcher=session/'repo/examples/cfpo_trainer/run_formal.sh'
    old=launcher.read_text(encoding='utf-8')
    anchor='export TRAIN_MAX_SAMPLES=-1 TOTAL_STEPS=202 MAX_RESPONSE_LENGTH=2048\n'
    if old.count(anchor)!=1 or 'export RAY_NUM_CPUS=' in old:
        raise ValueError('Launcher differs from reviewed pre-fix source')
    new=old.replace(anchor,anchor+'export RAY_NUM_CPUS=8\n')
    backup=session/('blocked-smoke-'+job+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir()
    (backup/'run_formal.before.sh').write_text(old,encoding='utf-8')
    launcher.write_text(new,encoding='utf-8')
    smoke.rename(backup/'smoke')
    report={'old_job':job,'old_ray_cpus':4,'new_ray_cpus':8,'slurm_requested_cpus':8,
            'worker_request_cpus':6,'worker_request_gpus':2,'training_parameters_changed':False,
            'previous_preflight_config':'config/resolved-config.yaml (before CPU amendment)',
            'next_smoke_config':'smoke/resolved-config.yaml (resolved again on compute node)',
            'launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),
            'preserved_logs':str(backup)}
    (session/'config/ray-resource-amendment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RAY_CPU_FIX_READY: 8 CPUs; no job submitted; old smoke logs preserved')


if __name__=='__main__': main()
