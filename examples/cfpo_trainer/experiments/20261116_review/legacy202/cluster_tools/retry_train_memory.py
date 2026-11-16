"""Preserve confirmed pre-update OOM run; reuse successful preflight/smoke; no submission."""
import getpass
import json
from pathlib import Path
import subprocess
from datetime import datetime


def command(args, runner):
    result=runner(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    if result.returncode:
        raise RuntimeError('Query failed; no changes made: '+result.stderr.strip())
    return result.stdout


def main(pack=None, base=None, runner=None):
    pack=Path(pack) if pack is not None else Path(__file__).resolve().parent
    base=Path(base) if base is not None else Path('/groups/xitucheng213/home/share/u2025210309')
    runner=runner or subprocess.run
    session=Path((pack/'session-path.txt').read_text().strip())
    if not session.is_absolute() or session.parent!=base:
        raise ValueError('Unexpected session path')
    train=session/'train'
    if not train.exists():
        marker=session/'config/memory-resource-amendment.json'
        if marker.exists():
            print('MEMORY_RETRY_ALREADY_READY: inspect status; do not repeat recovery')
            return
        raise ValueError('No existing train directory; inspect active session')
    if train.resolve().parent!=session.resolve(): raise ValueError('Unexpected training path')
    job=(train/'job-id.txt').read_text().strip()
    if job!='222374': raise ValueError('Expected OOM job 222374; refusing another run')
    queue=command(['squeue','-h','-u',getpass.getuser(),'-o','%A|%T'],runner)
    for line in queue.splitlines():
        fields=line.strip().split('|')
        if len(fields)!=2: raise ValueError('Unexpected scheduler response')
        if fields[0].strip()==job: raise RuntimeError('Old job still in live queue; wait before retry')
    accounting=command(['sacct','-n','-X','-P','-j',job,'--format=JobIDRaw,State,ReqMem,ExitCode'],runner)
    rows=[line.strip().split('|') for line in accounting.splitlines() if line.strip()]
    rows=[row for row in rows if len(row)>=4 and row[0].strip()==job]
    if len(rows)!=1 or rows[0][1].strip().split()[0].rstrip('+')!='OUT_OF_MEMORY':
        raise RuntimeError('Accounting must confirm OUT_OF_MEMORY; no changes made')
    if not rows[0][2].strip().startswith('128G'):
        raise ValueError('Original memory request differs from reviewed 128G')
    if (session/'smoke/exit-code.txt').read_text().strip()!='0':
        raise ValueError('Successful smoke receipt missing')
    if list((train/'checkpoints').glob('global_step_*')):
        raise ValueError('Checkpoint exists: inspect resume instead of restarting')
    if list((train/'generations').glob('*.jsonl')):
        raise ValueError('Training generations exist: inspect actual progress before restarting')
    out=(train/'train.out').read_text(errors='replace')
    err=(train/'train.err').read_text(errors='replace')
    if 'Initial validation metrics' not in out or 'oom-kill' not in err:
        raise ValueError('Logs do not match reviewed initial-validation OOM')
    steps=[]
    import re
    steps=re.findall(r'training/global_step:(\d+)',out)
    if any(int(step)>0 for step in steps):
        raise ValueError('Completed training step exists: inspect before restarting')
    if '--mem="$memory"' not in (pack/'remote_steps.sh').read_text() or 'memory=256G' not in (pack/'remote_steps.sh').read_text():
        raise ValueError('Updated 256G submitter not installed')
    backup=session/('oom-train-'+job+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir()
    train.rename(backup/'train')
    report={'old_job':job,'confirmed_state':'OUT_OF_MEMORY','previous_memory':'128G',
            'next_train_memory':'256G','preserved_run':str(backup/'train'),
            'training_config_changed':False,'next_action':'remote_steps.sh train; separate explicit submission'}
    (session/'config/memory-resource-amendment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('MEMORY_RETRY_READY: next train requests 256G; no job submitted')
    print('PRESERVED_RUN='+str(backup/'train'))
    print('Existing dependencies, validation baseline, manifests and successful smoke retained.')


if __name__=='__main__': main()
