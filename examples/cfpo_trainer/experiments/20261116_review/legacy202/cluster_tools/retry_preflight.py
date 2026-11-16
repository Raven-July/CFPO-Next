"""Preserve a failed CLI preflight, install verified sources, leave resubmission explicit."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime


def confirm_failed_job(job, user, runner=None):
    """Query the user's live queue, then require the failed accounting record."""
    runner=runner or subprocess.run
    queue=runner(['squeue','-h','-u',user,'-o','%A|%T'],
                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    if queue.returncode:
        raise RuntimeError('Cannot query live queue; no changes made: '+queue.stderr.strip())
    for line in queue.stdout.splitlines():
        fields=line.strip().split('|')
        if len(fields)!=2:
            raise RuntimeError('Unexpected squeue format; no changes made')
        if fields[0].strip()==job:
            raise RuntimeError('Job still present in live queue: '+fields[1].strip())
    record=runner(['sacct','-n','-X','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],
                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    if record.returncode:
        raise RuntimeError('Cannot query accounting; no changes made: '+record.stderr.strip())
    rows=[line.strip().split('|') for line in record.stdout.splitlines() if line.strip()]
    rows=[row for row in rows if len(row)>=3 and row[0].strip()==job]
    if len(rows)!=1:
        raise RuntimeError('Accounting record not available or ambiguous; retry later, no changes made')
    state=rows[0][1].strip().split()[0].rstrip('+')
    if state!='FAILED' or rows[0][2].strip()!='2:0':
        raise RuntimeError('Expected FAILED with exit 2:0, found '+str(rows[0]))
    print('FAILED_JOB_CONFIRMED='+job+'; live queue absent; accounting FAILED 2:0')


def main(pack=None, base=None):
    pack=Path(pack) if pack is not None else Path(__file__).resolve().parent
    session=Path((pack/'session-path.txt').read_text().strip())
    base=Path(base) if base is not None else Path('/groups/xitucheng213/home/share/u2025210309')
    if not session.is_absolute() or session.parent!=base:
        raise ValueError('Unexpected session path')
    prep=session/'prep-batch'
    job=(prep/'job-id.txt').read_text().strip()
    if not job.isdigit(): raise ValueError('Invalid job id')
    import getpass
    confirm_failed_job(job,getpass.getuser())
    if (prep/'exit-code.txt').read_text().strip()!='2':
        raise ValueError('This recovery expects the recorded exit-code 2 CLI failure')
    error=(prep/'prepare.err').read_text(errors='replace')
    if 'unrecognized arguments:' not in error or 'main_ppo.py' not in error:
        raise ValueError('Failure does not match the reviewed Hydra CLI issue')
    for stage in ('smoke','train','eval'):
        if (session/stage).exists(): raise ValueError('Later stage exists; do not modify source: '+stage)
    if (session/'manifests/train.json').exists():
        raise ValueError('Manifests already prepared; inspect instead of overwriting')
    for stage in ('prep-batch','config'):
        source=session/stage
        if source.exists() and source.resolve().parent!=session.resolve():
            raise ValueError('Stage directory resolves outside checked session')
    manifest=json.loads((pack/'sources.json').read_text(encoding='utf-8'))
    repo=session/'repo'
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()!=manifest['base_commit']:
        raise ValueError('Base commit differs')
    expected={row['path']:row['sha256'] for row in manifest['files']}
    with tarfile.open(str(pack/'changes.tar')) as archive:
        if set(archive.getnames())!=set(expected): raise ValueError('Unexpected source members')
        for member in archive.getmembers():
            rel=Path(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe source path')
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=expected[member.name]:
                raise ValueError('Source archive hash differs')
        archive.extractall(str(repo))
    for name,digest in expected.items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest: raise ValueError('Installed source hash differs')
    # Rename only the checked session's failed stage directories, preserving all logs.
    backup=session/('failed-preflight-'+job+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir()
    for stage in ('prep-batch','config'):
        source=session/stage
        if source.exists():
            source.rename(backup/stage)
    print('PREFLIGHT_RETRY_READY')
    print('PRESERVED_FAILED_LOGS='+str(backup))
    print('Image repair, dependencies and session.env preserved; no GPU job submitted.')


if __name__=='__main__': main()
