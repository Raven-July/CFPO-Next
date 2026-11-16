"""Prepare an isolated replay of job222487's source snapshot; Python3.6+, no GPU."""
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

REWARD_SHA='32ce9d72f58c662465ff7ab984c5ac7a161f03f25ebe9c0d80bbad56361f7c9e'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    if p.returncode: raise RuntimeError('Query failed: '+p.stderr.strip())
    return p.stdout


def check_archive(archive):
    names=[]
    for member in archive.getmembers():
        p=PurePosixPath(member.name)
        if not member.isfile() or p.is_absolute() or '..' in p.parts or '.git' in p.parts or '\\' in member.name:
            raise ValueError('Unsafe/non-file snapshot member: '+member.name)
        names.append(member.name)
    if len(names)!=len(set(names)): raise ValueError('Duplicate snapshot members')


def restore_source(old_repo, old_run, destination):
    base=(old_run/'base-commit.txt').read_text().strip()
    if len(base)!=40 or any(c not in '0123456789abcdef' for c in base): raise ValueError('Invalid base commit')
    with tarfile.open(str(old_run/'changed-source.tar')) as archive:
        check_archive(archive)
        subprocess.run(['git','clone','-q','--no-hardlinks','--no-checkout',str(old_repo),str(destination)],check=True)
        subprocess.run(['git','-C',str(destination),'config','core.autocrlf','false'],check=True)
        subprocess.run(['git','-C',str(destination),'checkout','-q','--detach',base],check=True)
        archive.extractall(str(destination))
    reward=destination/'examples/cfpo_trainer/reward_function/base.py'
    if sha(reward)!=REWARD_SHA:
        raise ValueError('Startup reward differs from reviewed original; do not mix reward versions')
    return base


def checkpoint_files(checkpoint):
    names=['data.pt','actor/fsdp_config.json']
    names += ['actor/{}_world_size_2_rank_{}.pt'.format(kind,rank)
              for rank in (0,1) for kind in ('model','optim','extra_state')]
    sizes={}
    for name in names:
        path=checkpoint/name
        if not path.is_file() or path.stat().st_size==0: raise ValueError('Missing/empty checkpoint file: '+name)
        with path.open('rb') as f: f.read(1)
        sizes[name]=path.stat().st_size
    return sizes


def main():
    pack=Path(__file__).resolve().parent
    session=Path((pack/'session-path.txt').read_text().strip())
    if session.parent!=Path('/groups/xitucheng213/home/share/u2025210309'): raise ValueError('Unexpected session')
    old=session/'train'
    if (old/'job-id.txt').read_text().strip()!='222487': raise ValueError('Unexpected original training job')
    queue=query(['squeue','-h','-u',getpass.getuser(),'-o','%A|%T'])
    for line in queue.splitlines():
        parts=line.strip().split('|')
        if len(parts)!=2: raise ValueError('Unexpected queue output')
        if parts[0].strip()=='222487': raise ValueError('Original job still active')
    rows=[s.split('|') for s in query(['sacct','-n','-X','-P','-j','222487','--format=JobIDRaw,State,ExitCode']).splitlines() if s.strip()]
    rows=[r for r in rows if len(r)>=3 and r[0].strip()=='222487']
    if len(rows)!=1 or rows[0][1].strip()!='FAILED' or rows[0][2].strip()!='1:0':
        raise ValueError('Original FAILED 1:0 receipt not confirmed')
    if (old/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip()!='100':
        raise ValueError('Latest checkpoint is no longer100; inspect progress')
    sizes=checkpoint_files(old/'checkpoints/global_step_100')
    # Bind to the previously reviewed manifests, then audit their annotation hashes.
    approved=(session/'review-approved.sha256').read_text().splitlines()
    manifests={}
    for line in approved:
        fields=line.split(None,1)
        if len(fields)==2 and fields[1].strip().startswith('manifests/'):
            name=fields[1].strip(); p=PurePosixPath(name)
            if '..' in p.parts: raise ValueError('Unsafe manifest path')
            if sha(session/name)!=fields[0]: raise ValueError('Frozen manifest changed: '+name)
            manifests[name]=fields[0]
    if set(manifests)!={'manifests/train.json','manifests/val.json','manifests/test.json','manifests/data-audit.json'}:
        raise ValueError('Expected four reviewed manifests')
    audit=json.loads((session/'manifests/data-audit.json').read_text(encoding='utf-8'))
    annotations={}
    for split in ('train','validation'):
        for row in audit[split]:
            if sha(row['source'])!=row['sha256']: raise ValueError('Annotation changed: '+row['source'])
            annotations[row['source']]=row['sha256']
    pointer=pack/'resume100-path.txt'
    if pointer.exists():
        candidate=Path(pointer.read_text().strip())
        if candidate.parent!=session or not (candidate/'resume-receipt.json').is_file(): raise ValueError('Unexpected resume pointer')
        print('RESUME100_ALREADY_PREPARED='+str(candidate)); return
    candidate=Path(tempfile.mkdtemp(prefix='resume100-',dir=str(session)))
    source=candidate/'source'
    base=restore_source(session/'repo',old,source)
    receipt={'original_job':'222487','original_run':str(old),'session':str(session),'checkpoint_step':100,
             'checkpoint':str(old/'checkpoints/global_step_100'),'checkpoint_files':sizes,
             'data_state_sha256':sha(old/'checkpoints/global_step_100/data.pt'),
             'total_steps':202,'remaining_updates':102,'source_base_commit':base,
             'startup_snapshot_sha256':sha(old/'changed-source.tar'),'reward_sha256':REWARD_SHA,
             'original_config_sha256':sha(old/'resolved-config.yaml'),'manifests':manifests,
             'annotations':annotations,'root_cause':'UNCONFIRMED_ACTOR_DISCONNECTION',
             'source_files':{str(p.relative_to(source)):sha(p) for p in source.rglob('*')
                             if p.is_file() and '.git' not in p.relative_to(source).parts}}
    (candidate/'run').mkdir()
    (candidate/'resume-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    pointer.write_text(str(candidate)+'\n',encoding='utf-8')
    print('RESUME100_PREPARED='+str(candidate))
    print('Startup source snapshot restored privately; original run untouched. No GPU job submitted.')
    print('Root cause remains unconfirmed. Continuation preserves the old reward; it is not a corrected-reward experiment.')


if __name__=='__main__': main()
