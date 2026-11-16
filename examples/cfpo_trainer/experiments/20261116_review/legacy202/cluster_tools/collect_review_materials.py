"""Export the two-run review evidence; no GPU, no weights/images, no changes to runs. Python3.6+."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def audit_validation(path):
    counts={}; n=0; missing_ids=0
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line); n+=1
            for key in ('input','output','gts','accuracy','format','score','data_source','sample_id','output_tokens'):
                if key not in row: raise ValueError('Missing field '+key)
            if row['sample_id'] is None: missing_ids+=1
            source=str(row['data_source'])
            item=counts.setdefault(source,{'n':0,'accuracy_sum':0.,'format_sum':0.,'reward_sum':0.})
            item['n']+=1
            for key,column in [('accuracy','accuracy_sum'),('format','format_sum'),('score','reward_sum')]:
                value=float(row[key])
                if not 0<=value<=1: raise ValueError('Invalid reward component')
                item[column]+=value
    for item in counts.values():
        item['accuracy']=item['accuracy_sum']/item['n']
        item['format']=item['format_sum']/item['n']
        item['reward']=item['reward_sum']/item['n']
    return {'rows':n,'missing_sample_ids':missing_ids,'datasets':counts,'sha256':digest(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path); a=p.parse_args()
    pack=Path(__file__).resolve().parent
    candidate=Path((pack/'resume100-path.txt').read_text().strip())
    receipt=json.loads((candidate/'resume-receipt.json').read_text())
    old=Path(receipt['original_run']); new=candidate/'run'; session=Path(receipt['session'])
    if candidate.parent!=session or new.parent!=candidate: raise ValueError('Unexpected run paths')
    if (new/'exit-code.txt').read_text().strip()!='0': raise ValueError('Final job has not exited successfully')
    jobs=[(old/'job-id.txt').read_text().strip(),(new/'job-id.txt').read_text().strip()]
    checkpoint=new/'checkpoints/global_step_202'
    required=['data.pt','actor/fsdp_config.json']+['actor/{}_world_size_2_rank_{}.pt'.format(k,r)
        for r in (0,1) for k in ('model','optim','extra_state')]
    ckpt_sizes={}
    for name in required:
        path=checkpoint/name
        if not path.is_file() or path.stat().st_size==0: raise ValueError('Missing checkpoint '+name)
        ckpt_sizes[name]=path.stat().st_size
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    output=a.output or Path.home()/('cfpo-review-materials-'+stamp+'.tar.gz')
    if output.exists(): raise FileExistsError('Do not overwrite a prior evidence archive')
    stage=Path(tempfile.mkdtemp(prefix='review-export-',dir=str(session)))
    files=[]; missing=[]
    def include(source,name):
        source=Path(source)
        if source.is_file():
            files.append((source,name)); return True
        missing.append(str(source)); return False
    for label,run in [('original',old),('resume',new)]:
        for name in ('train.out','train.err','resolved-config.yaml','exit-code.txt','job-id.txt',
                     'base-commit.txt','changes.patch','changed-source.tar','submitted.sbatch',
                     'resume-config-check.json','slurm-resources.txt'):
            if (run/name).is_file(): include(run/name,'runs/'+label+'/'+name)
    for name in ('resume-receipt.json',): include(candidate/name,'runs/resume/'+name)
    for name in ('train.json','val.json','test.json','data-audit.json'):
        include(session/'manifests'/name,'manifests/'+name)
    for name in ('length-audit.json','formal-config-check.json','runtime-and-board-check.json',
                 'ray-resource-amendment.json','memory-resource-amendment.json'):
        if (session/'config'/name).is_file(): include(session/'config'/name,'preflight/'+name)
    selected=['run_formal.sh','run_grpo.sh','formal_contract.py','formal_tools.py','formal_metrics.py',
              'reference_optimizer.py','reference-optimizer-provenance.json','reference-inputs.json',
              'format_prompt/reference_base.jinja','reward_function/base.py','datasets_math_train.json']
    for name in selected:
        source=candidate/'source/examples/cfpo_trainer'/name
        if source.is_file(): include(source,'source/examples/cfpo_trainer/'+name)
    for name in ('verl/utils/dataset/cfpo.py','verl/utils/dataset/rl_dataset.py','verl/trainer/ppo/ray_trainer.py'):
        include(candidate/'source'/name,'source/'+name)
    # Canonical history: original0..100, resumed105..202. The failed101..117 branch stays in raw logs only.
    validation=[]
    for step in list(range(0,201,5))+[202]:
        run=old if step<=100 else new
        source=run/'validation'/(str(step)+'.jsonl')
        row={'step':step,'branch':'original' if step<=100 else 'resume','path':str(source),'expected_rows':2848}
        if source.is_file():
            try:
                row.update(audit_validation(source)); row['status']='complete' if row['rows']==2848 else 'count_mismatch'
                include(source,'validation/canonical/{}.jsonl'.format(step))
            except (OSError,ValueError,KeyError) as error:
                row.update(status='audit_failed',error=str(error))
        else: row['status']='missing'; missing.append(str(source))
        validation.append(row)
    if (new/'validation/100.jsonl').is_file(): include(new/'validation/100.jsonl','validation/resume_recheck/100.jsonl')
    for label,run,steps in [('original',old,[1,100]),('resume',new,[101,150,202])]:
        for step in steps:
            include(run/'generations'/(str(step)+'.jsonl'),'training_examples/{}/{}.jsonl'.format(label,step))
    test_inventory=[]
    for item in json.loads((session/'manifests/test.json').read_text(encoding='utf-8')):
        row=dict(item,status='NOT_EVALUATED',temperature=1.0,n=8,metric='Avg@8',images_audited=False)
        try:
            source=Path(item['json_path']); rows=json.loads(source.read_text(encoding='utf-8'))
            row.update(expected_samples=len(rows),annotation_sha256=digest(source))
        except (OSError,ValueError) as error: row.update(annotation_error=str(error))
        test_inventory.append(row)
    accounting=subprocess.run(['sacct','-j',','.join(jobs),'-X','--format=JobID,State,Elapsed,ReqMem,MaxRSS,NodeList,ExitCode','-P'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    (stage/'slurm-accounting.txt').write_text(accounting.stdout+'\n'+accounting.stderr,encoding='utf-8')
    report={'schema':1,'original_job':jobs[0],'resume_job':jobs[1],'checkpoint202_files':ckpt_sizes,
            'validation':validation,'test_inventory':test_inventory,'missing':missing,
            'canonical_validation_complete':all(v['status']=='complete' for v in validation),
            'native_history':'Two separate runs. Original failed101..117 is not part of canonical continuation.',
            'final_test_status':'NOT_EVALUATED','weights_included':False,'images_included':False}
    (stage/'review-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (stage/'validation-rounds.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['step','branch','status','rows','missing_sample_ids','sha256'])
        for r in validation: w.writerow([r.get(k,'') for k in ('step','branch','status','rows','missing_sample_ids','sha256')])
    for path in stage.iterdir(): include(path,'audit/'+path.name)
    with tarfile.open(str(output),'w:gz') as archive:
        for source,name in files: archive.add(str(source),arcname=name,recursive=False)
        for label,run in [('original',old),('resume',new)]:
            if (run/'swanlog').is_dir(): archive.add(str(run/'swanlog'),arcname='native/'+label)
    # Validate every archive member can be read; not only that tar creation returned0.
    with tarfile.open(str(output)) as archive:
        for member in archive:
            if member.isfile():
                stream=archive.extractfile(member)
                while stream.read(1024*1024): pass
    (pack/'review-export-path.txt').write_text(str(output)+'\n',encoding='utf-8')
    (Path(str(output)+'.sha256')).write_text(digest(output)+'  '+output.name+'\n',encoding='utf-8')
    print('REVIEW_EXPORT='+str(output))
    print('REVIEW_EXPORT_BYTES='+str(output.stat().st_size))
    print('CANONICAL_VALIDATION_ROUNDS={}/42'.format(sum(v['status']=='complete' for v in validation)))
    print('FINAL_TEST_STATUS=NOT_EVALUATED; no GPU submitted, no model/image weights exported.')
    print('Download with scp, then provide the local archive path for the full review update.')


if __name__=='__main__': main()
