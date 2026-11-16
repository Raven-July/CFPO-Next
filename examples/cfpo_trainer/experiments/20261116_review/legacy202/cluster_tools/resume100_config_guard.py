"""Compare a resume config to the original run; no experimental changes allowed."""
import argparse
import hashlib
import json
from pathlib import Path

ALLOWED={'trainer.resume_mode','trainer.resume_from_path','trainer.default_local_dir',
         'trainer.rollout_data_dir','trainer.validation_data_dir','trainer.experiment_name',
         'actor_rollout_ref.rollout.trace.experiment_name',
         'ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR'}


def flatten(obj,prefix=''):
    result={}
    if isinstance(obj,dict):
        for k,v in obj.items(): result.update(flatten(v,prefix+'.'+k if prefix else k))
    else: result[prefix]=obj
    return result


def compare(old,new,old_repo,new_repo,checkpoint):
    roots=[str(old_repo),str(new_repo),str(Path(old_repo).resolve()),str(Path(new_repo).resolve())]
    def normalize(value):
        if isinstance(value,str):
            for root in roots: value=value.replace(root,'<SOURCE>')
            return value.replace('/groups/xitucheng213/','/groups/g900319/')
        if isinstance(value,list): return [normalize(v) for v in value]
        return value
    a,b=flatten(old),flatten(new); errors=[]
    for key in set(a)|set(b):
        if key not in ALLOWED and (key not in a or key not in b or normalize(a[key])!=normalize(b[key])):
            errors.append(key)
    if errors: raise ValueError('Unexpected config changes: '+', '.join(sorted(errors)))
    if b.get('trainer.resume_mode')!='resume_path' or Path(b.get('trainer.resume_from_path','')).resolve()!=Path(checkpoint).resolve():
        raise ValueError('Must explicitly resume original step100')
    if b.get('trainer.total_training_steps')!=202 or b.get('trainer.total_epochs')!=2:
        raise ValueError('Total budget must remain202 over2epochs')
    if b.get('trainer.del_local_ckpt_after_load') is not False:
        raise ValueError('Original checkpoint must not be deleted after load')
    return sorted(k for k in ALLOWED if a.get(k)!=b.get(k))


def main():
    from omegaconf import OmegaConf
    p=argparse.ArgumentParser(); p.add_argument('candidate',type=Path); args=p.parse_args()
    candidate=args.candidate; r=json.loads((candidate/'resume-receipt.json').read_text())
    for name,size in r['checkpoint_files'].items():
        p=Path(r['checkpoint'])/name
        if not p.is_file() or p.stat().st_size!=size: raise ValueError('Checkpoint changed/missing: '+name)
    if hashlib.sha256((Path(r['checkpoint'])/'data.pt').read_bytes()).hexdigest()!=r['data_state_sha256']:
        raise ValueError('Dataloader checkpoint changed')
    for name,digest in r['source_files'].items():
        if hashlib.sha256((candidate/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Prepared source changed: '+name)
    original=Path(r['original_run'])/'resolved-config.yaml'
    if hashlib.sha256(original.read_bytes()).hexdigest()!=r['original_config_sha256']:
        raise ValueError('Original resolved config changed')
    for name,digest in r['manifests'].items():
        if hashlib.sha256((Path(r['session'])/name).read_bytes()).hexdigest()!=digest: raise ValueError('Manifest changed')
    for name,digest in r['annotations'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest: raise ValueError('Annotation changed')
    old=OmegaConf.to_container(OmegaConf.load(original),resolve=True)
    new=OmegaConf.to_container(OmegaConf.load(candidate/'run/resolved-config.yaml'),resolve=True)
    differences=compare(old,new,Path(r['session'])/'repo',candidate/'source',r['checkpoint'])
    (candidate/'run/resume-config-check.json').write_text(json.dumps({'allowed_changes':differences,
        'checkpoint':r['checkpoint'],'total_steps':202,'remaining_updates':102,'root_cause':r['root_cause']},indent=2))
    print('RESUME_CONFIG_MATCH_OK: original experiment preserved; resume100 ->202')


if __name__=='__main__': main()
