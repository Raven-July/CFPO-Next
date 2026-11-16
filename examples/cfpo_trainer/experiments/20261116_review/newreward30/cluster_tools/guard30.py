"""Strict old/new config comparison for an independent 30-step reward check; no GPU imports."""
import hashlib
import json
from pathlib import Path

EXPECTED={
 'trainer.total_training_steps':30,'trainer.save_freq':10,'trainer.resume_mode':'disable',
 'trainer.resume_from_path':None,'actor_rollout_ref.actor.optim.lr_warmup_steps':10,
 'data.max_response_length':2048,'actor_rollout_ref.rollout.max_model_len':6144,
 'data.train_batch_size':384,'actor_rollout_ref.rollout.n':5,
 'trainer.n_gpus_per_node':2,'ray_kwargs.ray_init.num_cpus':8,
 'trainer.test_freq':5,'trainer.val_before_train':True,'trainer.log_val_generations':30,
}
CHANGED={'trainer.total_training_steps','trainer.save_freq','trainer.resume_mode',
 'trainer.resume_from_path','actor_rollout_ref.actor.optim.lr_warmup_steps',
 'trainer.experiment_name','trainer.default_local_dir','trainer.rollout_data_dir',
 'trainer.validation_data_dir','actor_rollout_ref.rollout.trace.experiment_name',
 'ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_package(pack):
    pack=Path(pack)
    for name,digest in json.loads((pack/'payload-sha256.json').read_text()).items():
        if sha(pack/name)!=digest:raise ValueError('Package changed: '+name)

def flatten(obj,prefix=''):
    if not isinstance(obj,dict):return {prefix:obj}
    result={}
    for k,v in obj.items():result.update(flatten(v,prefix+'.'+k if prefix else k))
    return result

def compare(old,new,old_root,new_root):
    a,b=flatten(old),flatten(new)
    def norm(x):
        if isinstance(x,str):
            for root in [str(old_root),str(new_root),str(Path(old_root).resolve()),str(Path(new_root).resolve())]:x=x.replace(root,'<SOURCE>')
            return x.replace('/groups/xitucheng213/','/groups/g900319/')
        if isinstance(x,list):return [norm(v) for v in x]
        return x
    errors=[k for k in set(a)|set(b) if k not in CHANGED and norm(a.get(k))!=norm(b.get(k))]
    errors += [k for k,v in EXPECTED.items() if b.get(k)!=v]
    if not str(b.get('actor_rollout_ref.model.path','')).endswith('/Qwen3-VL-2B-Instruct'):errors.append('base model')
    if errors:raise ValueError('Unexpected validation30 config: '+', '.join(sorted(set(errors))))
    return {k:dict(original=a.get(k),validation30=b.get(k)) for k in sorted(CHANGED) if a.get(k)!=b.get(k)}

def verify_receipt(candidate):
    candidate=Path(candidate);r=json.loads((candidate/'receipt.json').read_text())
    for name,digest in r['source_files'].items():
        if sha(candidate/'source'/name)!=digest:raise ValueError('Prepared source changed: '+name)
    for path,digest in r['frozen_inputs'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    return r

def main():
    import argparse
    from omegaconf import OmegaConf
    p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);a=p.parse_args()
    verify_package(Path(__file__).resolve().parent)
    r=verify_receipt(a.candidate)
    old=OmegaConf.to_container(OmegaConf.load(r['original_config']),resolve=True)
    new=OmegaConf.to_container(OmegaConf.load(a.candidate/'run/resolved-config.yaml'),resolve=True)
    differences=compare(old,new,Path(r['session'])/'repo',a.candidate/'source')
    (a.candidate/'run/config-guard.json').write_text(json.dumps(differences,indent=2),encoding='utf-8')
    print('VALIDATION30_CONFIG_AND_SOURCE_OK: fresh base model; warmup10; max_response2048; 30 updates')

if __name__=='__main__':main()
