"""Check resolved formal GRPO settings before any model training."""
import argparse
import json
import hashlib
import ast
from pathlib import Path

EXPECTED = {
    'algorithm.adv_estimator':'grpo', 'algorithm.use_kl_in_reward':False,
    'algorithm.norm_adv_by_std_in_grpo':True,
    'data.train_batch_size':384, 'data.train_max_samples':-1, 'data.val_max_samples':-1,
    'data.val_batch_size':512, 'data.max_prompt_length':4096, 'data.max_response_length':2048,
    'data.seed':1, 'data.mm_processor_kwargs.min_pixels':200704, 'data.mm_processor_kwargs.max_pixels':1003520,
    'actor_rollout_ref.actor.ppo_mini_batch_size':128,
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,
    'actor_rollout_ref.actor.clip_ratio_low':0.2, 'actor_rollout_ref.actor.clip_ratio_high':0.3,
    'actor_rollout_ref.actor.clip_ratio_c':3.0, 'actor_rollout_ref.actor.loss_agg_mode':'token-mean',
    'actor_rollout_ref.actor.ppo_epochs':1, 'actor_rollout_ref.actor.use_dynamic_bsz':False,
    'actor_rollout_ref.actor.use_kl_loss':True, 'actor_rollout_ref.actor.kl_loss_coef':0.01,
    'actor_rollout_ref.actor.kl_loss_type':'low_var_kl', 'actor_rollout_ref.actor.entropy_coeff':0,
    'actor_rollout_ref.actor.optim.optimizer':'AnyPrecisionAdamW',
    'actor_rollout_ref.actor.optim.optimizer_impl':'examples.cfpo_trainer.reference_optimizer',
    'actor_rollout_ref.actor.optim.lr':1e-6, 'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio':0.05,
    'actor_rollout_ref.actor.optim.weight_decay':0.01,
    'actor_rollout_ref.actor.fsdp_config.model_dtype':'bf16', 'actor_rollout_ref.ref.fsdp_config.model_dtype':'bf16',
    'actor_rollout_ref.rollout.n':5, 'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
    'actor_rollout_ref.rollout.temperature':1.0, 'actor_rollout_ref.rollout.top_p':0.99,
    'actor_rollout_ref.rollout.val_kwargs.n':1, 'actor_rollout_ref.rollout.val_kwargs.temperature':0.0,
    'actor_rollout_ref.rollout.val_kwargs.do_sample':False,
    'trainer.n_gpus_per_node':2, 'trainer.nnodes':1, 'trainer.total_epochs':2,
    'trainer.total_training_steps':202, 'trainer.val_before_train':True, 'trainer.test_freq':5,
    'trainer.log_val_generations':30, 'trainer.save_freq':50, 'trainer.max_actor_ckpt_to_keep':5,
    'trainer.cfpo_component_metrics':True,
}


def validate(config):
    checked={}; errors=[]
    for path,expected in EXPECTED.items():
        actual=config
        try:
            for key in path.split('.'): actual=actual[key]
        except (KeyError,TypeError):
            errors.append(path+': missing'); continue
        checked[path]=actual
        if actual!=expected: errors.append('{}: expected {!r}, got {!r}'.format(path,expected,actual))
    if not {'console','swanlab'}.issubset(config.get('trainer',{}).get('logger',[])):
        errors.append('trainer.logger must include console and swanlab')
    if config.get('actor_rollout_ref',{}).get('actor',{}).get('optim',{}).get('betas') != [0.9,0.999]:
        errors.append('optimizer betas must be [0.9,0.999]')
    model=str(config.get('actor_rollout_ref',{}).get('model',{}).get('path',''))
    if not model.endswith('Qwen3-VL-2B-Instruct'): errors.append('Unexpected base model path: '+model)
    for path in ('rollout_data_dir','validation_data_dir'):
        if not config.get('trainer',{}).get(path): errors.append('trainer.'+path+' must be enabled')
    if errors: raise ValueError('FORMAL_CONFIG_MISMATCH:\n'+'\n'.join(errors))
    return {'checked_settings':checked,'base_model':model,'prompts_per_step':384,
            'responses_per_step':384*5,'responses_per_actor_minibatch':128*5}


def main():
    from omegaconf import OmegaConf
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); cfg=OmegaConf.load(a.run/'resolved-config.yaml')
    report=validate(OmegaConf.to_container(cfg,resolve=True))
    base=Path(__file__).resolve().parent
    reference=json.loads((base/'reference-inputs.json').read_text(encoding='utf-8'))
    prompt_hash=hashlib.sha256(Path(cfg.data.format_prompt).read_text(encoding='utf-8').strip().encode()).hexdigest()
    if prompt_hash!=reference['prompt_text_sha256']:
        raise ValueError('Formal system prompt differs from the pinned original')
    reward_path=Path(cfg.reward.custom_reward_function.path)
    if hashlib.sha256(reward_path.read_bytes()).hexdigest()!=reference['reward_adapter_sha256']:
        raise ValueError('Reward adapter differs from the audited source')
    if cfg.reward.custom_reward_function.name!='compute_score':
        raise ValueError('Unexpected reward function entry')
    report['prompt_text_sha256']=prompt_hash
    report['reference_inputs']=reference
    optimizer_reference=json.loads((base/'reference-optimizer-provenance.json').read_text(encoding='utf-8'))
    optimizer_source=(base/'reference_optimizer.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(optimizer_source).body if isinstance(n,ast.ClassDef) and n.name=='AnyPrecisionAdamW')
    if hashlib.sha256(ast.get_source_segment(optimizer_source,node).encode()).hexdigest()!=optimizer_reference['class_source_sha256']:
        raise ValueError('Optimizer class differs from pinned original')
    report['reference_optimizer']=optimizer_reference
    report['runtime_settings']={key:OmegaConf.select(cfg,'actor_rollout_ref.rollout.'+key) for key in
                                ['enforce_eager','enable_chunked_prefill','gpu_memory_utilization','max_num_seqs','max_model_len']}
    (a.run/'formal-config-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FORMAL_CONFIG_CONTRACT_OK')


if __name__=='__main__': main()
