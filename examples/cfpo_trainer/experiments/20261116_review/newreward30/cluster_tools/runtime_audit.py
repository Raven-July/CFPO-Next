"""Compute-node preflight: actual imports/reward/tokenizer and all train/val image decodes."""
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
import transformers
import verl
from guard30 import verify_receipt,sha

def main():
    candidate=Path(os.environ['CANDIDATE']);source=candidate/'source';run=candidate/'run'
    receipt=verify_receipt(candidate)
    resources=dict(line.split('=',1) for line in (run/'slurm-resources.txt').read_text().splitlines())
    assert int(resources['cpus'])==8 and int(resources['memory_per_node_MiB'])==262144,resources
    if shutil.disk_usage(run).free<80*1024**3:raise ValueError('Less than80GiB free for checkpoints/logs; inspect storage')
    assert Path(verl.__file__).resolve().is_relative_to(source.resolve())
    assert Path(transformers.__file__).resolve().is_relative_to(Path(os.environ['CFPO_TF_OVERLAY']).resolve())
    cfg=OmegaConf.load(run/'resolved-config.yaml');model=cfg.actor_rollout_ref.model.path
    assert AutoConfig.from_pretrained(model,local_files_only=True).model_type=='qwen3_vl'
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    token_checks={}
    for token in ('<think>','</think>'):
        ids=tokenizer.encode(token,add_special_tokens=False)
        yes=tokenizer.decode(ids,skip_special_tokens=True);no=tokenizer.decode(ids,skip_special_tokens=False)
        token_checks[token]=dict(ids=ids,skip_special_true=yes,skip_special_false=no)
        if yes!=token:raise ValueError('Reward boundary removed/changed by actual tokenizer: '+token)
    reward_path=Path(cfg.reward.custom_reward_function.path)
    assert sha(reward_path)==receipt['reward_sha256']
    spec=importlib.util.spec_from_file_location('actual_reward30',reward_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for text,answer,score in [(r'x</think>\boxed{42}',42,1.),(r'x</think>\boxed{41}',42,.1),
                               (r'\boxed{42}',42,.9),(r'\boxed{41}',42,0.)]:
        result=module.compute_score('preflight',text,answer)
        assert abs(result['score']-score)<1e-8 and result['score']==result['overall']
    processor=AutoProcessor.from_pretrained(model,local_files_only=True)
    image_hashes={};datasets=[];processor_probes=[]
    for split,manifest in [('train',cfg.data.train_files),('validation',cfg.data.val_files)]:
        entries=json.loads(Path(manifest).read_text(encoding='utf-8'))
        for entry in entries:
            annotation=Path(entry['dataset']);rows=json.loads(annotation.read_text(encoding='utf-8'))
            assert sha(annotation)==receipt['frozen_inputs'][str(annotation)]
            probed=False
            for row in rows:
                pics=row.get('images') or []
                if row['problem'].count('<image>')!=len(pics):raise ValueError('Image placeholder mismatch')
                if row.get('answer') is None:raise ValueError('Missing answer')
                for pic in pics:
                    path=Path(entry['image'])/pic;key=str(path.resolve())
                    if not probed:
                        with Image.open(path) as original:image=original.convert('RGB')
                        processed=processor.image_processor(images=[image],return_tensors='pt',
                            **OmegaConf.to_container(cfg.data.mm_processor_kwargs))
                        grid=processed['image_grid_thw'].tolist()
                        patch=processor.image_processor.patch_size
                        areas=[g[1]*g[2]*patch*patch for g in grid]
                        assert all(200704<=v<=1003520 for v in areas),(str(path),grid,areas)
                        processor_probes.append(dict(source=str(annotation),image=str(path),grid=grid,processed_areas=areas))
                        probed=True
                    if key in image_hashes:continue
                    with Image.open(path) as image:image.verify()
                    image_hashes[key]=sha(path)
                    if len(image_hashes)%5000==0:print('VERIFIED_IMAGE_FILES',len(image_hashes),flush=True)
            datasets.append(dict(split=split,source=str(annotation),N=len(rows)))
    assert sum(d['N'] for d in datasets if d['split']=='validation')==2848
    assert sum(d['N'] for d in datasets if d['split']=='train')==38870
    report=dict(reward_import=str(module.__file__),reward_sha256=sha(reward_path),
        verl_import=str(verl.__file__),transformers_import=str(transformers.__file__),
        tokenizer=token_checks,model=model,datasets=datasets,image_files=len(image_hashes),processor_probes=processor_probes,
        image_hash_manifest='image-hashes.json',note='No model weights loaded and no test-set inference executed')
    (run/'image-hashes.json').write_text(json.dumps(image_hashes,indent=2),encoding='utf-8')
    (run/'runtime-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ACTUAL_REWARD_TOKENIZER_ALL_IMAGES_OK',json.dumps(report),flush=True)

if __name__=='__main__':main()
