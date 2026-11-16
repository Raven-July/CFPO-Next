"""Preparation/audit commands. prepare/summarize use Python 3.6+, no GPU."""
import argparse
import hashlib
import json
from pathlib import Path

DATA_ROOT = Path('/groups/xitucheng213/home/share/yzy/Counterfact-Projects/Datasets')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rows_audit(path, images, verify=False):
    rows = read(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Empty/non-list source: ' + str(path))
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row.get('problem'), str) or row.get('answer') is None:
            raise ValueError('Missing problem/answer: {}:{}'.format(path, i))
        pics = row.get('images') or []
        if not isinstance(pics, list) or row['problem'].count('<image>') != len(pics):
            raise ValueError('Image placeholders mismatch: {}:{}'.format(path, i))
        for pic in pics:
            p = Path(images) / pic
            if not p.is_file():
                raise FileNotFoundError(p)
            if verify and str(p) not in seen:
                from PIL import Image
                with Image.open(p) as image:
                    image.verify()
                seen.add(str(p))
    return {'source': str(path), 'rows': len(rows),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def prepare(args):
    root, out = args.data_root, args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'train.json').exists():
        raise FileExistsError('Use a fresh manifest directory')
    train = [{'dataset': str(root/'V3-math/ViRL39K_train.json'), 'image': str(root/'ViRL39K_images')}]
    validation = [
        ('V3-math/geometry3k_val_V3.json','geometry3k_images'),
        ('V3-math/LogicVista_val_V3.json','LogicVista_images'),
        ('V3-math/MathVerse_val_V3.json','MathVerse_images'),
        ('V3/C-VQA-Real_val_V3.json','COCO_val2014_images'),
        ('V3/MARS_Bench_val_V3.json','COCO_val2014_images'),
    ]
    val = [{'dataset': str(root/a), 'image': str(root/b)} for a,b in validation]
    mathverse_images = getattr(args, 'mathverse_images', None)
    if mathverse_images:
        for item in val:
            if Path(item['image']).name == 'MathVerse_images':
                item['image'] = str(mathverse_images)
    source = Path(__file__).resolve().parents[2]/'Counterfactual-Eval/datasets_config_math.json'
    test = read(source)
    if len(test) != 10 or len({v['dataset_name'] for v in test}) != 10:
        raise ValueError('Expected ten distinct test datasets')
    for item in test:
        rel = item['json_path'].split('/Datasets/',1)[1]
        item['json_path'] = str(root/rel)
        item['image_root'] = str(root/Path(item['image_root']).name)
        if mathverse_images and Path(item['image_root']).name == 'MathVerse_images':
            item['image_root'] = str(mathverse_images)
    audit = {}
    for label, specs in [('train',train),('validation',val),('test',test)]:
        audit[label] = []
        for item in specs:
            a=item.get('dataset',item.get('json_path')); b=item.get('image',item.get('image_root'))
            if label=='test' and getattr(args,'scope','all')=='training':
                audit[label].append({'source':a,'status':'deferred_to_final_evaluation','image_decode_verified':False})
                continue
            print('AUDIT_SOURCE:',label,a,flush=True)
            save(out/'data-audit.json',dict(audit,in_progress={'split':label,'source':a}))
            try:
                record=rows_audit(a,b,args.verify_images)
            except (OSError,ValueError) as error:
                audit[label].append({'source':a,'status':'failed','error_type':type(error).__name__,
                                     'error':str(error),'filename':getattr(error,'filename',None)})
                save(out/'data-audit.json',audit)
                raise
            if label != 'test' and any(isinstance(r['answer'],list) for r in read(a)):
                raise ValueError('List answer unsupported by current training adapter; review source: '+a)
            audit[label].append(record)
    train_image_names={str(Path(train[0]['image'])/p) for r in read(train[0]['dataset']) for p in (r.get('images') or [])}
    # Same-path check only. Do not silently remove shared images or claim semantic isolation.
    audit['validation_shared_image_paths'] = {
        s['dataset']: sum(bool(train_image_names.intersection(str(Path(s['image'])/p) for p in (r.get('images') or []))) for r in read(s['dataset'])) for s in val}
    audit['overlap_scope'] = 'Exact resolved image paths only; does not rule out byte-identical copies or semantic overlap. Review source provenance before approval.'
    audit['proposed_steps_raw'] = audit['train'][0]['rows']//384*2
    audit['image_decode_verified'] = args.verify_images
    audit['scope'] = getattr(args,'scope','all')
    audit['final_test_image_audit_complete'] = getattr(args,'scope','all')=='all'
    for name,value in [('train',train),('val',val),('test',test),('data-audit',audit)]:
        save(out/(name+'.json'),value)
    print(json.dumps(audit,ensure_ascii=True,indent=2))
    print('MANIFESTS_PREPARED: supervisor must review splits, overlap scope and counts')


def check_length(args):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer, AutoProcessor
    from verl.trainer.ppo.utils import create_rl_dataset
    cfg=OmegaConf.load(args.run/'resolved-config.yaml')
    model=cfg.actor_rollout_ref.model.path
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    proc=AutoProcessor.from_pretrained(model,local_files_only=True)
    ds=create_rl_dataset(cfg.data.train_files,cfg.data,tok,proc,max_samples=-1)
    val_ds=create_rl_dataset(cfg.data.val_files,cfg.data,tok,proc,max_samples=-1)
    steps=(len(ds)//cfg.data.train_batch_size)*cfg.trainer.total_epochs
    save(args.run/'length-audit.json',{'filtered_train_rows':len(ds),'filtered_validation_rows':len(val_ds),
                                     'batch':cfg.data.train_batch_size,'epochs':cfg.trainer.total_epochs,'computed_steps':steps,
                                     'prompt_filter_policy':'Original filter_overlong_prompts with the selected model/processor and 4096 prompt cap'})
    if len(val_ds)<1:
        raise ValueError('Validation dataset is empty after filtering')
    if steps != 202:
        raise ValueError('Filtered loader gives {} steps, not 202. Review budget; do not force 202.'.format(steps))
    print('FILTERED_LENGTH_OK: 202 steps')


def check_board(args):
    import swanlab
    swanlab.init(project='bao_board_preflight',experiment_name='SYNTHETIC-NOT-TRAINING',mode='local',logdir=str(args.run/'board-smoke'))
    swanlab.log({'train/accuracy':0.5,'train/reward':0.45},step=1)
    # The installed local dashboard registers text, but skips ECharts tables.
    # Keep this explicitly synthetic and do not imply real generations passed.
    swanlab.log({'val/generations':swanlab.Text(
        'SYNTHETIC CHECK ONLY\nstep=1\ninput=SYNTHETIC INPUT\n'
        'output=SYNTHETIC OUTPUT\nscore=0.45')},step=1)
    swanlab.finish()
    print('BOARD_WRITE_OK: open native watch and verify scalars and synthetic text manually')


def check_training(args):
    checkpoint=args.run/'checkpoints'/('global_step_'+str(args.step))/'actor'
    for rank in (0,1):
        for kind in ('model','optim'):
            p=checkpoint/('{}_world_size_2_rank_{}.pt'.format(kind,rank))
            if not p.is_file() or p.stat().st_size==0:
                raise FileNotFoundError(p)
    for name in ('generations','validation'):
        p=args.run/name/(str(args.step)+'.jsonl')
        if not p.is_file() or p.stat().st_size==0:
            raise FileNotFoundError(p)
        rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows: raise ValueError('No records: '+str(p))
        for row in rows:
            for key in ('input','output','gts','score','accuracy','format'):
                if key not in row: raise ValueError('Missing {} in {}'.format(key,p))
            if name=='validation':
                for key in ('sample_id','data_source','output_tokens'):
                    if key not in row: raise ValueError('Missing validation '+key)
        if name=='generations':
            from formal_metrics import component_metrics
            print(json.dumps(component_metrics({k:[r[k] for r in rows] for k in ('accuracy','format','score')})))
    print('CHECKPOINT_AND_GENERATIONS_PRESENT: verify dashboard metrics and sample contents too')


def summarize(args):
    before=read(args.run/'before/summary.json'); after=read(args.run/'after/summary.json')
    if before['manifest_sha256'] != after['manifest_sha256'] or before['n'] != after['n']:
        raise ValueError('Different evaluation settings')
    if len(before['datasets'])!=10 or set(before['datasets'])!=set(after['datasets']):
        raise ValueError('Incomplete dataset coverage')
    lines=['# 完整数据集评测结果','', 'Avg@{}，同一冻结manifest；差值为百分点。'.format(before['n']),'',
           '| 数据集 | N | Before (%) | After (%) | 变化 |','|---|---:|---:|---:|---:|']
    deltas=[]
    for name,b in before['datasets'].items():
        a=after['datasets'][name]
        if a['count']!=b['count'] or a['source_sha256']!=b['source_sha256']:
            raise ValueError('Source/count differs: '+name)
        delta=100*(a['accuracy']-b['accuracy']); deltas.append(delta)
        lines.append('| {} | {} | {:.4f} | {:.4f} | {:+.4f} |'.format(name,b['count'],100*b['accuracy'],100*a['accuracy'],delta))
    bmean=sum(v['accuracy'] for v in before['datasets'].values())/10
    amean=sum(v['accuracy'] for v in after['datasets'].values())/10
    lines.append('| 十集宏平均 | — | {:.4f} | {:.4f} | {:+.4f} |'.format(100*bmean,100*amean,100*(amean-bmean)))
    (args.run/'results-table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FULL_TEN_DATASET_TABLE_COMPLETE:', args.run/'results-table.md')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command')
    prep=sub.add_parser('prepare'); prep.add_argument('--output',type=Path,required=True); prep.add_argument('--data-root',type=Path,default=DATA_ROOT); prep.add_argument('--verify-images',action='store_true'); prep.add_argument('--scope',choices=('all','training'),default='all')
    prep.add_argument('--mathverse-images', type=Path)
    for name in ('check-length','check-board','check-training','summarize'):
        q=sub.add_parser(name); q.add_argument('--run',type=Path,required=True)
        if name=='check-training': q.add_argument('--step',type=int,required=True)
    a=p.parse_args()
    commands={'prepare':prepare,'check-length':check_length,'check-board':check_board,'check-training':check_training,'summarize':summarize}
    if a.command not in commands: p.error('Choose a subcommand')
    commands[a.command](a)


if __name__=='__main__': main()
