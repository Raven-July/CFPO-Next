"""Pure helpers for the fixed32-sample, three-arm inference diagnostic. No model imports."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

ARMS=('A','B','C')
PREFIX='<think>\n'
EXAMPLE='\n\nUse this output structure, replacing the placeholders with your own reasoning and final answer:\n<think>\n[reasoning]\n</think>\n\\boxed{[answer]}'
FIXED=dict(temperature=1.0,top_p=.99,top_k=-1,n=5,max_tokens=2048,max_model_len=6144,
           min_pixels=200704,max_pixels=1003520,seed=42)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_package(pack):
    pack=Path(pack)
    for name,digest in json.loads((pack/'files-sha256.json').read_text(encoding='utf-8')).items():
        if sha(pack/name)!=digest:raise ValueError('Package changed: '+name)

def verify_prepared(candidate):
    candidate=Path(candidate);r=json.loads((candidate/'receipt.json').read_text(encoding='utf-8'))
    if r['fixed']!=FIXED or r['total_outputs']!=480:raise ValueError('Probe budget/settings changed')
    if Path(r['model']).name!='Qwen3-VL-2B-Instruct':raise ValueError('Unexpected base model')
    for path,digest in r['frozen_files'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    samples=json.loads((candidate/'samples.json').read_text(encoding='utf-8'))
    if len(samples)!=32 or len({(s['data_source'],str(s['sample_id'])) for s in samples})!=32:raise ValueError('Not32 unique samples')
    for row in samples:
        if len(row['images'])!=len(row['image_sha256']):raise ValueError('Incomplete image hash list')
        for path,digest in zip(row['images'],row['image_sha256']):
            if sha(path)!=digest:raise ValueError('Sample image changed: '+path)
    return r

def arm_system(system,arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    return system+EXAMPLE if arm=='C' else system

def prompt_ids_for_arm(base_ids,arm,prefix_ids):
    if arm not in ARMS:raise ValueError('Unknown arm')
    return list(base_ids)+list(prefix_ids) if arm=='B' else list(base_ids)

def engine_prompt_ids(expanded_ids,image_id,vision_start_id,vision_end_id,image_counts):
    """HF expands image pads; vLLM token input needs one placeholder per image.

    Collapse only verified image runs, preserving every other token exactly.
    Counts must agree with the image grid, not merely with the number of images.
    """
    result=[];counts=[];ids=list(expanded_ids);i=0
    while i<len(ids):
        if ids[i]!=image_id:
            result.append(ids[i]);i+=1;continue
        start=i
        while i<len(ids) and ids[i]==image_id:i+=1
        if start==0 or ids[start-1]!=vision_start_id or i==len(ids) or ids[i]!=vision_end_id:
            raise ValueError('Image token outside a real vision block')
        counts.append(i-start);result.append(image_id)
    if counts!=list(image_counts):raise ValueError('Image pads do not match processed image grid: '+str(counts)+' vs '+str(image_counts))
    return result

def summarize(rows):
    groups=defaultdict(list);keys=set()
    for row in rows:
        if row['arm'] not in ARMS or int(row['generation_index']) not in range(5):raise ValueError('Invalid arm/generation index')
        key=(row['arm'],row['data_source'],str(row['sample_id']),int(row['generation_index']))
        if key in keys:raise ValueError('Duplicate output '+str(key))
        keys.add(key);groups[row['arm']].append(row)
        if row['finish_reason'] not in ('stop','length'):raise ValueError('Aborted output')
        if not 0<=int(row['output_tokens'])<=2048:raise ValueError('Wrong response budget')
    if set(groups)!=set(ARMS):raise ValueError('Missing arm')
    ids=None;result={}
    for arm in ARMS:
        data=groups[arm]
        by_id=Counter((r['data_source'],str(r['sample_id'])) for r in data)
        if len(data)!=160 or len(by_id)!=32 or set(by_id.values())!={5}:raise ValueError('Not32 times5 per arm')
        if ids is None:ids=set(by_id)
        elif set(by_id)!=ids:raise ValueError('Arms use different samples')
        c=Counter()
        for r in data:
            c['N']+=1;c['correct']+=float(r['accuracy']);c['format_positive']+=float(r['format'])>0
            c['length_finish']+=r['finish_reason']=='length';c['at_2048']+=r['output_tokens']==2048
            c['boxed']+='\\boxed{' in r['output'];c['closed_think']+='</think>' in r['output']
            c['empty']+=not r['output'].strip();c['tokens']+=r['output_tokens']
            c['copied_placeholder']+='[reasoning]' in r['output'] or '[answer]' in r['output']
            lines=Counter(s.strip() for s in r['output'].splitlines() if len(s.strip())>=20)
            c['repeated_line_flag']+=bool(lines and max(lines.values())>=3)
        result[arm]=dict(c,accuracy=c['correct']/160,format_rate=c['format_positive']/160,
                         length_rate=c['length_finish']/160,mean_tokens=c['tokens']/160)
    return dict(total_outputs=len(rows),arms=result,scope='Selected diagnostic32 samples; not a benchmark estimate or training result')
