"""Frozen 12-completion diagnostic; host-compatible helpers, no ML imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path

REWARD_SHA = '9f07ac4d431cb2fc7cca671781244c239d068b4433132bd9a3baba742516ea61'
SIMPLE_SYSTEM = "Answer the user's question briefly. Put the final answer in \\boxed{}."
SAMPLE_IDS = ['C-VQA-Real-10', 'C-VQA-Real-1170', 'C-VQA-Real-1355', 'MARS_Bench-0029']
FIXED = dict(temperature=0.0, top_p=1.0, top_k=-1, n=1, max_tokens=2048,
             repetition_penalty=1.0, ignore_eos=False, max_model_len=6144,
             max_num_batched_tokens=6144, min_pixels=200704, max_pixels=1003520, seed=42)
CELLS = [('hf', 'original'), ('vllm', 'original'), ('vllm', 'simple')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_package(pack):
    pack = Path(pack)
    for name, digest in json.loads((pack/'files-sha256.json').read_text(encoding='utf-8')).items():
        if sha(pack/name) != digest:
            raise ValueError('Package changed: '+name)


def verify_prepared(candidate):
    candidate = Path(candidate)
    r = json.loads((candidate/'receipt.json').read_text(encoding='utf-8'))
    if r['fixed'] != FIXED or r['total_outputs'] != 12 or r['reward_sha256'] != REWARD_SHA:
        raise ValueError('Changed diagnostic contract')
    if Path(r['model']).name != 'Qwen3-VL-2B-Instruct':
        raise ValueError('Wrong base model; no training checkpoint is permitted')
    for name, metadata in r['model_weight_metadata'].items():
        current = (Path(r['model'])/name).stat()
        if current.st_size != metadata['size'] or current.st_mtime_ns != metadata['mtime_ns']:
            raise ValueError('Original weight metadata changed: '+name)
    for name, digest in r['frozen_files'].items():
        if sha(name) != digest:
            raise ValueError('Frozen input changed: '+name)
    samples = json.loads((candidate/'samples.json').read_text(encoding='utf-8'))
    if [s['sample_id'] for s in samples] != SAMPLE_IDS:
        raise ValueError('Wrong sample identities/order')
    for sample in samples:
        if len(sample['images']) != len(sample['image_sha256']):
            raise ValueError('Image hash count mismatch')
        for name, digest in zip(sample['images'], sample['image_sha256']):
            if sha(name) != digest:
                raise ValueError('Changed source image: '+name)
    return r


def summarize(rows):
    expected = set((backend, arm, sid) for backend, arm in CELLS for sid in SAMPLE_IDS)
    actual = [(r['backend'], r['arm'], r['sample_id']) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Missing, duplicate or extra diagnostic outputs')
    result = {}
    for backend, arm in CELLS:
        selected = [r for r in rows if (r['backend'], r['arm']) == (backend, arm)]
        for row in selected:
            if row['finish_reason'] not in ['stop', 'length'] or not 0 < row['output_tokens'] <= 2048:
                raise ValueError('Invalid generated length/finish')
            if row['output_tokens'] != len(row['output_token_ids']):
                raise ValueError('Token count differs')
        result[backend+'/'+arm] = dict(N=4, correct=sum(r['accuracy'] for r in selected),
            format_positive=sum(r['format'] for r in selected),
            at_limit=sum(r['output_tokens'] == 2048 for r in selected),
            normal_stop=sum(r['finish_reason'] == 'stop' for r in selected))
    return dict(cells=result, total_outputs=12, ppo_clip='not_applicable_no_updates',
                scope='Four selected images; causal localization probe, not a performance benchmark')
