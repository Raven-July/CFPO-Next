"""Two sequential processes: HF/original (4), vLLM/original+simple (8). No training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from protocol import FIXED, SIMPLE_SYSTEM, verify_package, verify_prepared
from image_tokens import engine_prompt_ids


def tensor_summary(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(__import__('torch').uint8).numpy().tobytes()).hexdigest())


def timeout_handler(signum, frame):
    raise TimeoutError('Grader exceeded 10 seconds; preserve run and inspect')


def main():
    if sys.version_info < (3, 12):
        raise RuntimeError('Container Python3.12+ required; no ML imports on login node')
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['hf', 'vllm'], required=True)
    args = parser.parse_args()
    pack = Path(__file__).resolve().parent
    candidate = Path(os.environ['CANDIDATE'])
    run = candidate/'run'
    verify_package(pack)
    receipt = verify_prepared(candidate)
    if str(Path(receipt['source']).resolve()) not in [str(Path(p).resolve()) for p in sys.path if p]:
        raise ValueError('Frozen source must be on PYTHONPATH')
    import asyncio
    import importlib.metadata
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig
    from verl.utils.dataset.cfpo import normalize_row
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
    import verl
    assert Path(verl.__file__).resolve().is_relative_to(Path(receipt['source']).resolve())
    assert torch.cuda.device_count() == 1, 'Exactly one Slurm GPU required'
    assert importlib.metadata.version('transformers') == '5.3.0.dev0'
    assert importlib.metadata.version('mathruler') == '0.1.0'
    assert importlib.metadata.version('vllm') == '0.18.0'
    resources = dict(line.split('=', 1) for line in (run/'slurm-resources.txt').read_text().splitlines())
    assert int(resources['cpus']) == 8 and int(resources['memory_per_node_MiB']) == 131072
    model_path = receipt['model']
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert config.model_type == 'qwen3_vl'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    generation = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    eos_ids = generation.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    assert tokenizer.eos_token_id in eos_ids and all(isinstance(i, int) for i in eos_ids)
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.max_new_tokens = 2048
    generation.num_beams = 1
    generation.num_return_sequences = 1
    generation.repetition_penalty = 1.0
    generation.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generation.return_dict_in_generate = True
    generation.output_scores = False
    for key in ['stop_strings', 'forced_bos_token_id', 'forced_eos_token_id', 'bad_words_ids', 'force_words_ids']:
        assert not getattr(generation, key, None), 'Review extra generation constraint: '+key
    assert not getattr(generation, 'min_new_tokens', None)
    reward_path = Path(receipt['source'])/'examples/cfpo_trainer/reward_function/base.py'
    spec = importlib.util.spec_from_file_location('fixed_reward', reward_path)
    reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward)
    for text, truth, expected in [(r'x</think>\boxed{42}', 42, 1.), (r'\boxed{42}', 42, .9),
                                  (r'x</think>\boxed{41}', 42, .1), (r'\boxed{41}', 42, 0.)]:
        assert abs(reward.compute_score('preflight', text, truth)['score']-expected) < 1e-8
    system = (Path(receipt['source'])/'examples/cfpo_trainer/format_prompt/reference_base.jinja').read_text(encoding='utf-8').strip()
    builder = create_continuous_token_builder(tokenizer, hf_model_type='qwen3_vl', processor=processor,
        chat_template_kwargs={}, mm_processor_kwargs=dict(min_pixels=200704, max_pixels=1003520))
    samples = json.loads((candidate/'samples.json').read_text(encoding='utf-8'))
    prepared = []
    input_audit = []
    arms = ['original'] if args.backend == 'hf' else ['original', 'simple']
    for sample in samples:
        mapped = normalize_row(dict(problem=sample['problem'], answer=sample['answer'],
            images=sample['images'], id=sample['sample_id']), 0, image_root=None, source=sample['data_source'])
        for arm in arms:
            adapter = SimpleNamespace(processor=processor, image_key='images', video_key='videos', audio_key='audios',
                                      format_prompt=system if arm == 'original' else SIMPLE_SYSTEM)
            messages = RLHFDataset._build_messages(adapter, mapped, key='prompt')
            images, videos, audios = asyncio.run(RLHFDataset.process_multi_modal_info(messages,
                image_patch_size=processor.image_processor.patch_size, config=OmegaConf.create({})))
            assert images and not videos and not audios
            raw_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Direct standard HF processor call; compare it to the real CT builder.
            direct = processor(text=[raw_text], images=images, return_tensors='pt',
                               min_pixels=200704, max_pixels=1003520)
            features = {}
            ids = builder.build_initial_tokens(messages, images=images, mm_inputs_out=features)
            assert direct['input_ids'][0].tolist() == ids, 'Direct HF input differs from CT'
            for key in ['pixel_values', 'image_grid_thw']:
                assert torch.equal(direct[key], features[key]), 'Direct HF features differ from CT: '+key
            grid = direct['image_grid_thw'].tolist()
            patch = processor.image_processor.patch_size
            assert all(200704 <= t*h*w*patch**2 <= 1003520 for t, h, w in grid)
            assert len(ids) <= 4096 and len(ids)+2048 <= 6144
            if arm == 'original':
                decoded = tokenizer.decode(ids, skip_special_tokens=True)
                assert hashlib.sha256(decoded.encode()).hexdigest() == sample['expected_base_input_sha256']
            processed = []
            for i, image in enumerate(images):
                name = '{}-{}-{}-{}.png'.format(args.backend, arm, sample['sample_id'], i)
                image.save(run/name)
                processed.append(dict(path=name, size=list(image.size),
                    rgb_sha256=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()))
            input_audit.append(dict(backend=args.backend, arm=arm, sample_id=sample['sample_id'],
                messages=messages, raw_text=raw_text, input_ids=ids, processed_images=processed,
                pixel_values=tensor_summary(direct['pixel_values']), image_grid_thw=grid,
                note='HF/CT processor features; vLLM internal pixel tensor is not observed'))
            counts = [t*h*w//(config.vision_config.spatial_merge_size**2) for t, h, w in grid]
            request_ids = engine_prompt_ids(ids, config.image_token_id,
                tokenizer.convert_tokens_to_ids('<|vision_start|>'), tokenizer.convert_tokens_to_ids('<|vision_end|>'), counts)
            prepared.append((sample, arm, images, direct, ids, request_ids))
    audit_path = run/(args.backend+'-input-audit.json')
    audit_path.write_text(json.dumps(input_audit, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.backend == 'vllm':
        prior = json.loads((run/'hf-input-audit.json').read_text(encoding='utf-8'))
        originals = {r['sample_id']: r for r in prior}
        for record in input_audit:
            if record['arm'] == 'original':
                for key in ['input_ids', 'pixel_values', 'image_grid_thw']:
                    assert record[key] == originals[record['sample_id']][key], 'HF/vLLM request input differs: '+key
                assert [x['rgb_sha256'] for x in record['processed_images']] == [x['rgb_sha256'] for x in originals[record['sample_id']]['processed_images']]
    runtime = dict(backend=args.backend, model=model_path, source=str(verl.__file__),
        versions={key: importlib.metadata.version(key) for key in ['torch', 'transformers', 'vllm', 'mathruler']},
        fixed=FIXED, hf_generation_config=generation.to_dict(), eos_token_ids=eos_ids,
        no_optimizer_no_update_no_checkpoint_restore=True)
    if args.backend == 'hf':
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        runtime['model_class'] = type(model).__name__
    else:
        from vllm import LLM, SamplingParams
        model = LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=1, dtype='bfloat16',
            gpu_memory_utilization=.5, max_model_len=6144, max_num_batched_tokens=6144, max_num_seqs=8,
            enforce_eager=True, enable_chunked_prefill=True, enable_prefix_caching=False, seed=42,
            generation_config='vllm', limit_mm_per_prompt={'image':1},
            mm_processor_kwargs=dict(min_pixels=200704, max_pixels=1003520))
        params = SamplingParams(n=1, temperature=0., top_p=1., top_k=-1, max_tokens=2048,
            repetition_penalty=1., ignore_eos=False, seed=42,
            stop_token_ids=[i for i in eos_ids if i != tokenizer.eos_token_id], skip_special_tokens=True)
        runtime['actual_sampling_params'] = str(params)
    (run/(args.backend+'-runtime.json')).write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        with (run/(args.backend+'-outputs.jsonl')).open('x', encoding='utf-8') as stream:
            for sample, arm, images, direct, ids, request_ids in prepared:
                started = time.monotonic()
                if args.backend == 'hf':
                    with torch.inference_mode():
                        generated = model.generate(**direct.to('cuda'), generation_config=generation)
                    tokens = generated.sequences[0, len(ids):].tolist()
                    finish = 'stop' if tokens and tokens[-1] in eos_ids else 'length'
                    assert finish == 'stop' or len(tokens) == 2048
                    stop_reason = tokens[-1] if finish == 'stop' else None
                    direct.to('cpu')
                else:
                    generated = model.generate([dict(prompt_token_ids=request_ids,
                        multi_modal_data={'image':images})], params, use_tqdm=False)
                    actual_ids = list(generated[0].prompt_token_ids)
                    engine_audit = dict(sample_id=sample['sample_id'], arm=arm, request_token_ids=request_ids,
                        expected_expanded_token_ids=ids, actual_expanded_token_ids=actual_ids,
                        exact_match=actual_ids == ids)
                    with (run/'engine-prompt-audit.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(engine_audit)+'\n')
                    assert engine_audit['exact_match'], 'Actual vLLM expanded input differs from HF'
                    assert len(generated) == 1 and len(generated[0].outputs) == 1
                    completion = generated[0].outputs[0]
                    tokens = list(completion.token_ids)
                    finish, stop_reason = completion.finish_reason, completion.stop_reason
                elapsed = time.monotonic()-started
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                signal.alarm(10)
                try:
                    score = reward.compute_score(sample['data_source'], text, sample['answer'])
                finally:
                    signal.alarm(0)
                record = dict(backend=args.backend, arm=arm, sample_id=sample['sample_id'],
                    data_source=sample['data_source'], ground_truth=sample['answer'], output=text,
                    output_token_ids=tokens, output_tokens=len(tokens), finish_reason=finish,
                    stop_reason=stop_reason, generation_seconds=elapsed, **score)
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                stream.flush()
                print('DIAGNOSTIC_OUTPUT', args.backend, arm, sample['sample_id'],
                      len(tokens), finish, score, flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    main()
