"""One-GPU inference only, fixed original model,480 completions; run under container3.12."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace
from core import ARMS,FIXED,PREFIX,arm_system,prompt_ids_for_arm,engine_prompt_ids,sha,summarize,verify_package,verify_prepared

def main():
    if sys.version_info<(3,12):raise RuntimeError('Run probe only in container Python3.12+')
    pack=Path(__file__).resolve().parent;candidate=Path(os.environ['CANDIDATE']);run=candidate/'run'
    verify_package(pack);receipt=verify_prepared(candidate)
    # Heavy imports deliberately occur after the host/runtime boundary check.
    import asyncio
    import importlib.metadata
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig,AutoProcessor,AutoTokenizer
    from vllm import LLM,SamplingParams
    from verl.utils.dataset.cfpo import normalize_row
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
    import verl
    assert torch.cuda.device_count()==1,'Exactly one allocated GPU required'
    assert Path(verl.__file__).resolve().is_relative_to(Path(receipt['source']).resolve())
    assert importlib.metadata.version('vllm')=='0.18.0','Review changed vLLM version before running'
    assert importlib.metadata.version('mathruler')=='0.1.0','Review changed math grader before running'
    model=receipt['model'];processor=AutoProcessor.from_pretrained(model,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    model_config=AutoConfig.from_pretrained(model,local_files_only=True)
    assert model_config.model_type=='qwen3_vl'
    image_id=int(model_config.image_token_id)
    vision_start_id=tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id=tokenizer.convert_tokens_to_ids('<|vision_end|>')
    assert tokenizer.encode('<|image_pad|>',add_special_tokens=False)==[image_id]
    merge_size=int(model_config.vision_config.spatial_merge_size)
    system_path=Path(receipt['source'])/'examples/cfpo_trainer/format_prompt/reference_base.jinja'
    system=system_path.read_text(encoding='utf-8').strip()
    reward_path=Path(receipt['source'])/'examples/cfpo_trainer/reward_function/base.py'
    assert sha(reward_path)==receipt['reward_sha256']
    spec=importlib.util.spec_from_file_location('actual_format_probe_reward',reward_path)
    reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)
    for text,expected in [(r'x</think>\boxed{42}',1.),(r'\boxed{42}',.9),
                          (r'x</think>\boxed{41}',.1),(r'\boxed{41}',0.)]:
        assert abs(reward.compute_score('preflight',text,42)['score']-expected)<1e-8
    prefix_ids=tokenizer.encode(PREFIX,add_special_tokens=False)
    assert tokenizer.decode(prefix_ids,skip_special_tokens=True)==PREFIX
    builder=create_continuous_token_builder(tokenizer,hf_model_type='qwen3_vl',processor=processor,
        chat_template_kwargs={},mm_processor_kwargs=dict(min_pixels=200704,max_pixels=1003520))
    samples=json.loads((candidate/'samples.json').read_text(encoding='utf-8'))
    resources=dict(line.split('=',1) for line in (run/'slurm-resources.txt').read_text().splitlines())
    assert int(resources['cpus'])==8 and int(resources['memory_per_node_MiB'])==131072,resources
    adapter=SimpleNamespace(processor=processor,image_key='images',video_key='videos',audio_key='audios',format_prompt=system)
    prepared=[];rendering=[];retained_images=[]
    for i,sample in enumerate(samples):
        mapped=normalize_row(dict(problem=sample['problem'],answer=sample['answer'],images=sample['images'],id=sample['sample_id']),
                             i,image_root=None,source=sample['data_source'])
        adapter.format_prompt=system
        messages=RLHFDataset._build_messages(adapter,mapped,key='prompt')
        images,videos,audios=asyncio.run(RLHFDataset.process_multi_modal_info(messages,
            image_patch_size=processor.image_processor.patch_size,config=OmegaConf.create({})))
        if videos or audios:raise ValueError('This probe is limited to the selected image samples')
        images=images or [];retained_images.extend(images)
        base_features={};a_ids=builder.build_initial_tokens(messages,images=images,mm_inputs_out=base_features)
        actual_input=tokenizer.decode(a_ids,skip_special_tokens=True)
        assert hashlib.sha256(actual_input.encode()).hexdigest()==sample['expected_base_input_sha256'],'Base prompt differs from30-step snapshot'
        grid=base_features.get('image_grid_thw');grid=grid.tolist() if grid is not None else []
        assert all(200704<=g[1]*g[2]*processor.image_processor.patch_size**2<=1003520 for g in grid)
        adapter.format_prompt=arm_system(system,'C');c_messages=RLHFDataset._build_messages(adapter,mapped,key='prompt')
        c_features={};c_ids=builder.build_initial_tokens(c_messages,images=images,mm_inputs_out=c_features)
        c_grid=c_features.get('image_grid_thw');c_grid=c_grid.tolist() if c_grid is not None else []
        assert grid==c_grid,'Image preprocessing differs between prompt arms'
        ids={'A':a_ids,'B':prompt_ids_for_arm(a_ids,'B',prefix_ids),'C':c_ids}
        assert ids['B'][:len(a_ids)]==a_ids
        assert tokenizer.decode(ids['B'],skip_special_tokens=True).endswith(PREFIX)
        image_counts=[g[0]*g[1]*g[2]//(merge_size**2) for g in grid]
        engine_ids={arm:engine_prompt_ids(ids[arm],image_id,vision_start_id,vision_end_id,image_counts) for arm in ARMS}
        for arm in ARMS:
            if len(ids[arm])>4096:raise ValueError('Prompt exceeds4096; no silent truncation')
            assert len(ids[arm])+2048<=6144
            rendering.append(dict(sample_id=sample['sample_id'],data_source=sample['data_source'],arm=arm,
                prompt_tokens=len(ids[arm]),prompt_tail=tokenizer.decode(ids[arm][-50:],skip_special_tokens=False),
                prompt_token_ids=ids[arm],engine_request_token_ids=engine_ids[arm],grid=grid,
                image_token_counts=image_counts,prefix_belongs_to_input=arm=='B'))
        prepared.append(dict(sample=sample,images=images,ids=ids,engine_ids=engine_ids))
    (run/'prompt-audit.json').write_text(json.dumps(rendering,ensure_ascii=False,indent=2),encoding='utf-8')
    runtime=dict(model=model,reward_import=str(reward_path),reward_sha256=sha(reward_path),
        vllm=importlib.metadata.version('vllm'),transformers=importlib.metadata.version('transformers'),
        torch=importlib.metadata.version('torch'),mathruler=importlib.metadata.version('mathruler'),gpu=torch.cuda.get_device_name(0),fixed=FIXED,
        source=str(verl.__file__),system_sha256=sha(system_path),samples=32,planned_outputs=480,
        note='Standalone inference uses the same frozen dataset message/vision and Continuous Token builders. No optimizer, update or checkpoint restoration.')
    (run/'runtime-audit.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    print('ALL32_PROMPTS_IMAGES_REWARD_AUDITED',json.dumps(runtime),flush=True)
    llm=LLM(model=model,tokenizer=model,tensor_parallel_size=1,dtype='bfloat16',
        gpu_memory_utilization=.5,max_model_len=6144,max_num_seqs=8,max_num_batched_tokens=6144,
        enforce_eager=True,enable_chunked_prefill=True,enable_prefix_caching=False,seed=42,
        limit_mm_per_prompt={'image':max(1,max(len(v['images']) for v in prepared))},
        mm_processor_kwargs=dict(min_pixels=200704,max_pixels=1003520))
    def timeout_handler(signum,frame):raise TimeoutError('Math grading exceeded10 seconds')
    old_handler=signal.signal(signal.SIGALRM,timeout_handler)
    rows=[];started=time.time()
    try:
        with (run/'outputs.jsonl').open('x',encoding='utf-8') as stream:
            for i,item in enumerate(prepared):
                sample=item['sample'];order=list(ARMS);random.Random(42+i).shuffle(order)
                for arm in order:
                    request={'prompt_token_ids':item['engine_ids'][arm]}
                    if item['images']:request['multi_modal_data']={'image':item['images']}
                    params=SamplingParams(n=5,temperature=1.,top_p=.99,top_k=-1,max_tokens=2048,
                        repetition_penalty=1.,ignore_eos=False,seed=42+i,skip_special_tokens=True)
                    output=llm.generate([request],params,use_tqdm=False)
                    assert len(output)==1 and len(output[0].outputs)==5,'Missing completion'
                    actual_ids=list(output[0].prompt_token_ids)
                    prompt_check=dict(sample_id=sample['sample_id'],data_source=sample['data_source'],arm=arm,
                        request_token_ids=item['engine_ids'][arm],expected_expanded_token_ids=item['ids'][arm],
                        actual_expanded_token_ids=actual_ids,exact_match=actual_ids==list(item['ids'][arm]))
                    with (run/'engine-prompt-audit.jsonl').open('a',encoding='utf-8') as audit:
                        audit.write(json.dumps(prompt_check)+'\n')
                    assert prompt_check['exact_match'],'Engine expansion differs from HF expected tokens; actual/expected saved in engine-prompt-audit.jsonl'
                    assert sorted(o.index for o in output[0].outputs)==list(range(5))
                    for completion in output[0].outputs:
                        if completion.finish_reason not in ('stop','length'):raise ValueError('Unexpected finish reason')
                        tokens=list(completion.token_ids);text=tokenizer.decode(tokens,skip_special_tokens=True)
                        signal.alarm(10)
                        try:score=reward.compute_score(sample['data_source'],text,sample['answer'])
                        finally:signal.alarm(0)
                        row=dict(arm=arm,sample_id=sample['sample_id'],data_source=sample['data_source'],
                            generation_index=completion.index,diagnostic_stratum=sample['diagnostic_stratum'],
                            output=text,ground_truth=sample['answer'],output_tokens=len(tokens),output_token_ids=tokens,
                            finish_reason=completion.finish_reason,stop_reason=completion.stop_reason,
                            prefix_in_input=arm=='B',seed=42+i,**score)
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n');rows.append(row)
                    stream.flush();print('PROBE_PROGRESS',i+1,'/32','arm='+arm,'outputs='+str(len(rows)),flush=True)
    finally:
        signal.signal(signal.SIGALRM,old_handler)
        for image in retained_images:image.close()
    summary=summarize(rows);summary['elapsed_seconds']=time.time()-started
    (run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FORMAT_PROBE480_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
