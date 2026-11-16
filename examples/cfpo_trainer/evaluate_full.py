"""Strict wrapper around preserved Counterfactual-Eval; no silent data loss."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,choices=(8,),default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()): raise FileExistsError('Use an empty evaluation output directory')
    from formal_tools import rows_audit
    datasets=json.loads(args.manifest.read_text(encoding='utf-8'))
    if len(datasets)!=10 or len({d['dataset_name'] for d in datasets})!=10:
        raise ValueError('Expected ten distinct datasets')
    audits={d['dataset_name']:rows_audit(d['json_path'],d['image_root'],True) for d in datasets}
    source=Path(__file__).resolve().parents[2]/'Counterfactual-Eval/inference_eval.py'
    spec=importlib.util.spec_from_file_location('reference_evaluator',source)
    ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    original_preprocess=ref.load_and_preprocess_single_item

    def strict_preprocess(item,image_root,processor,prompt_template,params):
        # Validate decode again at use time, before reference fallback code can run.
        for name in item['images']:
            with ref.Image.open(Path(image_root)/name) as image:
                image.load()
        result=original_preprocess(item,image_root,processor,prompt_template,params)
        if result is None: raise ValueError('Chat template failed; refusing skipped sample')
        return result
    ref.load_and_preprocess_single_item=strict_preprocess
    # Pass only supported vLLM request keys; collect finish metadata separately.
    class AuditedLLM:
        def __init__(self,inner): self.inner=inner; self.finishes=[]
        def generate(self,prompts,sampling_params):
            requests=[{'prompt':p['prompt'],'multi_modal_data':p['multi_modal_data']} for p in prompts]
            outputs=self.inner.generate(prompts=requests,sampling_params=sampling_params)
            if len(outputs)!=len(prompts): raise ValueError('Missing model responses')
            for output in outputs:
                if len(output.outputs)!=args.n: raise ValueError('Incomplete n responses')
                meta=[]
                for r in output.outputs:
                    if r.finish_reason not in ('stop','length'): raise ValueError('Aborted response')
                    meta.append({'finish_reason':r.finish_reason,'output_tokens':len(r.token_ids)})
                self.finishes.append(meta)
            return outputs
    processor=ref.AutoProcessor.from_pretrained(str(args.model),local_files_only=True)
    inner=ref.LLM(model=str(args.model),tensor_parallel_size=2,dtype='bfloat16',
                  gpu_memory_utilization=0.6,max_model_len=8192,max_num_seqs=32,
                  max_num_batched_tokens=8192,enable_chunked_prefill=True,enforce_eager=True,seed=42)
    llm=AuditedLLM(inner)
    summary={'manifest_sha256':sha(args.manifest),'n':args.n,'temperature':1.0,
             'metric':'Avg@8','reference_sha256':sha(source),'datasets':{}}
    for ds in datasets:
        name=ds['dataset_name']; llm.finishes=[]
        params=SimpleNamespace(n=args.n,cot=True,papo=False,batch_size=4,num_workers=4,
                               model_name='Qwen3-VL-2B-Instruct',model_prefix='formal',output_dir=str(args.output))
        ref.run_evaluation(params,llm,processor,ds)
        paths=list(args.output.glob('*_'+name+'_avg@'+str(args.n)+'_COT.json'))
        if len(paths)!=1: raise ValueError('Missing/ambiguous dataset result: '+name)
        result=json.loads(paths[0].read_text(encoding='utf-8')); details=result['details']
        raw=json.loads(Path(ds['json_path']).read_text(encoding='utf-8'))
        if len(details)!=len(raw) or len(llm.finishes)!=len(raw): raise ValueError('Dropped samples: '+name)
        if sha(ds['json_path'])!=audits[name]['sha256']: raise ValueError('Dataset changed')
        for i,(row,original) in enumerate(zip(details,raw)):
            if row['question']!=original['problem'] or row['ground_truth']!=original['answer'] or row['images']!=original['images']:
                raise ValueError('Result order/source mismatch')
            row['source_index']=i; row['id']=original.get('id'); row['generation_metadata']=llm.finishes[i]
        paths[0].write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        summary['datasets'][name]={'count':len(details),'accuracy':sum(r['avg_score'] for r in details)/len(details),
                                  'source_sha256':audits[name]['sha256'],'result':paths[0].name}
        (args.output/'progress.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('TEN_DATASET_EVALUATION_COMPLETE')


if __name__=='__main__': main()
