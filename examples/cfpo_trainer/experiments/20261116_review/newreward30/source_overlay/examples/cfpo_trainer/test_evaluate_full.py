"""Synthetic end-to-end wrapper checks; no model is loaded."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('full_eval',BASE/'evaluate_full.py')
full=importlib.util.module_from_spec(spec); spec.loader.exec_module(full)


class WrapperTests(unittest.TestCase):
    def run_wrapper(self, missing_output=False, bad_template=False):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            root=Path(tmp); manifest=[]
            for i in range(10):
                p=root/('dataset'+str(i)+'.json')
                p.write_text(json.dumps([{'problem':'question','images':[],'answer':'1','id':'sample-'+str(i)}]))
                manifest.append({'dataset_name':'dataset'+str(i),'json_path':str(p),'image_root':str(root)})
            config=root/'manifest.json'; config.write_text(json.dumps(manifest))
            class Processor:
                @classmethod
                def from_pretrained(cls,*a,**kw): return cls()
                def apply_chat_template(self,*a,**kw):
                    if bad_template: raise ValueError('synthetic template failure')
                    return 'prompt'
            class LLM:
                def __init__(self,**kw): pass
                def generate(self,prompts,sampling_params):
                    for prompt in prompts:
                        self.assert_keys(prompt)
                    if missing_output: return []
                    return [SimpleNamespace(outputs=[SimpleNamespace(text='1',token_ids=[1],finish_reason='stop') for _ in range(sampling_params.n)]) for p in prompts]
                @staticmethod
                def assert_keys(prompt):
                    if set(prompt)!={'prompt','multi_modal_data'}: raise AssertionError(prompt)
            trans=ModuleType('transformers'); trans.AutoProcessor=Processor
            def sampling_params(**kw):
                self.assertEqual(kw['temperature'],1.0)
                self.assertEqual(kw['n'],8)
                return SimpleNamespace(**kw)
            vllm=ModuleType('vllm'); vllm.LLM=LLM; vllm.SamplingParams=sampling_params
            grader=ModuleType('mathruler.grader'); grader.grade_answer=lambda a,b:a==b; grader.extract_boxed_content=lambda s:s
            mathruler=ModuleType('mathruler'); mathruler.grader=grader
            progress=ModuleType('tqdm'); progress.tqdm=lambda x,**kw:x
            args=['evaluate_full.py','--manifest',str(config),'--model',str(root),'--output',str(root/'result'),'--n','8']
            with patch.dict(sys.modules,{'transformers':trans,'vllm':vllm,'mathruler':mathruler,'mathruler.grader':grader,'tqdm':progress}), patch.object(sys,'argv',args), contextlib.redirect_stdout(io.StringIO()):
                full.main()
            summary=json.loads((root/'result/summary.json').read_text())
            self.assertEqual(len(summary['datasets']),10)
            self.assertEqual(summary['temperature'],1.0)
            self.assertEqual(summary['metric'],'Avg@8')
            self.assertTrue(all(v['count']==1 and v['accuracy']==1 for v in summary['datasets'].values()))

    def test_complete_ten_dataset_run(self): self.run_wrapper()
    def test_missing_model_response_rejected(self):
        with self.assertRaises(ValueError): self.run_wrapper(missing_output=True)
    def test_template_failure_not_silently_skipped(self):
        with self.assertRaises(ValueError): self.run_wrapper(bad_template=True)


if __name__=='__main__': unittest.main()
