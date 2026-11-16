"""CPU checks of metrics, dataset coverage/counts and immutable preparation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

BASE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

tools=load('formal_tools'); metrics=load('formal_metrics')


class FormalTests(unittest.TestCase):
    def test_private_mathverse_view_preserves_annotations_and_counts(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            root=Path(tmp); self.prepare_fixture(root)
            original=root/'V3-math/MathVerse_val_V3.json'
            before=original.read_bytes()
            view=root/'private/MathVerse_images'; view.mkdir(parents=True)
            tools.prepare(SimpleNamespace(data_root=root,output=root/'out',verify_images=False,
                                          scope='training',mathverse_images=view))
            val=tools.read(root/'out/val.json')
            self.assertEqual(val[2]['image'],str(view))
            self.assertEqual(original.read_bytes(),before)
            self.assertEqual(tools.read(root/'out/data-audit.json')['validation'][2]['rows'],1)
            test=tools.read(root/'out/test.json')
            mv=next(row for row in test if row['dataset_name']=='MathVerse_fixed')
            self.assertEqual(mv['image_root'],str(view))
            self.assertTrue(mv['json_path'].endswith('MathVerse-closed_test.json'))

    def prepare_fixture(self,root):
        reference=tools.read(BASE.parents[1]/'Counterfactual-Eval/datasets_config_math.json')
        paths=['V3-math/ViRL39K_train.json','V3-math/geometry3k_val_V3.json','V3-math/LogicVista_val_V3.json',
               'V3-math/MathVerse_val_V3.json','V3/C-VQA-Real_val_V3.json','V3/MARS_Bench_val_V3.json']
        paths.extend(item['json_path'].split('/Datasets/',1)[1] for item in reference)
        for name in paths:
            p=root/name; p.parent.mkdir(parents=True,exist_ok=True)
            tools.save(p,[{'problem':'q','answer':'1','images':[]}])

    def test_training_scope_preserves_all_tests_as_deferred(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            root=Path(tmp); self.prepare_fixture(root)
            original=tools.rows_audit
            def fail_test(path,*args):
                if str(path).endswith('geometry3k_test.json'): raise PermissionError('synthetic test-only denial')
                return original(path,*args)
            with patch.object(tools,'rows_audit',side_effect=fail_test):
                tools.prepare(SimpleNamespace(data_root=root,output=root/'out',verify_images=False,scope='training'))
            self.assertEqual(len(tools.read(root/'out/test.json')),10)
            audit=tools.read(root/'out/data-audit.json')
            self.assertFalse(audit['final_test_image_audit_complete'])
            self.assertTrue(all(row['status']=='deferred_to_final_evaluation' for row in audit['test']))

    def test_validation_permission_failure_never_skipped(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            root=Path(tmp); self.prepare_fixture(root)
            original=tools.rows_audit
            def fail_val(path,*args):
                if str(path).endswith('MathVerse_val_V3.json'): raise PermissionError(13,'synthetic val denial','image_745.png')
                return original(path,*args)
            with patch.object(tools,'rows_audit',side_effect=fail_val), self.assertRaises(PermissionError):
                tools.prepare(SimpleNamespace(data_root=root,output=root/'out',verify_images=False,scope='training'))
            self.assertFalse((root/'out/train.json').exists())
            audit=tools.read(root/'out/data-audit.json')
            self.assertEqual(audit['validation'][-1]['error_type'],'PermissionError')

    def test_all_scope_test_failure_is_fatal(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            root=Path(tmp); self.prepare_fixture(root)
            original=tools.rows_audit
            def fail_test(path,*args):
                if str(path).endswith('geometry3k_test.json'): raise PermissionError('synthetic test denial')
                return original(path,*args)
            with patch.object(tools,'rows_audit',side_effect=fail_test), self.assertRaises(PermissionError):
                tools.prepare(SimpleNamespace(data_root=root,output=root/'out',verify_images=False,scope='all'))

    def test_components(self):
        got=metrics.component_metrics({'accuracy':[0,1],'format':[0,0],'score':[0,.9]})
        self.assertEqual(got['train/accuracy'],.5)
        self.assertEqual(got['train/reward'],.45)
        self.assertEqual(got['train/score_count'],2)

    def test_missing_invalid_or_mismatched(self):
        for values in ({'accuracy':[1]}, {'accuracy':[float('nan')],'format':[0],'score':[0]},
                       {'accuracy':[1,0],'format':[0],'score':[.9]}):
            with self.assertRaises(ValueError): metrics.component_metrics(values)

    def test_missing_image_or_placeholder(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            p=Path(tmp)/'rows.json'
            p.write_text(json.dumps([{'problem':'<image> question','images':['missing.png'],'answer':'A'}]))
            with self.assertRaises(FileNotFoundError): tools.rows_audit(p,tmp)
            p.write_text(json.dumps([{'problem':'question','images':['missing.png'],'answer':'A'}]))
            with self.assertRaises(ValueError): tools.rows_audit(p,tmp)

    def test_complete_summary_and_source_mismatch(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            root=Path(tmp)
            for label in ('before','after'):
                (root/label).mkdir()
                tools.save(root/label/'summary.json',{'manifest_sha256':'fixed','n':8,'datasets':{
                    'dataset-'+str(i):{'count':2,'source_sha256':'same','accuracy':(.5 if label=='before' else .75)} for i in range(10)}})
            tools.summarize(SimpleNamespace(run=root))
            self.assertIn('25.0000',(root/'results-table.md').read_text(encoding='utf-8'))
            bad=tools.read(root/'after/summary.json'); bad['datasets']['dataset-0']['source_sha256']='changed'
            tools.save(root/'after/summary.json',bad)
            with self.assertRaises(ValueError): tools.summarize(SimpleNamespace(run=root))


if __name__=='__main__': unittest.main()
