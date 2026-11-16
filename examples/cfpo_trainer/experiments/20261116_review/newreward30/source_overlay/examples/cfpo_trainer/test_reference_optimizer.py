"""CPU parity with original CFPO optimizer; no model/GPU or original package imports."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import List, Tuple
import unittest
import torch

BASE=Path(__file__).resolve().parent
ORIGINAL=BASE.parents[2]/'CFPO/verl/utils/torch_functional.py'
# Load only the dtype helper; the local CPU test does not install Ray or import verl.__init__.
for name in ('verl','verl.utils'):
    sys.modules[name]=ModuleType(name)
    sys.modules[name].__path__=[]
spec=importlib.util.spec_from_file_location('verl.utils.torch_dtypes',BASE.parents[1]/'verl/utils/torch_dtypes.py')
dtype_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(dtype_module)
sys.modules[spec.name]=dtype_module
PrecisionType=dtype_module.PrecisionType
from reference_optimizer import AnyPrecisionAdamW


class OptimizerTests(unittest.TestCase):
    def original(self):
        if not ORIGINAL.exists(): self.skipTest('Original sibling repository not available; run provenance audit on Bao machine')
        tree=ast.parse(ORIGINAL.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AnyPrecisionAdamW')
        ns={'torch':torch,'List':List,'Tuple':Tuple,'PrecisionType':PrecisionType}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(ORIGINAL),'exec'),ns)
        return ns['AnyPrecisionAdamW'],cls

    def test_class_body_unchanged(self):
        _,old=self.original()
        new=next(n for n in ast.parse((BASE/'reference_optimizer.py').read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef))
        self.assertEqual(ast.dump(old),ast.dump(new))

    def test_step_and_state_parity(self):
        old,_=self.original()
        for dtype in (torch.float32,torch.bfloat16):
            a=torch.nn.Parameter(torch.tensor([1.,-2.,.5],dtype=dtype)); b=torch.nn.Parameter(a.detach().clone())
            x=old([a],lr=1e-3,weight_decay=.01); y=AnyPrecisionAdamW([b],lr=1e-3,weight_decay=.01)
            for grad in ([.1,-.3,.2],[-.2,.1,.01],[.02,.03,-.2]):
                a.grad=torch.tensor(grad,dtype=dtype); b.grad=a.grad.clone()
                x.step(); y.step()
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                for key in x.state[a]: torch.testing.assert_close(x.state[a][key],y.state[b][key],rtol=0,atol=0)
            self.assertEqual(y.state[b]['exp_avg'].dtype,torch.bfloat16)
            self.assertEqual(y.state[b]['compensation'].dtype,torch.bfloat16)

    def test_bf16_state_restore(self):
        a=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.bfloat16))
        x=AnyPrecisionAdamW([a],lr=1e-3); a.grad=torch.tensor([.1,.2],dtype=torch.bfloat16); x.step()
        b=torch.nn.Parameter(a.detach().clone()); y=AnyPrecisionAdamW([b],lr=1e-3); y.load_state_dict(x.state_dict())
        self.assertEqual(y.state[b]['exp_avg'].dtype,torch.bfloat16)
        self.assertEqual(y.state[b]['compensation'].dtype,torch.bfloat16)


if __name__=='__main__': unittest.main()
