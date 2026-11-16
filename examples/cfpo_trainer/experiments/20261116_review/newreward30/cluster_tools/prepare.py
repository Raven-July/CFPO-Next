"""Create a private startup snapshot plus audited patches. Python3.6+, no GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from restore_snapshot import restore_source
from guard30 import sha,verify_receipt

def patch_logging(source):
    p=source/'verl/utils/tracking.py';text=p.read_text(encoding='utf-8')
    start=text.index('    def log_generations_to_swanlab(')
    end=text.index('    def log_generations_to_mlflow(',start)
    replacement='''    def log_generations_to_swanlab(self, samples, step):
        """Local dashboard text compatibility; preserve complete display outputs."""
        import swanlab
        texts = [swanlab.Text("STEP: " + str(step) + "\\nINPUT\\n" + str(row[0]) +
                             "\\nOUTPUT\\n" + str(row[1]) + "\\nSCORE: " + str(row[2]))
                 for row in samples]
        swanlab.log({"val/generations": texts}, step=step)

'''
    # Login-node Python3.6 may edit this text, but must not parse Python3.12 source.
    text=text[:start]+replacement+text[end:];p.write_text(text,encoding='utf-8')
    p=source/'verl/trainer/ppo/ray_trainer.py';text=p.read_text(encoding='utf-8')
    marker='            if "request_id" in batch.non_tensor_batch:'
    assert text.count(marker)==1,'Unexpected rollout logging code'
    insert='''            reward_extra_infos_to_dump["output_tokens"] = batch.batch["response_mask"].sum(-1).cpu().tolist()
            if "data_source" in batch.non_tensor_batch:
                reward_extra_infos_to_dump["data_source"] = batch.non_tensor_batch["data_source"].tolist()
            if "extra_info" in batch.non_tensor_batch:
                reward_extra_infos_to_dump["sample_id"] = [
                    None if v.get("id", v.get("index")) is None else str(v.get("id", v.get("index")))
                    for v in batch.non_tensor_batch["extra_info"]]
'''
    text=text.replace(marker,insert+marker);p.write_text(text,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-pack',type=Path,default=Path.home()/'formal-command-pack-20261108');a=p.parse_args()
    pack=Path(__file__).resolve().parent
    manifest=json.loads((pack/'payload-sha256.json').read_text())
    for name,digest in manifest.items():
        if sha(pack/name)!=digest:raise ValueError('Transfer mismatch: '+name)
    pointer=pack/'candidate-path.txt'
    if pointer.exists():
        candidate=Path(pointer.read_text().strip());verify_receipt(candidate)
        print('ALREADY_PREPARED='+str(candidate));return
    session=Path((a.legacy_pack/'session-path.txt').read_text().strip())
    allowed=Path('/groups/xitucheng213/home/share/u2025210309').resolve()
    if session.resolve().parent!=allowed:raise ValueError('Unexpected session root')
    prior=Path((a.legacy_pack/'resume100-path.txt').read_text().strip())
    if (prior/'run/exit-code.txt').read_text().strip()!='0':raise ValueError('Old continuation has not completed')
    old=session/'train';frozen={str(old/'resolved-config.yaml'):sha(old/'resolved-config.yaml')}
    audit=json.loads((session/'manifests/data-audit.json').read_text(encoding='utf-8'))
    for name in ('train.json','val.json','test.json','data-audit.json'):
        path=session/'manifests'/name;frozen[str(path)]=sha(path)
    prior_receipt=json.loads((prior/'resume-receipt.json').read_text())
    for name,digest in prior_receipt['manifests'].items():
        if sha(session/name)!=digest:raise ValueError('Reviewed manifest changed: '+name)
    for split in ('train','validation'):
        for row in audit[split]:
            if sha(row['source'])!=row['sha256']:raise ValueError('Annotation changed: '+row['source'])
            frozen[row['source']]=row['sha256']
    candidate=Path(tempfile.mkdtemp(prefix='newreward30-',dir=str(session)));source=candidate/'source'
    base=restore_source(session/'repo',old,source)
    for src in (pack/'payload').rglob('*'):
        if src.is_file():
            target=source/src.relative_to(pack/'payload');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(src),str(target))
    patch_logging(source)
    (candidate/'run').mkdir()
    receipt=dict(session=str(session),original_config=str(old/'resolved-config.yaml'),base_commit=base,
        startup_snapshot_sha256=sha(old/'changed-source.tar'),steps=30,start='base-model-no-resume',
        reward='meeting-20261111',reward_sha256=sha(source/'examples/cfpo_trainer/reward_function/base.py'),
        max_response_length=2048,warmup_steps=10,authorization='Bao requested original model 30 steps with this reward',
        frozen_inputs=frozen,source_files={str(f.relative_to(source)):sha(f) for f in source.rglob('*')
            if f.is_file() and '.git' not in f.relative_to(source).parts})
    (candidate/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    pointer.write_text(str(candidate)+'\n',encoding='utf-8')
    print('VALIDATION30_PREPARED='+str(candidate));print('NO_GPU_JOB_SUBMITTED')

if __name__=='__main__':main()
