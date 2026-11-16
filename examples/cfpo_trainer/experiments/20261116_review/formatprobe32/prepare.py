"""Freeze the selected annotations, source images and known successful source tree. Python3.6+."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from core import sha,verify_package,verify_prepared,FIXED

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation-pack',type=Path,default=Path.home()/'validation30-pack-20261115');a=p.parse_args()
    pack=Path(__file__).resolve().parent;verify_package(pack)
    pointer=pack/'candidate-path.txt'
    if pointer.exists():
        candidate=Path(pointer.read_text(encoding='utf-8').strip());verify_prepared(candidate)
        print('FORMAT_PROBE_ALREADY_PREPARED='+str(candidate));return
    previous=Path((a.validation_pack/'candidate-path.txt').read_text(encoding='utf-8').strip())
    if (previous/'run/exit-code.txt').read_text().strip()!='0':raise ValueError('30-step run is not successful yet')
    old=json.loads((previous/'receipt.json').read_text(encoding='utf-8'));session=Path(old['session']);source=previous/'source'
    runtime=json.loads((previous/'run/runtime-audit.json').read_text(encoding='utf-8'))
    if runtime['reward_sha256']!=json.loads((pack/'plan.json').read_text())['fixed']['reward_sha256']:raise ValueError('Wrong reward contract')
    frozen={}
    for name,digest in old['source_files'].items():
        path=source/name
        if sha(path)!=digest:raise ValueError('Successful source snapshot changed: '+name)
        frozen[str(path)]=digest
    val_manifest=session/'manifests/val.json';frozen[str(val_manifest)]=sha(val_manifest)
    entries={Path(r['dataset']).stem:r for r in json.loads(val_manifest.read_text(encoding='utf-8'))}
    plan=json.loads((pack/'plan.json').read_text(encoding='utf-8'));selected=plan['samples']
    if len(selected)!=32:raise ValueError('Expected32 diagnostic samples')
    history=previous/'run/validation/30.jsonl';history_rows={}
    with history.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);history_rows[(row['data_source'],str(row['sample_id']))]=row
    frozen[str(history)]=sha(history)
    candidate=Path(tempfile.mkdtemp(prefix='formatprobe32-',dir=str(session)));(candidate/'run/source-images').mkdir(parents=True)
    image_manifest=json.loads((previous/'run/image-hashes.json').read_text(encoding='utf-8'))
    cache={};samples=[]
    for i,item in enumerate(selected):
        source_name=item['data_source'];sample_id=str(item['sample_id']);entry=entries[source_name]
        annotation=Path(entry['dataset'])
        if str(annotation) not in old['frozen_inputs'] or sha(annotation)!=old['frozen_inputs'][str(annotation)]:raise ValueError('Annotation changed')
        frozen[str(annotation)]=sha(annotation)
        if source_name not in cache:
            rows=json.loads(annotation.read_text(encoding='utf-8'));cache[source_name]=default_index(rows)
        if sample_id not in cache[source_name]:raise ValueError('Missing selected sample '+sample_id)
        row=cache[source_name][sample_id];prior=history_rows[(source_name,sample_id)]
        if __import__('hashlib').sha256(prior['input'].encode()).hexdigest()!=item['input_sha256']:raise ValueError('Selected history changed')
        if str(row['answer'])!=str(prior['gts']):raise ValueError('Ground truth differs')
        images=[];hashes=[]
        if row['problem'].count('<image>')!=len(row.get('images') or []):raise ValueError('Image placeholder mismatch')
        for j,name in enumerate(row.get('images') or []):
            path=Path(entry['image'])/name;digest=sha(path)
            if image_manifest.get(str(path.resolve()))!=digest:raise ValueError('Image differs from30-step run: '+str(path))
            target=candidate/'run/source-images'/('{}-{}{}'.format(i,j,path.suffix or '.img'))
            shutil.copyfile(str(path),str(target));images.append(str(target));hashes.append(digest)
        samples.append(dict(sample_id=sample_id,data_source=source_name,problem=row['problem'],answer=str(row['answer']),
            images=images,image_sha256=hashes,expected_base_input_sha256=item['input_sha256'],diagnostic_stratum=item['diagnostic_stratum']))
    sample_path=candidate/'samples.json';sample_path.write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
    frozen[str(sample_path)]=sha(sample_path)
    model=Path(runtime['model'])
    for name in ['config.json','tokenizer_config.json','tokenizer.json','chat_template.json','chat_template.jinja','generation_config.json']:
        if (model/name).is_file():frozen[str(model/name)]=sha(model/name)
    receipt=dict(source=str(source),session=str(session),previous_run=str(previous/'run'),model=str(model),
        reward_sha256=runtime['reward_sha256'],fixed=FIXED,total_outputs=480,authorization='Bao requested next small-scale validation based on the diagnostic plan',
        resources=dict(gpus=1,cpus=8,memory_GiB=128,max_hours=4),frozen_files=frozen,
        selection='Outcome-stratified diagnostic sample set, not an unbiased benchmark')
    (candidate/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    pointer.write_text(str(candidate)+'\n',encoding='utf-8')
    print('FORMAT_PROBE_PREPARED='+str(candidate));print('32 samples,3 arms,5 outputs each; no GPU job submitted')

def default_index(rows):
    result={}
    for i,row in enumerate(rows):
        key=str(row.get('id',i))
        if key in result:raise ValueError('Duplicate annotation ID '+key)
        result[key]=row
    return result

if __name__=='__main__':main()
