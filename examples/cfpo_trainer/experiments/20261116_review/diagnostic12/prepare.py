"""Freeze existing successful source and four bundled source images; Python3.6+."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from protocol import FIXED, REWARD_SHA, SAMPLE_IDS, sha, verify_package, verify_prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, default=Path(
        '/groups/xitucheng213/home/share/u2025210309/cfpo-formal-20261109-F3PXx3/newreward30-v87nfmcw'))
    args = parser.parse_args()
    pack = Path(__file__).resolve().parent
    verify_package(pack)
    pointer = pack/'candidate-path.txt'
    if pointer.exists():
        verify_prepared(pointer.read_text(encoding='utf-8').strip())
        print('ALREADY_PREPARED='+pointer.read_text(encoding='utf-8').strip())
        return
    previous = args.previous
    if (previous/'run/exit-code.txt').read_text().strip() != '0':
        raise ValueError('Expected the successful new30 source snapshot')
    old = json.loads((previous/'receipt.json').read_text(encoding='utf-8'))
    runtime = json.loads((previous/'run/runtime-audit.json').read_text(encoding='utf-8'))
    if runtime['reward_sha256'] != REWARD_SHA:
        raise ValueError('Wrong saved reward')
    source = previous/'source'
    frozen = {}
    for name, digest in old['source_files'].items():
        path = source/name
        if sha(path) != digest:
            raise ValueError('Successful source snapshot changed: '+name)
        frozen[str(path)] = digest
    session = Path(old['session'])
    candidate = Path(tempfile.mkdtemp(prefix='diagnostic12-', dir=str(session)))
    (candidate/'run/source-images').mkdir(parents=True)
    samples = json.loads((pack/'samples.json').read_text(encoding='utf-8'))
    if [r['sample_id'] for r in samples] != SAMPLE_IDS:
        raise ValueError('Wrong four samples')
    for sample in samples:
        files = []
        for name, digest in zip(sample['images'], sample['image_sha256']):
            image = pack/name
            if sha(image) != digest:
                raise ValueError('Bundled image changed')
            target = candidate/'run/source-images'/image.name
            shutil.copyfile(str(image), str(target))
            frozen[str(target)] = digest
            files.append(str(target))
        sample['images'] = files
    sample_path = candidate/'samples.json'
    sample_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
    frozen[str(sample_path)] = sha(sample_path)
    model = Path(runtime['model'])
    if model.name != 'Qwen3-VL-2B-Instruct':
        raise ValueError('Not the original Instruct model')
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json',
                 'preprocessor_config.json', 'processor_config.json', 'chat_template.json', 'chat_template.jinja']:
        if (model/name).is_file():
            frozen[str(model/name)] = sha(model/name)
    # Weights remain on the cluster. Record file sizes/mtime; do not pretend to hash all weights.
    weights = {p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
               for p in model.glob('*.safetensors')}
    if not weights:
        raise ValueError('No original safetensors weights found')
    receipt = dict(source=str(source), session=str(session), model=str(model), fixed=FIXED,
        total_outputs=12, reward_sha256=REWARD_SHA, frozen_files=frozen, model_weight_metadata=weights,
        resources=dict(gpus=1, cpus=8, memory_GiB=128, max_hours=1),
        scope='No checkpoint restoration, optimizer or model updates; no claim of full weight hash verification')
    (candidate/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    verify_prepared(candidate)
    pointer.write_text(str(candidate)+'\n', encoding='utf-8')
    print('PREPARED='+str(candidate)+'; no GPU job submitted')


if __name__ == '__main__':
    main()
