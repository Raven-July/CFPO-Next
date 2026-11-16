"""Collect complete results; never fill missing inference rows or scores with zero."""
import json
import os
from pathlib import Path
from protocol import summarize, verify_package, verify_prepared


def main():
    pack = Path(__file__).resolve().parent
    verify_package(pack)
    candidate = Path(os.environ['CANDIDATE'])
    verify_prepared(candidate)
    run = candidate/'run'
    rows = []
    for backend in ['hf', 'vllm']:
        with (run/(backend+'-outputs.jsonl')).open(encoding='utf-8') as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    summary = summarize(rows)
    audits = [json.loads(line) for line in (run/'engine-prompt-audit.jsonl').read_text().splitlines()]
    if len(audits) != 8 or not all(a['exact_match'] and a['actual_expanded_token_ids'] == a['expected_expanded_token_ids'] for a in audits):
        raise ValueError('Eight exact engine input checks required')
    comparisons = []
    for sid in sorted({r['sample_id'] for r in rows}):
        selected = {(r['backend'], r['arm']):r for r in rows if r['sample_id'] == sid}
        a, b, c = [selected[key] for key in [('hf','original'),('vllm','original'),('vllm','simple')]]
        comparisons.append(dict(sample_id=sid, hf_original_accuracy=a['accuracy'],
            vllm_original_accuracy=b['accuracy'], vllm_simple_accuracy=c['accuracy'],
            original_backend_text_identical=a['output']==b['output'],
            note='Different greedy outputs alone do not prove a backend bug; inspect semantics and numerics'))
    summary['paired_samples'] = comparisons
    (run/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (run/'outputs.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
