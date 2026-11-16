"""Audit the first complete rollout while training continues; flag zeros without cancelling."""
import json
import os
from pathlib import Path
import sys
import time

def inspect(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    if len(rows)!=1920:raise ValueError('First rollout is not exactly1920 records')
    for r in rows:
        for k in ('output','accuracy','format','score','output_tokens'):
            if k not in r:raise ValueError('Missing '+k)
        if abs(float(r['score'])-(.9*float(r['accuracy'])+.1*float(r['format'])))>1e-6:raise ValueError('Reward component mismatch')
    return dict(step=int(Path(path).stem),rows=len(rows),accuracy=sum(float(r['accuracy']) for r in rows)/len(rows),
        format=sum(float(r['format']) for r in rows)/len(rows),reward=sum(float(r['score']) for r in rows)/len(rows),
        at_length_limit=sum(int(r['output_tokens'])>=2048 for r in rows),
        with_open=sum('<think>' in r['output'] for r in rows),with_close=sum('</think>' in r['output'] for r in rows),
        with_boxed=sum('\\boxed{' in r['output'] for r in rows),
        empty=sum(not r['output'].strip() for r in rows))

def main():
    run=Path(sys.argv[1]);pid=int(sys.argv[2]);done=set()
    while True:
        for step in [1,10,20,30]:
            if step in done or not (run/'generations'/('{}.jsonl'.format(step))).exists():continue
            try:report=inspect(run/'generations'/('{}.jsonl'.format(step)))
            except (OSError,ValueError,KeyError):report=None
            if report:
                report['format_zero_warning']=report['format']==0
                (run/('output-audit-{}.json'.format(step))).write_text(json.dumps(report,indent=2),encoding='utf-8')
                if step==1:(run/'first-output-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print('REAL_OUTPUT_AUDIT',json.dumps(report),flush=True);done.add(step)
        try:os.kill(pid,0)
        except ProcessLookupError:break
        time.sleep(10)
    if 1 not in done:raise RuntimeError('Training exited without a complete first rollout; inspect logs')

if __name__=='__main__':main()
