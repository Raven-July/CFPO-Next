"""Accept only a real 30-step run with complete validation and native text registration."""
import json
from pathlib import Path
import re
import sqlite3
import sys

def main():
    run=Path(sys.argv[1]);text=(run/'train.out').read_text(encoding='utf-8',errors='replace')
    steps=[int(v) for v in re.findall(r'training/global_step:\s*(\d+)',text)]
    if not steps or max(steps)!=30 or set(steps)!=set(range(1,31)):raise ValueError('Not exactly30 completed updates')
    counts={}
    for step in [0,5,10,15,20,25,30]:
        p=run/'validation'/('{}.jsonl'.format(step))
        with p.open(encoding='utf-8') as stream:rows=[json.loads(s) for s in stream if s.strip()]
        if len(rows)!=2848:raise ValueError('Incomplete validation step '+str(step))
        counts[str(step)]=len(rows)
    with (run/'generations/30.jsonl').open(encoding='utf-8') as stream:train=[json.loads(s) for s in stream if s.strip()]
    if len(train)!=1920:raise ValueError('Incomplete final rollout')
    checkpoint=run/'checkpoints/global_step_30'
    files=['data.pt','actor/fsdp_config.json']+['actor/{}_world_size_2_rank_{}.pt'.format(k,r)
        for r in (0,1) for k in ('model','optim','extra_state')]
    for name in files:
        path=checkpoint/name
        if not path.is_file() or path.stat().st_size==0:raise ValueError('Missing checkpoint '+name)
    db=sqlite3.connect((run/'swanlog/runs.swanlab').resolve().as_uri()+'?mode=ro',uri=True)
    tags=db.execute("select name,type from tag where name in ('train/accuracy','train/format','train/reward','val/generations')").fetchall();db.close()
    if not {('train/accuracy','line'),('train/format','line'),('train/reward','line'),('val/generations','text')}.issubset(set(tags)):
        raise ValueError('Required native charts not registered')
    report=dict(completed_updates=30,validation=counts,final_training_rows=len(train),native_tags=tags,
        format_mean=sum(float(r['format']) for r in train)/len(train),
        accuracy_mean=sum(float(r['accuracy']) for r in train)/len(train),
        scope='30-step new reward verification, not completed202-step baseline or final ten-dataset evaluation')
    (run/'completion-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('VALIDATION30_ARTIFACTS_ACCEPTED',json.dumps(report))

if __name__=='__main__':main()
