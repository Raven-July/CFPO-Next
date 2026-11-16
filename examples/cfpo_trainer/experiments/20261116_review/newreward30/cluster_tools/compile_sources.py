"""Parse training source only inside the actual container interpreter, not the login node."""
from pathlib import Path
import sys

def main():
    if sys.version_info < (3,12):
        raise RuntimeError('Use container Python3.12+ to validate training source; current: '+sys.version)
    root=Path(sys.argv[1])
    files=['verl/utils/tracking.py','verl/trainer/ppo/ray_trainer.py',
           'examples/cfpo_trainer/reward_function/base.py','examples/cfpo_trainer/test_meeting_reward.py']
    for name in files:
        path=root/name
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    print('CONTAINER_SOURCE_SYNTAX_OK',sys.version.split()[0],len(files),flush=True)

if __name__=='__main__':main()
