"""Read-only, Python 3.6+. No GPU, chmod, copying datasets or replacement images."""
import json
import os
from pathlib import Path
import stat
import subprocess

root=Path('/groups/xitucheng213/home/share/yzy/Counterfact-Projects/Datasets')
relative='images_version_6/image_745.png'
image=root/'MathVerse_images'/relative
subprocess.call(['id'])
print('IMAGE:',str(image),flush=True)
try:
    value=image.stat()
    print('FILE_MODE:',stat.filemode(value.st_mode),'UID:',value.st_uid,'GID:',value.st_gid,'SIZE:',value.st_size)
    with image.open('rb') as stream: stream.read(1)
    print('LOGIN_NODE_READ_OK (compute-node access still needs confirmation)')
except OSError as error:
    print('READ_FAILED:',type(error).__name__,'errno=',error.errno,str(error))
for path in sorted((root/'V3-math').glob('*MathVerse*.json')):
    try:
        rows=json.loads(path.read_text(encoding='utf-8'))
        matches=[{'index':i,'id':r.get('id')} for i,r in enumerate(rows) if relative in (r.get('images') or [])]
        print('ANNOTATION:',str(path),'TOTAL:',len(rows),'MATCHES:',json.dumps(matches,ensure_ascii=True))
    except (OSError,ValueError,TypeError) as error:
        print('ANNOTATION_READ_FAILED:',str(path),type(error).__name__,str(error))
try:
    subprocess.call(['namei','-l',str(image)])
except OSError:
    print('namei unavailable; file ownership and read test are shown above')
