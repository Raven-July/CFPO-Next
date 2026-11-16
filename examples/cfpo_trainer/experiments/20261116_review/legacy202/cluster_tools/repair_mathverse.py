"""Install the public original image in a private view; no shared writes, Python 3.6+."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

PACK = Path(__file__).resolve().parent
ROOT = Path('/groups/xitucheng213/home/share/yzy/Counterfact-Projects/Datasets')
RELATIVE = 'images_version_6/image_745.png'
IMAGE_SHA = 'b34047a3e26d123e3b71b11020e4184640b4b38cad4919b6f3e315d700437f80'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_view(root, view, replacement, official):
    """Retain original annotations and every sample. Only one path gets public bytes."""
    paths = set()
    val = json.loads((root/'V3-math/MathVerse_val_V3.json').read_text(encoding='utf-8'))
    matches = [row for row in val if row.get('id') == 'MathVerse_3725']
    official_matches = [row for row in official if row.get('id') == 'MathVerse_3725']
    if len(matches) != 1 or matches != official_matches or matches[0]['images'] != [RELATIVE]:
        raise ValueError('Affected annotation does not match pinned public dataset')
    test = json.loads((root/'V3-math/MathVerse-closed_test.json').read_text(encoding='utf-8'))
    for row in val + test:
        paths.update(row.get('images') or [])
    for relative in sorted(paths):
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe image path: '+relative)
        target = view/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == RELATIVE:
            content = replacement.read_bytes()
            if hashlib.sha256(content).hexdigest() != IMAGE_SHA or len(content) != 19100:
                raise ValueError('Public replacement image hash/size mismatch')
            if target.exists():
                if target.is_symlink() or target.read_bytes() != content:
                    raise ValueError('Existing replacement differs')
            else:
                target.write_bytes(content)
        else:
            original = (root/'MathVerse_images'/rel).absolute()
            if target.is_symlink():
                if os.readlink(str(target)) != str(original):
                    raise ValueError('Unexpected existing image link')
            elif target.exists():
                raise ValueError('Unexpected existing image file')
            else:
                target.symlink_to(original)
    return {'validation_rows':len(val), 'test_rows':len(test), 'unique_image_paths':len(paths),
            'replaced_paths':[RELATIVE], 'image_sha256':IMAGE_SHA,
            'original_val_sha256':sha(root/'V3-math/MathVerse_val_V3.json'),
            'original_test_sha256':sha(root/'V3-math/MathVerse-closed_test.json'),
            'dropped_samples':0}


def main():
    session = Path((PACK/'session-path.txt').read_text().strip())
    if not session.is_absolute() or session.parent != Path('/groups/xitucheng213/home/share/u2025210309'):
        raise ValueError('Unexpected session directory')
    # Source updates are permitted only before preparation/training has been submitted.
    for stage in ('prep-batch','config','smoke','train','eval'):
        if (session/stage).exists():
            raise ValueError('Existing stage requires inspection before repair: '+stage)
    manifest = json.loads((PACK/'sources.json').read_text(encoding='utf-8'))
    repo = session/'repo'
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip() != manifest['base_commit']:
        raise ValueError('Source base mismatch')
    expected = {row['path']:row['sha256'] for row in manifest['files']}
    with tarfile.open(str(PACK/'changes.tar')) as archive:
        if set(archive.getnames()) != set(expected):
            raise ValueError('Source archive names differ')
        for member in archive.getmembers():
            rel = Path(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe source member')
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != expected[member.name]:
                raise ValueError('Source archive hash mismatch')
        archive.extractall(str(repo))
    for name,digest in expected.items():
        if sha(repo/name) != digest:
            raise ValueError('Source installation hash mismatch')
    replacement = PACK/'mathverse-repair/image_745.png'
    official_path = PACK/'official-MathVerse_val_V3.json'
    provenance = json.loads((PACK/'mathverse-repair/provenance.json').read_text(encoding='utf-8'))
    if sha(official_path) != provenance['annotation_sha256']:
        raise ValueError('Public annotation hash mismatch')
    view = session/'data/MathVerse_images'
    report = build_view(ROOT, view, replacement, json.loads(official_path.read_text(encoding='utf-8')))
    report['public_source'] = provenance
    (session/'mathverse-repair.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    env_path = session/'session.env'
    lines = [line for line in env_path.read_text().splitlines() if not line.startswith('CFPO_MATHVERSE_IMAGES=')]
    lines.append('CFPO_MATHVERSE_IMAGES='+shlex.quote(str(view)))
    temp = session/'session.env.mathverse-tmp'
    temp.write_text('\n'.join(lines)+'\n')
    os.replace(str(temp),str(env_path))
    subprocess.run(['python3',str(PACK/'audit_validation_access.py'),'--mathverse-images',str(view)],check=True)
    print('MATHVERSE_REPAIR_OK: original image restored privately; zero samples dropped; no GPU submitted.')


if __name__ == '__main__':
    main()
