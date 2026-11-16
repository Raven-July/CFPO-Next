"""Read-only access audit of the five validation sources; Python 3.6+, no GPU."""
import argparse
import json
from pathlib import Path
import stat
import sys

ROOT=Path('/groups/xitucheng213/home/share/yzy/Counterfact-Projects/Datasets')
SOURCES=[('V3-math/geometry3k_val_V3.json','geometry3k_images'),
         ('V3-math/LogicVista_val_V3.json','LogicVista_images'),
         ('V3-math/MathVerse_val_V3.json','MathVerse_images'),
         ('V3/C-VQA-Real_val_V3.json','COCO_val2014_images'),
         ('V3/MARS_Bench_val_V3.json','COCO_val2014_images')]


def audit(root, mathverse_images=None):
    failures=[]; checked={}; sources=[]
    for annotation,image_dir in SOURCES:
        source=root/annotation
        try: rows=json.loads(source.read_text(encoding='utf-8'))
        except (OSError,ValueError) as error:
            failures.append({'kind':'annotation','path':str(source),'error_type':type(error).__name__,'error':str(error)})
            continue
        print('ACCESS_CHECK:',str(source),'rows=',len(rows),flush=True)
        sources.append({'annotation':str(source),'rows':len(rows)})
        for index,row in enumerate(rows):
            for relative in (row.get('images') or []):
                image=(Path(mathverse_images) if mathverse_images and image_dir=='MathVerse_images' else root/image_dir)/relative; key=str(image)
                if key not in checked:
                    try:
                        with image.open('rb') as stream: stream.read(1)
                        checked[key]=None
                    except OSError as error:
                        record={'kind':'image','path':key,'error_type':type(error).__name__,'errno':error.errno,'error':str(error)}
                        try:
                            info=image.stat(); record.update(mode=stat.filemode(info.st_mode),uid=info.st_uid,gid=info.st_gid)
                        except OSError: pass
                        checked[key]=record
                if checked[key] is not None:
                    failures.append(dict(checked[key],annotation=str(source),source_index=index,id=row.get('id')))
    return {'scope':'File read access only; decoding and compute-node access still require preflight',
            'sources':sources,'unique_images_checked':len(checked),'failures':failures,
            'all_validation_files_readable':not failures and len(sources)==5}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--mathverse-images',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'validation-access-report.json')
    args=p.parse_args(); report=audit(args.data_root,args.mathverse_images)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    paths=sorted({f['path'] for f in report['failures']})
    path_list=args.output.with_suffix('.unreadable-paths.txt')
    path_list.write_text('\n'.join(paths)+('\n' if paths else ''),encoding='utf-8')
    print('UNREADABLE_PATHS:',len(paths))
    for path in paths[:8]: print(path)
    print('REPORT:',args.output); print('UNREADABLE_LIST:',path_list)
    if not report['all_validation_files_readable']:
        print('VALIDATION_ACCESS_BLOCKED: request read access from the data owner; do not omit samples')
        return 2
    print('VALIDATION_ACCESS_OK'); return 0


if __name__=='__main__': sys.exit(main())
