#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, shlex, shutil, subprocess, sys
from pathlib import Path

DEFAULT_MODULE_INIT = (
    "/etc/profile.d/modules.sh",
    "/usr/share/Modules/init/bash",
    "/etc/profile.d/lmod.sh",
)

def parse_com(path: Path) -> dict:
    result={"fullimage": None, "thickness": None, "sources": []}
    if not path or not path.is_file(): return result
    try: lines=path.read_text(errors='replace').splitlines()
    except OSError: return result
    for line in lines:
        s=line.strip()
        if not s or s.startswith('#'): continue
        m=re.search(r"(?i)(?:^|\s)-?FULLIMAGE(?:\s+|=)([-+]?\d+)\s*[, ]\s*([-+]?\d+)", s)
        if m:
            result["fullimage"]=[int(m.group(1)),int(m.group(2))]
            result["sources"].append(f"{path}: FULLIMAGE")
        m=re.search(r"(?i)(?:^|\s)-?THICKNESS(?:\s+|=)([-+]?\d+)", s)
        if m:
            result["thickness"]=int(m.group(1))
            result["sources"].append(f"{path}: THICKNESS")
    return result

def module_shell_prefix(init_script: str) -> str:
    if init_script:
        return f"source {shlex.quote(init_script)}"
    checks='; '.join(f'[[ -r {shlex.quote(p)} ]] && source {shlex.quote(p)} && break' for p in DEFAULT_MODULE_INIT)
    return f"if ! command -v module >/dev/null 2>&1; then for f in {' '.join(shlex.quote(p) for p in DEFAULT_MODULE_INIT)}; do [[ -r \"$f\" ]] && source \"$f\" && break; done; fi"

def run_header_via_module(path: Path, module_name: str, init_script: str) -> tuple[list[int] | None, str | None, str | None]:
    prefix=module_shell_prefix(init_script)
    script=(f"set -e; {prefix}; command -v module >/dev/null 2>&1; "
            f"module load {shlex.quote(module_name)} >/dev/null 2>&1; "
            f"header -size {shlex.quote(str(path))}")
    cp=subprocess.run(['bash','-lc',script],text=True,capture_output=True,check=False)
    if cp.returncode != 0:
        msg=(cp.stderr or cp.stdout or '').strip()
        return None,None,msg
    nums=[int(x) for x in re.findall(r"\b\d+\b", cp.stdout)]
    if len(nums)>=3:
        return nums[:3], f"IMOD header via module {module_name}: {path}", None
    return None,None,'header -size returned no XYZ triplet'

def header_size(path: Path, module_mode: str, module_name: str, init_script: str) -> tuple[list[int] | None, str | None, list[str]]:
    notes=[]
    if not path or not path.is_file(): return None,None,notes
    # Portable path first: no IMOD installation or module is needed.
    try:
        import mrcfile
        with mrcfile.open(path,permissive=True,header_only=True) as m:
            return [int(m.header.nx),int(m.header.ny),int(m.header.nz)], f"mrcfile header: {path}", notes
    except Exception as exc:
        notes.append(f"mrcfile could not read {path}: {exc}")
    exe=shutil.which('header')
    if exe:
        cp=subprocess.run([exe,'-size',str(path)],text=True,capture_output=True,check=False)
        nums=[int(x) for x in re.findall(r"\b\d+\b", (cp.stdout or '')+' '+(cp.stderr or ''))]
        if cp.returncode==0 and len(nums)>=3:
            return nums[:3], f"IMOD header already in PATH: {path}", notes
        notes.append('IMOD header is in PATH but did not return a valid size')
    if module_mode != 'never':
        size,src,err=run_header_via_module(path,module_name,init_script)
        if size: return size,src,notes
        notes.append(f"module load {module_name} unavailable/failed: {err}")
        if module_mode == 'always':
            raise RuntimeError(notes[-1])
    return None,None,notes

def main():
    ap=argparse.ArgumentParser(description='Resolve target IMOD reconstruction geometry without creating a reconstruction.')
    ap.add_argument('--imod-dir',required=True,type=Path)
    ap.add_argument('--data-dir',required=True,type=Path)
    ap.add_argument('--basename',required=True)
    ap.add_argument('--reconstruction-stack',type=Path)
    ap.add_argument('--out-json',required=True,type=Path)
    ap.add_argument('--out-text',required=True,type=Path)
    ap.add_argument('--module-mode',choices=('auto','always','never'),default='auto')
    ap.add_argument('--imod-module',default='imod')
    ap.add_argument('--module-init-script',default='')
    args=ap.parse_args()
    imod=args.imod_dir.resolve(); data=args.data_dir.resolve()
    tilt_com=imod/'tilt.com'; newst_com=imod/'newst.com'
    parsed=parse_com(tilt_com)
    xyz=None; method=None; sources=list(parsed['sources']); notes=[]
    if parsed['fullimage'] and parsed['thickness']:
        xyz=[*parsed['fullimage'],parsed['thickness']]
        method='tilt.com FULLIMAGE + THICKNESS'
    rec=args.reconstruction_stack
    if rec:
        rec=rec if rec.is_absolute() else data/rec
        try: size,src,n=header_size(rec.resolve(),args.module_mode,args.imod_module,args.module_init_script)
        except RuntimeError as exc:
            print(f'ERROR: {exc}',file=sys.stderr); return 2
        notes.extend(n)
        if size: sources.append(src); xyz=xyz or size; method=method or 'reconstruction header'
    if xyz is None and parsed['thickness']:
        raw_candidates=[data/f'{args.basename}.mrc',data/f'{args.basename}.st']
        for p in raw_candidates:
            try: size,src,n=header_size(p,args.module_mode,args.imod_module,args.module_init_script)
            except RuntimeError as exc:
                print(f'ERROR: {exc}',file=sys.stderr); return 2
            notes.extend(n)
            if size:
                xyz=[size[0],size[1],parsed['thickness']]
                sources.extend([src,f'{tilt_com}: THICKNESS'])
                method='raw stack XY + tilt.com THICKNESS'
                break
    payload={
      'basename':args.basename,'data_dir':str(data),'imod_dir':str(imod),
      'tilt_com':str(tilt_com) if tilt_com.exists() else None,
      'newst_com':str(newst_com) if newst_com.exists() else None,
      'target_volume_xyz':xyz,'method':method,'sources':sources,'notes':notes,
      'external_tools':{'module_mode':args.module_mode,'imod_module':args.imod_module,'module_init_script':args.module_init_script},
    }
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(payload,indent=2)+'\n')
    lines=[f"basename: {args.basename}",f"method: {method}",f"target_volume_xyz: {xyz}",f"module_mode: {args.module_mode}",f"imod_module: {args.imod_module}"]
    lines += [f"source: {s}" for s in sources]
    lines += [f"note: {s}" for s in notes]
    args.out_text.write_text('\n'.join(lines)+'\n')
    if xyz:
        print('x'.join(map(str,xyz))); return 0
    print('ERROR: target XYZ could not be resolved.',file=sys.stderr)
    print(f'Checked: {tilt_com}',file=sys.stderr)
    print('Provide --xyz X,Y,Z or --reconstruction-stack FILE.',file=sys.stderr)
    if args.module_mode == 'never':
        print('IMOD module loading is disabled by module_mode=never.',file=sys.stderr)
    else:
        print(f'IMOD module setting: module_mode={args.module_mode}, imod_module={args.imod_module}',file=sys.stderr)
    return 2
if __name__=='__main__': raise SystemExit(main())
