#!/usr/bin/env python3
"""Collect reproducibility information for the affine conversion workflow."""
from __future__ import annotations
import argparse, importlib.metadata, json, platform, shutil, subprocess, sys
from pathlib import Path

PACKAGES=['numpy','mrcfile','torch','torch-projectors','warpylib','miss-alignment']

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    packages={}
    for name in PACKAGES:
        try: packages[name]={'version':importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError: packages[name]={'version':None}
    imports={}
    for name in ['numpy','mrcfile','torch','torch_projectors','warpylib','miss_alignment']:
        try: __import__(name); imports[name]={'status':'OK'}
        except Exception as exc: imports[name]={'status':'FAILED','error':f'{type(exc).__name__}: {exc}'}
    commands={}
    for name in ['newstack','header','miss-alignment']:
        path=shutil.which(name); item={'path':path}
        if path:
            for args in ([name,'--version'],[name,'-version'],[name,'-h']):
                try:
                    cp=subprocess.run(args,text=True,capture_output=True,timeout=10); text=(cp.stdout+'\n'+cp.stderr).strip()
                    if text: item['version_output']=text[:2000]; break
                except Exception: pass
        commands[name]=item
    report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,'imports':imports,'commands':commands}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(a.out); return 0
if __name__=='__main__': raise SystemExit(main())
