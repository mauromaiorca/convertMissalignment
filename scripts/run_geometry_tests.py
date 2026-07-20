#!/usr/bin/env python3
"""Run pure geometry tests, optional warpylib tests, and optional IMOD newstack test."""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--work-dir',type=Path,default=None); p.add_argument('--with-imod',action='store_true'); p.add_argument('--module-mode',choices=['auto','always','never'],default='auto'); p.add_argument('--imod-module',default='imod'); p.add_argument('--module-init-script',default=''); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]; work=(a.work_dir or root/'test_output').resolve(); work.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable, str(root/'scripts/collect_environment.py'), '--out', str(work/'environment.json')], check=False)
    test_cmd=[sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v']
    cp=subprocess.run(test_cmd,text=True,capture_output=True); print(cp.stdout,end=''); print(cp.stderr,end='',file=sys.stderr)
    synth=work/'synthetic'; subprocess.run([sys.executable,str(root/'scripts/generate_synthetic_affine_test.py'),'--out-dir',str(synth),'--overwrite'],check=True)
    imod_status='SKIP'
    if a.with_imod:
        cmd=[sys.executable,str(root/'scripts/validate_with_imod.py'),'--test-dir',str(synth),'--module-mode',a.module_mode,'--imod-module',a.imod_module]
        if a.module_init_script: cmd += ['--module-init-script',a.module_init_script]
        icp=subprocess.run(cmd); imod_status='PASS' if icp.returncode==0 else 'FAIL'
    report={'unit_tests':'PASS' if cp.returncode==0 else 'FAIL','imod_test':imod_status,'synthetic_dir':str(synth)}
    (work/'geometry_test_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"Summary: {work/'geometry_test_summary.json'}")
    return 0 if cp.returncode==0 and imod_status!='FAIL' else 1
if __name__=='__main__': raise SystemExit(main())
