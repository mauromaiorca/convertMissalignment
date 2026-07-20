#!/usr/bin/env python3
"""Validate synthetic IMOD geometry against newstack using known spot positions."""
from __future__ import annotations
import argparse, csv, json, shlex, shutil, subprocess
from pathlib import Path
import mrcfile
import numpy as np
from imod_affine import forward_points_pixels, read_xf, residual_statistics

def module_command(command,module_mode,module_name,init):
    if module_mode=='never': return command,False
    prefix='set -euo pipefail; '
    if init: prefix+=f'source {shlex.quote(init)}; '
    else:
        prefix+="if ! type module >/dev/null 2>&1; then for f in /etc/profile /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do [ -r \"$f\" ] && source \"$f\" >/dev/null 2>&1 || true; type module >/dev/null 2>&1 && break; done; fi; "
    prefix+=f'module load {shlex.quote(module_name)}; exec '+ ' '.join(shlex.quote(str(x)) for x in command)
    return ['bash','-lc',prefix],True

def centroid(image, expected, radius=6):
    x,y=expected; x0=max(0,int(np.floor(x))-radius); x1=min(image.shape[1],int(np.floor(x))+radius+2); y0=max(0,int(np.floor(y))-radius); y1=min(image.shape[0],int(np.floor(y))+radius+2)
    patch=np.asarray(image[y0:y1,x0:x1],float); baseline=np.percentile(patch,20); weights=np.maximum(patch-baseline,0); threshold=0.12*weights.max() if weights.size else 0; weights=np.where(weights>=threshold,weights,0)
    total=weights.sum()
    if total<=0: return np.array([np.nan,np.nan])
    yy,xx=np.mgrid[y0:y1,x0:x1]; return np.array([(weights*xx).sum()/total,(weights*yy).sum()/total])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--test-dir',type=Path,required=True); p.add_argument('--module-mode',choices=['auto','always','never'],default='auto'); p.add_argument('--imod-module',default='imod'); p.add_argument('--module-init-script',default=''); p.add_argument('--rms-tolerance-px',type=float,default=0.20); p.add_argument('--max-tolerance-px',type=float,default=0.50); a=p.parse_args()
    d=a.test_dir.resolve(); raw=d/'synthetic_raw.mrc'; xf=d/'known.xf'; out=d/'synthetic_imod_aligned.ali'; command=['newstack','-input',str(raw),'-output',str(out),'-xform',str(xf),'-mode','2']
    use_module=a.module_mode=='always' or (a.module_mode=='auto' and shutil.which('newstack') is None); run,wrapped=module_command(command,a.module_mode,a.imod_module,a.module_init_script) if use_module else (command,False); cp=subprocess.run(run,text=True,capture_output=True)
    report={'command':run,'module_wrapped':wrapped,'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr,'status':'FAIL'}
    if cp.returncode==0 and out.is_file():
        with mrcfile.open(raw,permissive=True) as r,mrcfile.open(out,permissive=True) as o:
            raw_shape=tuple(r.data.shape); aligned=np.asarray(o.data); report['raw_shape_zyx']=list(raw_shape); report['aligned_shape_zyx']=list(aligned.shape)
        points=[]
        with (d/'ground_truth_points.csv').open() as h:
            for row in csv.DictReader(h): points.append([float(row['x']),float(row['y'])])
        points=np.asarray(points); matrices,shifts=read_xf(xf); shape_xy=(raw_shape[2],raw_shape[1]); all_res=[]; tilts=[]
        for t,(matrix,shift) in enumerate(zip(matrices,shifts,strict=True)):
            expected=forward_points_pixels(points,matrix,shift,shape_xy,shape_xy); observed=np.asarray([centroid(aligned[t],pt) for pt in expected]); residual=observed-expected; stats=residual_statistics(residual); all_res.append(residual); tilts.append({'tilt_index':t,'expected_xy':expected.tolist(),'observed_xy':observed.tolist(),'residual_px':stats})
        overall=residual_statistics(np.concatenate(all_res)); ok=aligned.shape==raw_shape and np.isfinite(np.concatenate(all_res)).all() and overall['rms']<=a.rms_tolerance_px and overall['max']<=a.max_tolerance_px
        report.update({'centroid_residual_px':overall,'rms_tolerance_px':a.rms_tolerance_px,'max_tolerance_px':a.max_tolerance_px,'tilts':tilts,'status':'PASS' if ok else 'FAIL'})
    path=d/'imod_validation.json'; path.write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='PASS': raise SystemExit(f'ERROR: IMOD validation failed; see {path}')
    print(path); return 0
if __name__=='__main__': raise SystemExit(main())
