#!/usr/bin/env python3
"""Audit all matrices in an IMOD .xf file and report affine components."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from imod_affine import diagnose_matrix, read_xf

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--xf',type=Path,required=True); p.add_argument('--out-prefix',type=Path,default=None); a=p.parse_args()
    xf=a.xf.resolve(); matrices,shifts=read_xf(xf); rows=[]
    for i,(m,s) in enumerate(zip(matrices,shifts,strict=True)):
        d=diagnose_matrix(m).__dict__; rows.append({'tilt_index':i,'a11':m[0,0],'a12':m[0,1],'a21':m[1,0],'a22':m[1,1],'dx_px':s[0],'dy_px':s[1],**d})
    prefix=(a.out_prefix or xf.with_suffix('')).resolve(); jp=Path(str(prefix)+'.affine_audit.json'); tp=Path(str(prefix)+'.affine_audit.tsv')
    summary={'xf':str(xf),'n_tilts':len(rows),'rotation_deg_range':[min(r['rotation_deg'] for r in rows),max(r['rotation_deg'] for r in rows)],'determinant_range':[min(r['determinant'] for r in rows),max(r['determinant'] for r in rows)],'max_anisotropy_ratio':max(r['anisotropy_ratio'] for r in rows),'max_shear_offdiag':max(r['shear_offdiag'] for r in rows),'max_orthogonality_error':max(r['orthogonality_error'] for r in rows),'rows':rows}
    jp.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    with tp.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),delimiter='\t'); w.writeheader(); w.writerows(rows)
    print(f'Wrote: {jp}\nWrote: {tp}'); return 0
if __name__=='__main__': raise SystemExit(main())
