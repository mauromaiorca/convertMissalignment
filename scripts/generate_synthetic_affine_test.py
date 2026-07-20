#!/usr/bin/env python3
"""Generate a compact asymmetric MRC tilt stack and known IMOD transforms."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import mrcfile
from imod_affine import write_xf


def rotation(deg: float) -> np.ndarray:
    a=np.deg2rad(deg); return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

def cases():
    return [
        ("identity", np.eye(2), np.array([0.,0.])),
        ("translation", np.eye(2), np.array([7.25,-4.5])),
        ("rotation", rotation(4.0), np.array([0.,0.])),
        ("anisotropic_scale", np.diag([1.02,0.98]), np.array([0.,0.])),
        ("shear", np.array([[1.,0.02],[-0.01,1.]]), np.array([0.,0.])),
        ("combined", rotation(3.0) @ np.array([[1.015,0.012],[0.,0.985]]), np.array([5.4,-3.7])),
        ("combined_inverse_sign", rotation(-2.5) @ np.array([[0.99,-0.015],[0.008,1.01]]), np.array([-6.1,2.2])),
    ]

def gaussian_image(shape_xy, points, sigma=1.25):
    nx,ny=shape_xy; yy,xx=np.mgrid[0:ny,0:nx]; image=np.zeros((ny,nx),np.float32)
    for idx,(x,y) in enumerate(points):
        amp=1.0+0.17*idx
        image += amp*np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma)).astype(np.float32)
    image += (xx/nx*0.013 + yy/ny*0.021).astype(np.float32)
    return image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True); p.add_argument('--shape',default='257,193'); p.add_argument('--pixel-size',type=float,default=10.0); p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); out=a.out_dir.resolve()
    if out.exists() and any(out.iterdir()) and not a.overwrite: raise SystemExit(f'ERROR: non-empty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    nx,ny=(int(v) for v in a.shape.replace('x',',').split(','))
    points=np.array([[19.5,17.25],[nx-28.75,21.5],[44.25,71.75],[nx/2+0.3,ny/2-0.2],[nx-31.4,ny-65.8],[27.1,ny-24.3],[nx-72.5,ny-20.7],[91.2,ny-57.6],[62.8,112.4]],float)
    cs=cases(); base=gaussian_image((nx,ny),points); stack=np.stack([base*(1+0.01*i) for i in range(len(cs))]).astype(np.float32)
    mrc=out/'synthetic_raw.mrc'
    with mrcfile.new(mrc,overwrite=True) as h: h.set_data(stack); h.voxel_size=a.pixel_size
    mats=np.stack([x[1] for x in cs]); shifts=np.stack([x[2] for x in cs]); write_xf(out/'known.xf',mats,shifts)
    (out/'synthetic.tlt').write_text('\n'.join(str(v) for v in np.linspace(-30,30,len(cs)))+'\n')
    with (out/'ground_truth_points.csv').open('w',newline='') as h:
        w=csv.writer(h); w.writerow(['id','x','y']); [w.writerow([i,*pt]) for i,pt in enumerate(points)]
    manifest={'shape_xy':[nx,ny],'n_tilts':len(cs),'pixel_size_A':a.pixel_size,'cases':[{'name':n,'matrix':m.tolist(),'shift_px':s.tolist()} for n,m,s in cs]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)
    return 0
if __name__=='__main__': raise SystemExit(main())
