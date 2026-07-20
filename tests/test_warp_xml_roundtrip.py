from __future__ import annotations
import importlib.util, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from imod_affine import forward_points_pixels, read_xf, regular_grid_points
try:
    import warpylib  # noqa: F401
    HAVE_WARP=True
except Exception:
    HAVE_WARP=False

@unittest.skipUnless(HAVE_WARP,'warpylib unavailable')
class WarpXMLRoundtrip(unittest.TestCase):
    def test_full_affine_roundtrip(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); data=td/'data'; subprocess.run([sys.executable,str(root/'scripts/generate_synthetic_affine_test.py'),'--out-dir',str(data)],check=True)
            inp=td/'staging'; ts=inp/'TS_synthetic'; ts.mkdir(parents=True); shutil.copy2(data/'synthetic_raw.mrc',ts/'TS_synthetic.st'); shutil.copy2(data/'synthetic.tlt',ts/'TS_synthetic.rawtlt'); shutil.copy2(data/'known.xf',ts/'TS_synthetic.xf'); shutil.copy2(data/'known.xf',ts/'TS_synthetic.source.xf')
            out=td/'warp'; subprocess.run([sys.executable,str(root/'scripts/etomo_to_warp.py'),'--input-dir',str(inp),'--output-dir',str(out),'--tilt-axis-angle','84','--volume-shape','257','193','64','--output-pixel-size','10','--alignment-mode','full-affine','--axis-frame','aligned','--movement-grid-shape','5','5'],check=True)
            exported=td/'exported.xf'; subprocess.run([sys.executable,str(root/'scripts/warp_to_imod_affine.py'),'--xml',str(out/'TS_synthetic.xml'),'--source-frame','raw','--input-shape','257,193','--input-pixel-size','10','--out-xf',str(exported),'--rms-tolerance-px','0.03','--max-tolerance-px','0.08'],check=True)
            A0,d0=read_xf(data/'known.xf'); A1,d1=read_xf(exported); pts=regular_grid_points((257,193),9,7); errors=[]
            for x0,s0,x1,s1 in zip(A0,d0,A1,d1,strict=True): errors.append(forward_points_pixels(pts,x0,s0,(257,193))-forward_points_pixels(pts,x1,s1,(257,193)))
            self.assertLess(np.max(np.linalg.norm(np.concatenate(errors),axis=1)),0.10)

if __name__=='__main__': unittest.main()
