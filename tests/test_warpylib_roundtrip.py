from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import torch
    from warpylib import CubicGrid
    HAVE=True
except Exception:
    HAVE=False
from imod_affine import build_movement_grid_values, inverse_physical_map

@unittest.skipUnless(HAVE,'warpylib unavailable')
class WarpylibGridTests(unittest.TestCase):
    def test_affine_grid_interpolation(self):
        shape=(257,193); p=10.; n=1; A=np.array([[1.013,0.021],[-0.008,0.987]]); d=np.array([5.2,-3.1])
        vx,vy,off=build_movement_grid_values(A,d,shape,p,p,(5,5)); gx=CubicGrid((5,5,1),torch.tensor(vx)); gy=CubicGrid((5,5,1),torch.tensor(vy))
        rng=np.random.default_rng(1); uv=rng.random((100,2)); coords=torch.tensor(np.column_stack([uv,np.full(100,0.5)]),dtype=torch.float32)
        got=np.column_stack([gx.get_interpolated(coords).detach().numpy(),gy.get_interpolated(coords).detach().numpy()])
        B,b=inverse_physical_map(A,d,p,p); dims=np.array(shape)*p; z=uv*dims; expected=(z-dims/2-b)@(np.eye(2)-B).T
        self.assertLess(np.max(np.linalg.norm(got-expected,axis=1))/p,0.02)

if __name__=='__main__': unittest.main()
