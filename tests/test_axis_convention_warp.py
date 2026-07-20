from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from imod_affine import transform_axis_angle_raw_to_aligned
try:
    import torch
    from warpylib import TiltSeries
    HAVE=True
except Exception:
    HAVE=False

def rotation(deg):
    a=np.deg2rad(deg); return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

@unittest.skipUnless(HAVE,'warpylib unavailable')
class AxisConventionWarpTest(unittest.TestCase):
    def test_image_rotation_matches_axis_angle_transform(self):
        phi_raw=84.0; image_rotation=-7.5; A=rotation(image_rotation)
        phi_aligned=transform_axis_angle_raw_to_aligned(phi_raw,A)
        raw=TiltSeries(n_tilts=1); ali=TiltSeries(n_tilts=1)
        for ts in (raw,ali):
            ts.image_dimensions_physical=torch.tensor([2000.,1600.]); ts.volume_dimensions_physical=torch.tensor([1200.,1000.,800.]); ts.angles=torch.tensor([25.]); ts.size_rounding_factors=torch.ones(3)
        raw.tilt_axis_angles=torch.tensor([phi_raw]); ali.tilt_axis_angles=torch.tensor([phi_aligned])
        coords=torch.tensor([[200.,300.,400.],[700.,500.,250.],[900.,800.,600.]])
        pr=raw.get_positions_in_one_tilt(coords,0)[:,:2].detach().numpy(); pa=ali.get_positions_in_one_tilt(coords,0)[:,:2].detach().numpy(); center=np.array([1000.,800.]); expected=(pr-center)@A.T+center
        self.assertLess(np.max(np.linalg.norm(pa-expected,axis=1)),1e-3)

if __name__=='__main__': unittest.main()
