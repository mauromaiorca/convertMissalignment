from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from imod_affine import *

class GridEncodingTests(unittest.TestCase):
    def test_nodes_reproduce_full_affine(self):
        shape=(257,193); p=10.0
        A=np.array([[1.013,0.021],[-0.008,0.987]]); d=np.array([5.2,-3.1])
        vx,vy,offsets=build_movement_grid_values(A,d,shape,p,p,(5,5))
        B,b=inverse_physical_map(A,d,p,p); dims=np.array(shape)*p
        idx=0
        for gy in range(5):
            for gx in range(5):
                z=np.array([gx/4*dims[0],gy/4*dims[1]])
                expected=movement_at_raw_absolute_physical(z,B,b,dims)
                self.assertAlmostEqual(vx[idx],expected[0],places=4); self.assertAlmostEqual(vy[idx],expected[1],places=4); idx+=1
        self.assertTrue(np.allclose(offsets[0],b))
    def test_axis_transform_identity(self): self.assertAlmostEqual(transform_axis_angle_raw_to_aligned(84,np.eye(2)),84)

if __name__=='__main__': unittest.main()
