from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from imod_affine import *

class AffineMathTests(unittest.TestCase):
    def setUp(self):
        self.shape=(257,193); self.p=10.0
        a=np.deg2rad(3.0); r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        self.A=r@np.array([[1.015,0.012],[0.0,0.985]])
        self.d=np.array([5.4,-3.7])
        self.points=regular_grid_points(self.shape,9,7)
    def test_forward_inverse(self):
        out=forward_points_pixels(self.points,self.A,self.d,self.shape,self.shape)
        back=inverse_points_pixels(out,self.A,self.d,self.shape,self.shape)
        self.assertLess(np.max(np.abs(back-self.points)),1e-10)
    def test_physical_inverse(self):
        B,b=inverse_physical_map(self.A,self.d,self.p,self.p)
        centered=(self.points-image_center_xy(self.shape))*self.p
        aligned=(forward_points_pixels(self.points,self.A,self.d,self.shape,self.shape)-image_center_xy(self.shape))*self.p
        recovered=aligned@B.T+b
        self.assertLess(np.max(np.abs(recovered-centered)),1e-9)
    def test_warp_component_formula(self):
        B,b=inverse_physical_map(self.A,self.d,self.p,self.p)
        dims=np.array(self.shape)*self.p
        def movement(z): return movement_at_raw_absolute_physical(z,B,b,dims)
        q=np.array([[0.,0.],[-700.,-400.],[600.,350.]])
        got=evaluate_inverse_affine_from_warp_components(q,b,movement,dims)
        expected=q@B.T+b
        self.assertLess(np.max(np.abs(got-expected)),1e-9)
    def test_roundtrip_homogeneous(self):
        H=xf_to_homogeneous(self.A,self.d,self.shape,self.shape)
        A2,d2=homogeneous_to_xf(H,self.shape,self.shape)
        self.assertTrue(np.allclose(A2,self.A,atol=1e-12)); self.assertTrue(np.allclose(d2,self.d,atol=1e-12))
    def test_residual_composition(self):
        Ar=np.array([[1.,0.001],[-0.002,1.]])
        dr=np.array([1.3,-0.8])
        Af,df=compose_xf(self.A,self.d,Ar,dr,self.shape,self.shape,self.shape)
        direct=forward_points_pixels(forward_points_pixels(self.points,self.A,self.d,self.shape,self.shape),Ar,dr,self.shape,self.shape)
        composed=forward_points_pixels(self.points,Af,df,self.shape,self.shape)
        self.assertLess(np.max(np.abs(direct-composed)),1e-10)
    def test_different_input_output_centres(self):
        output_shape=(224,176)
        transformed=forward_points_pixels(self.points,self.A,self.d,self.shape,output_shape)
        recovered=inverse_points_pixels(transformed,self.A,self.d,self.shape,output_shape)
        self.assertLess(np.max(np.abs(recovered-self.points)),1e-10)

    def test_physical_composition_with_different_pixels(self):
        raw_pixel=2.0; ali_pixel=4.0; final_pixel=2.5
        A0=self.A; d0=self.d
        Ar=np.array([[0.999,0.002],[-0.001,1.001]])
        dr=np.array([0.75,-0.4])
        M0=(ali_pixel/raw_pixel)*A0; t0=ali_pixel*d0
        Mr=Ar; tr=ali_pixel*dr
        Mf=Mr@M0; tf=Mr@t0+tr
        Af=(raw_pixel/final_pixel)*Mf; df=tf/final_pixel
        raw_centered=np.array([[10.0,-7.0],[0.0,0.0],[-25.0,12.0]])
        direct_A=raw_centered*raw_pixel
        direct_A=direct_A@M0.T+t0
        direct_A=direct_A@Mr.T+tr
        via_xf=(raw_centered@Af.T+df)*final_pixel
        self.assertLess(np.max(np.abs(direct_A-via_xf)),1e-10)

    def test_affine_fit(self):
        y=self.points@self.A.T+self.d
        A,d,res=fit_affine(self.points,y)
        self.assertTrue(np.allclose(A,self.A,atol=1e-12)); self.assertTrue(np.allclose(d,self.d,atol=1e-12)); self.assertLess(residual_statistics(res)['max'],1e-10)

if __name__=='__main__': unittest.main()
