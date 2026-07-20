from __future__ import annotations
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from imod_affine import read_xf

class SyntheticGenerationTests(unittest.TestCase):
    def test_generator(self):
        try: import mrcfile
        except ImportError: self.skipTest('mrcfile unavailable')
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([sys.executable,str(root/'scripts/generate_synthetic_affine_test.py'),'--out-dir',td],check=True,capture_output=True,text=True)
            m=json.loads((Path(td)/'manifest.json').read_text()); A,d=read_xf(Path(td)/'known.xf')
            self.assertEqual(len(A),m['n_tilts']); self.assertEqual(A.shape[1:],(2,2)); self.assertEqual(d.shape,(m['n_tilts'],2))
            with mrcfile.open(Path(td)/'synthetic_raw.mrc') as h: self.assertEqual(tuple(h.data.shape),(m['n_tilts'],193,257))

if __name__=='__main__': unittest.main()
