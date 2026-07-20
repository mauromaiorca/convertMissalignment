from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mrcfile
import numpy as np


class GenerateAlignedStackOrchestrationTests(unittest.TestCase):
    def test_standard_input_generation_and_param_update(self):
        root = Path(__file__).resolve().parents[1]
        generator = root / "scripts" / "generate_aligned_stack.py"
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            bin_dir = tmp / "bin"
            imod_dir = tmp / "imod"
            bin_dir.mkdir()
            imod_dir.mkdir()

            fake_newstack = bin_dir / "newstack"
            fake_newstack.write_text(
                f"#!{sys.executable}\n"
                "import re,sys\n"
                "from pathlib import Path\n"
                "import mrcfile,numpy as np\n"
                "text=sys.stdin.read()\n"
                "values={}\n"
                "for line in text.splitlines():\n"
                " m=re.match(r'\\s*([A-Za-z][A-Za-z0-9_]*)\\s+(.*)',line)\n"
                " if m: values[m.group(1).lower()]=m.group(2).strip()\n"
                "inp=Path(values['inputfile']); out=Path(values['outputfile'])\n"
                "size=[int(x) for x in re.findall(r'\\d+',values['sizetooutput'])[:2]]\n"
                "factor=float(values.get('binbyfactor','1'))\n"
                "with mrcfile.open(inp,permissive=True) as h:\n"
                " z=int(h.data.shape[0]); pixel=float(h.voxel_size.x)\n"
                "with mrcfile.new(out,overwrite=True) as h:\n"
                " h.set_data(np.zeros((z,size[1],size[0]),dtype=np.float32)); h.voxel_size=pixel*factor\n"
            )
            fake_newstack.chmod(0o755)

            raw = tmp / "raw.mrc"
            with mrcfile.new(raw, overwrite=True) as handle:
                handle.set_data(np.zeros((3, 48, 96), dtype=np.float32))
                handle.voxel_size = 4.0
            xf = tmp / "final.xf"
            xf.write_text("1 0 0 1 0 0\n" * 3)
            tlt = tmp / "final.tlt"
            tlt.write_text("-1\n0\n1\n")
            (imod_dir / "newst.com").write_text(
                "$newstack -StandardInput\n"
                "InputFile old.st\n"
                "OutputFile old.ali\n"
                "TransformFile old.xf\n"
                "SizeToOutput 64,80\n"
                "BinByFactor 2\n"
                "ModeToOutput 2\n"
            )
            params = tmp / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "series_name": "synthetic",
                        "imod_dir": str(imod_dir),
                        "files": {
                            "raw_stack": str(raw),
                            "final_xf": str(xf),
                            "final_tilt": str(tlt),
                            "aligned_stack": None,
                        },
                        "geometry": {
                            "raw_pixel_size_A": 4.0,
                            "target_output_pixel_size_A": 8.0,
                            "target_volume_shape_xyz": [64, 80, 20],
                        },
                        "conditions": {"ali_identity": {}},
                        "mrc_headers": {},
                        "counts": {},
                    }
                )
            )
            output = tmp / "generated.ali"
            environment = os.environ.copy()
            environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
            subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "--params", str(params),
                    "--output", str(output),
                    "--module-mode", "never",
                ],
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            with mrcfile.open(output, permissive=True) as handle:
                self.assertEqual(tuple(handle.data.shape), (3, 80, 64))
                self.assertAlmostEqual(float(handle.voxel_size.x), 8.0, places=5)
            updated = json.loads(params.read_text())
            condition = updated["conditions"]["ali_identity"]
            self.assertEqual(condition["stack"], str(output.resolve()))
            self.assertEqual(condition["xf_file"], "IDENTITY")
            provenance = updated["generated_inputs"]["aligned_stack"]["provenance"]
            self.assertEqual(provenance["size_to_output_xy"], [64, 80])
            self.assertEqual(provenance["bin_by_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()
