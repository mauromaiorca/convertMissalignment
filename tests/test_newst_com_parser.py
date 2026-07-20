from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_aligned_stack import prepare_newst_standard_input


class NewstComParserTests(unittest.TestCase):
    def test_preserves_geometry_and_rewrites_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "newst.com"
            raw = root / "raw input.mrc"
            xf = root / "final.xf"
            output = root / "generated.ali"
            source.write_text(
                "$newstack -StandardInput\n"
                "InputFile old.st\n"
                "OutputFile old.ali\n"
                "TransformFile old.xf\n"
                "SizeToOutput 512,720\n"
                "BinByFactor 2\n"
                "ModeToOutput 2\n"
                "$if (-e ./savework) ./savework\n"
            )
            result = prepare_newst_standard_input(source, raw, xf, output)
            self.assertIsNotNone(result)
            generated, provenance = result
            text = generated.read_text()
            self.assertIn(f"InputFile {raw}", text)
            self.assertIn(f"OutputFile {output}", text)
            self.assertIn(f"TransformFile {xf}", text)
            self.assertIn("SizeToOutput 512,720", text)
            self.assertIn("BinByFactor 2", text)
            self.assertEqual(provenance["size_to_output_xy"], [512, 720])
            self.assertEqual(provenance["bin_by_factor"], 2.0)

    def test_returns_none_without_standard_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "newst.com"
            source.write_text("newstack input.st output.ali\n")
            self.assertIsNone(
                prepare_newst_standard_input(
                    source, Path(tmp) / "raw.mrc", Path(tmp) / "a.xf", Path(tmp) / "out.ali"
                )
            )


if __name__ == "__main__":
    unittest.main()
