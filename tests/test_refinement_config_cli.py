"""Tests for refinement-config parsing/validation and the forward/validate CLIs."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch  # refinement_config imports model classes (torch-based)
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

if HAVE_TORCH:
    from alignment_models.refinement_config import from_toml_dict, RefinementConfig


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class RefinementConfigTests(unittest.TestCase):
    def test_defaults_and_automatic_schedule(self):
        cfg = from_toml_dict({"model": "affine", "schedule": "automatic"})
        self.assertEqual(cfg.model, "affine")
        stages = [s.model for s in cfg.resolved_stages()]
        self.assertEqual(stages, ["translation", "translation", "rigid", "similarity", "affine"])

    def test_invalid_model_rejected(self):
        with self.assertRaises(ValueError):
            from_toml_dict({"model": "projective"})

    def test_explicit_stage_exceeding_max_rejected(self):
        with self.assertRaises(ValueError):
            from_toml_dict({
                "model": "rigid", "schedule": "explicit",
                "stages": [{"model": "affine"}],  # affine > rigid
            })

    def test_explicit_requires_stages(self):
        with self.assertRaises(ValueError):
            from_toml_dict({"model": "rigid", "schedule": "explicit", "stages": []})

    def test_scope_model_incompat_warns(self):
        cfg = from_toml_dict({
            "model": "translation",
            "parameter_scope": {"rotation": "global"},  # translation has no rotation
        })
        self.assertTrue(any("rotation" in w for w in cfg.warnings))

    def test_cli_overrides(self):
        cfg = from_toml_dict({"model": "translation"}, {"model": "rigid", "rotation_scope": "global"})
        self.assertEqual(cfg.model, "rigid")
        self.assertEqual(cfg.scope.rotation, "global")


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class CliSmokeTests(unittest.TestCase):
    def test_prepare_validate_only(self):
        example = ROOT / "config" / "examples" / "affine_refinement.toml"
        cp = subprocess.run(
            [sys.executable, str(ROOT / "prepare_imod_to_warp.py"), str(example), "--validate-only"],
            text=True, capture_output=True,
        )
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
        self.assertIn("refinement model", cp.stdout)

    def test_validate_interoperability_math_level(self):
        cp = subprocess.run(
            [sys.executable, str(ROOT / "validate_interoperability.py"), "--level", "math"],
            text=True, capture_output=True,
        )
        # math level must pass on this machine (torch present)
        self.assertEqual(cp.returncode, 0, cp.stdout[-2000:] + cp.stderr[-2000:])
        self.assertIn("PASS", cp.stdout)


if __name__ == "__main__":
    unittest.main()
