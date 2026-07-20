# v7.0.0-alpha5: remove `--extra-binning` from public CLIs

The `--extra-binning` command-line option has been removed from the supported
entry points:

```text
setup_missalign_project.py
setup_warp_project.py
prepare_imod_to_warp.py
```

Passing the option now fails at argument parsing rather than appearing to change
the Warp reconstruction. Internal multiresolution and projection-binning support
is retained unchanged. The canonical internal setting remains:

```toml
[multiresolution]
extra_projection_binning = 1
```

Project setup writes the safe default value `1`. Internal APIs and TOML-driven
workflows may still use another validated value in future development. No new
binning operation is introduced, and existing project geometry, conversion and
reconstruction contracts are unchanged.
