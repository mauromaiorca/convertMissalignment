# Versioning

The installable distribution and the underlying scientific pipeline have separate version identifiers.

## Distribution version

The Python distribution version is defined in:

```text
convertMissalignment/_version.py
```

For release 0.1.11:

```python
__version__ = "0.1.11"
```

`pyproject.toml` reads this value dynamically. It controls:

- `convertMissalignment --version`
- `convertMissalignment version`
- `python -m pip show convertMissAlignment`
- wheel and source-distribution filenames
- installed package metadata

The top-level `VERSION` file also contains `0.1.11` for workflow provenance. Both values must be updated together for a future release.

## Pipeline version

The bundled workflow revision is defined by:

```text
PIPELINE_VERSION
```

and mirrored in:

```text
convertMissalignment/_version.py
```

For this release:

```text
8.0.0-alpha6-optional-smoke-run
```

Changing the package version does not imply a change to the scientific workflow. Conversely, a pipeline revision should be recorded even when command compatibility is retained.

## Release checklist

1. Update `convertMissalignment/_version.py`.
2. Update `VERSION`.
3. Update `PIPELINE_VERSION` and `__pipeline_version__` when the workflow changes.
4. Update `CHANGELOG.md`.
5. Run the portable tests.
6. Build and install the wheel in a clean environment.
7. Verify `convertMissalignment --version`, `where`, and `doctor`.
8. Regenerate `CODE_MANIFEST.sha256`.
9. Commit and create a matching Git tag, such as `v0.1.11`.
