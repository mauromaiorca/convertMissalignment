# Publishing the repository on GitHub

## Initialise the repository

From the extracted `convertMissalignment` directory:

```bash
git init -b main
git add .
git commit -m "Release convertMissAlignment 0.1.14"
git tag -a v0.1.14 -m "convertMissAlignment 0.1.14"
```

## Create a private GitHub repository with GitHub CLI

```bash
gh auth login
gh repo create convertMissAlignment \
  --private \
  --source=. \
  --remote=origin \
  --push
git push origin v0.1.14
```

## Use an existing remote repository

```bash
git remote add origin <repository-url>
git push -u origin main
git push origin v0.1.14
```

## Recommended initial visibility

Use a private repository until the following items have been reviewed:

- cluster-specific configuration and module names
- test fixtures and dataset-derived metadata
- bundled MissAlignment integration and patch material
- third-party redistribution permissions
- the repository licence

This release does not include an open-source licence. Add a licence only after confirming that all included code and integration material can be redistributed under that licence.

## Release assets

The source repository is sufficient for installation with `pip install -e .` or `pip install .`. A wheel can be generated for a GitHub release with:

```bash
python -m pip install build
python -m build
```

The expected wheel filename is:

```text
convertmissalignment-0.1.14-py3-none-any.whl
```
