# v5 to v6 migration

v6 requires `schema_version = 6` and does not silently accept v5 TOML files.
Use the explicit compatibility loader in `scripts/v6/config.py` when a v5
configuration must be inspected or migrated. The loader reports inferred
defaults such as stack-only source mode and the legacy affine backend.

The v5 affine converter remains available as the `legacy_affine` alignment
backend. It is the default until a WarpTools-native alignment import has
cluster validation proving equivalent geometry.

