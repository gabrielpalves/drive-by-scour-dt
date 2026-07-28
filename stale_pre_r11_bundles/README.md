# Quarantined pre-R11 dispatch bundles

The ZIP files and `bundle_sha256.txt` in this directory are the stale
2026-07-22 bundle set. The two `*_extracted` directories are older extracted
2026-07-19 bundles found in the repository root during the R11 audit. They are
retained locally only for recovery and audit history. None of these artifacts
may be dispatched or used to generate campaign data.

Build a new bundle set only after the R11 source commit has passed the full
qualification and the report-only authorization commit names that exact tested
source commit.
