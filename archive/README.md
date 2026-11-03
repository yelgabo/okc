# Historical records

Records dated before November 2, 2026 live under `archive/docs/`, preserving their former directory layout. Current reader dependencies and operating policies remain under `docs/` even when older.

[Move manifest](manifest-2026-11-02.json) records original paths, archive paths, hashes and document-link repairs. Raw receipts, logs and source payloads retain their bytes. Historical scripts may assume their original paths; replay them at the recorded Git revision or restore their original layout first.

Default ripgrep searches exclude this directory. Search it explicitly with `rg --files --no-ignore archive` or inspect it through `git ls-files archive`. The current queue remains in [shared steering](../docs/steering/README.md).
