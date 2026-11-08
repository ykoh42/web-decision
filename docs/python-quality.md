# Python development tools

Python is used for model compilation and verification, not by the browser package.
Use Python 3.12 and uv 0.13.7 to install the locked development tools:

```sh
uv sync --locked --only-group quality
uv run --locked --no-sync --only-group quality python scripts/ci/check_python_complexity.py
uv run --locked --no-sync --only-group quality pytest scripts/ci/test_python_complexity.py
```

The complexity check measures every Git-tracked Python file and unignored new Python files, including release snapshots. The limit is 15. `complexipy-snapshot.json` records existing functions and module bodies above that limit; the check fails for a new violation or an increased recorded score. Existing violations remain visible in that file. The checker runs in a temporary directory so it cannot silently rewrite the reviewed baseline. Inline complexity ignores do not bypass the gate. Pull request and push checks compare the baseline with the previous commit and reject added entries or increased scores. Manual CI runs compare it with `main`.

`complexipy` implements an independent cognitive complexity metric; its scores need not exactly match Sonar's Python analyzer.

This development environment does not replace `requirements-preview-build.lock`, which remains the hash-locked environment for the TVM model build.
