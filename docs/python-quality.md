# Python development tools

Python is used for model compilation and verification, not by the browser package.
Use Python 3.12 and uv 0.13.7 to install the locked development tools:

```sh
uv sync --locked --only-group quality
uv run --locked --no-sync --only-group quality python scripts/ci/check_python_complexity.py
uv run --locked --no-sync --only-group quality python scripts/ci/check_python_quality.py
uv pip install --require-hashes --only-binary=:all: --python .venv/bin/python -r requirements-preview-build.lock
uv run --locked --no-sync --only-group quality pytest -q \
  scripts/ci/test_python_complexity.py \
  scripts/ci/test_python_quality.py \
  scripts/test_safe_output.py \
  scripts/test_specialize_q4_direct_serialization.py \
  scripts/test_tvm_kev_shape_order.py
```

The complexity check measures every Git-tracked Python file and unignored new Python files, including release snapshots. The limit is 15. `complexipy-snapshot.json` records existing functions and module bodies above that limit; the check fails for a new violation or an increased recorded score. Existing violations remain visible in that file. The checker runs in a temporary directory so it cannot silently rewrite the reviewed baseline. Inline complexity ignores do not bypass the gate. Pull request and push checks compare the baseline with the previous commit and reject added entries or increased scores. Manual CI runs compare it with `main`.

`complexipy` implements an independent cognitive complexity metric; its scores need not exactly match Sonar's Python analyzer.

Ruff checks and formats every Git-tracked Python file. Its runner isolates Ruff from project and user configuration, disables Git-ignore and forced exclusions, and explicitly sets Python 3.12, line length 88, and lint rules E4/E7/E9/F/I. Existing findings in release snapshots and release-source-locked files are listed in `ruff-snapshot.json` with their exact file SHA-256 and diagnostic fingerprint; the gate checks those fingerprints even if a later configuration change would suppress a diagnostic, and rejects a changed file, a new finding, or a baseline addition relative to the target commit. The lint command ignores inline `noqa` comments so they cannot hide findings. New or changed Python files must pass Ruff without baseline entries.

The workflow installs the quality tools into `.venv`, then installs TVM/ONNX wheels into that same test environment from `requirements-preview-build.lock` with `--require-hashes --only-binary=:all:` before invoking pytest. The Python test run includes the `unittest` suites for safe output, ONNX serialization, and Kev shape ordering. Those tests import ONNX and TVM directly during collection; missing dependencies fail the job instead of turning the tests into skips. The workflow does not regenerate or replace the preview requirements lock.

This development environment does not replace `requirements-preview-build.lock`, which remains the hash-locked environment for the TVM model build.
