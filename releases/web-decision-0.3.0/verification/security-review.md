# Security review for `web-decision@0.3.0`

The Codex Security diff scan (ID `178eff7e-5bce-499c-82c2-729122c08c5b`) completed with no vulnerability candidates. It reviewed all 61 items produced by the changed-file inventory, including the package runtime, TAMEV asset handling, model contracts, and release build and archive tools.

A supplemental review checked the 12 TVM source overlays under `source/overlay/.toolchains/apache-tvm-src/web/src/`. They are byte-identical to the corresponding `0.1.0` overlays. The candidate build verifies locked source hashes, and the active Kev loader verifies model assets before passing the WASM bytes to TVM.

## Scope limits

- No consuming application deployment was supplied. The application must keep `assetsUrl` in trusted configuration and set appropriate request-size and concurrency limits for untrusted input.
- Real-model browser checks ran on one Apple M4 Pro. They do not establish compatibility across devices or decision quality.
- The package excludes model weights. This review does not establish model training-data provenance or broader weight redistribution rights.
- The GitHub workflow now selects a release version from the tag or manual input, validates it against supported versions, and runs that version's test, build, pack, and dependency-audit scripts. This workflow change was made after the completed source scan and was reviewed separately.
