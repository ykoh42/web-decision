# web-decision 0.2.0 security review

Date: 2026-11-07

## Result

No package vulnerability was validated in the 0.2.0 candidate. The review covered the 119-file candidate inventory, package build and allowlist, Kev 0.6B and TAMEV Nano public loading paths, pinned assets, browser cache, runtime lifecycle, and published dependency wiring. Independent reviewers checked the architecture, full candidate baseline, TAMEV input handling, and asset handling.

The Codex Security scan is `320ec247-59a2-49d4-bed4-240317661cfb`. It recorded zero findings. Its sealed coverage document is marked `partial` because the review receipts were stored outside the scan directory and the workbench skipped those receipt references. The source reviews themselves were completed; this limitation is recorded rather than presented as complete tool coverage.

## Reviewed controls

- The package archive is 102,265 bytes, contains 15 allowlisted package files, and excludes model weights.
- Kev and TAMEV model assets use pinned hashes and exact sizes. Network and cache reads are bounded and verified before runtime use.
- Asset paths are checked against the configured directory. Fetches follow redirects by default; pinned content prevents model substitution, and asset requests contain no decision fields. Redirect confinement remains a low-confidence hardening opportunity.
- The npm archive includes the TAMEV license and required model/runtime notices. Runtime dependencies are exact-version pinned. Prior exact-archive checks passed `publint`, ESM NodeNext/Bundler `attw`, strict consumer TypeScript checks, and `npm audit --omit=dev` with zero vulnerabilities.
- No private keys or common credential patterns were found in the candidate inventory.

## Integration constraints

TAMEV intentionally has no default aggregate request, question, or option cap. It tokenizes complete text and recursively renders typed input before applying the model's 128-token context and 48-token option row limits. A consumer that passes untrusted values must bound text size, question/option counts, and concurrent requests before calling the model. These are caller and device-resource constraints; the package contains no server-side input endpoint.

Both model routes fetch only model assets from the caller-configured `assetsUrl`; decision input is processed locally. The consuming application must keep `assetsUrl` in trusted configuration. Abort signals do not interrupt synchronous tokenization or runtime inference already dispatched.

## Runtime evidence

The exact candidate package had real-model browser evidence for Kev 0.6B TVM/WebGPU and TAMEV Nano ONNX/WASM in Chromium, Firefox, and WebKit on the M4 Pro. Those tests cover the package/runtime paths; they do not establish behavior for every browser, device, or consuming application.
