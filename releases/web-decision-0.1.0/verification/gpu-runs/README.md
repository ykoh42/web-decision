# Real-model WebGPU run artifacts

These logs cover the exact `web-decision@0.1.0` archive and bundle identified in `../real-model-gpu.json`. They were produced on one Apple M4 Pro with Chromium, Firefox, and WebKit using non-fallback WebGPU adapters with `shader-f16`.

Each browser directory contains the passing four-case semantic coverage log, seven independently run long-input stage logs and structured runtime event streams. Firefox and WebKit also include periodic macOS `memory_pressure` samples. Chromium includes the first combined diagnostic attempt, which timed out after 360 seconds while placing all seven inference stages in one Playwright test; all seven stages then passed when run individually. The timeout was in the diagnostic harness invocation, not a reported inference failure.

The structured stage streams contain token counts, execution events, validation outcomes, resource counters, and selected fixture outputs. They do not contain prompt text or local absolute paths. Each browser manifest lists the byte size and SHA-256 of every stored log and binds those files to the archive and bundle hashes.
