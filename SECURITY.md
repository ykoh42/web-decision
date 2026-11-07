# Security policy

## Reporting

Please do not publish exploit details in an issue. Use GitHub's private vulnerability reporting for this repository when it is available; otherwise contact the maintainer through the GitHub profile.

## Browser runtime

- Keep `assetsUrl` in trusted application configuration. The loader fetches model files from that URL and verifies their pinned hashes.
- Decision input stays in browser-side tokenization and inference. It is not sent to the model asset host.
- Apply request-size and concurrency controls at the application boundary when accepting untrusted input.
- An abort signal can stop queued work. It cannot preempt inference already dispatched to the runtime.

See the [model setup guide](releases/web-decision-0.3.0/package/README.md) and [0.3.0 verification record](releases/web-decision-0.3.0/verification/package-archive.json) for asset requirements and tested scope.
