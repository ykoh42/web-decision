# Security policy

## Reporting

Please do not publish exploit details in an issue. Use GitHub's private vulnerability reporting for this repository when it is available; otherwise contact the maintainer through the GitHub profile.

## Browser runtime

- The curated API accepts model IDs and catalog-pinned asset sources; it does not accept a caller-supplied `assetsUrl`.
- Decision input stays in browser-side tokenization and inference. Caller-supplied image URLs are fetched by the browser from their own hosts.
- Apply request-size and concurrency controls at the application boundary when accepting untrusted input. Keep the model's pinned input limits unchanged.
- An abort signal can stop queued work. It cannot preempt inference already dispatched to the runtime.

See the [package README](packages/web-decision/README.md) for setup and supported models.
