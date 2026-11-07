# Security policy

## Reporting

Please do not publish exploit details in an issue. Use GitHub's private vulnerability reporting for this repository when it is available; otherwise contact the maintainer through the GitHub profile.

## Preview-specific guidance

`web-decision@0.1.0-preview.1` is a browser-only preview for the Kev 0.6B Q4F16 route.

- Supply an application-configured `assetsUrl`; do not let user input choose the URL.
- For untrusted decision input, set `maxInputTokens` and `maxInputCharacters` and apply request-size and concurrency controls at the application boundary. The package has no default aggregate input cap so it can preserve the model's accepted input range.
- An abort signal can stop queued work. It cannot interrupt GPU inference after dispatch.
- The code package verifies pinned model asset bytes but does not include or host model weights.

See the [package security notes](releases/web-decision-0.1.0-preview.1/package/README.md#supported-scope) and [release scope](RELEASE_SCOPE.md) for the tested artifact and known limits.
