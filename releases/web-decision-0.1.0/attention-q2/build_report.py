from __future__ import annotations


def record_wasm_export_failure(report: dict, error: Exception) -> None:
    message = str(error)
    report.update(
        status="failed",
        wasmExportStatus="unavailable",
        wasmExportErrorType=type(error).__name__,
        wasmExportError=message,
        errorType=type(error).__name__,
        error=message,
    )


def raise_for_failed_build(report: dict) -> None:
    if report.get("status") == "failed":
        raise SystemExit(1)
