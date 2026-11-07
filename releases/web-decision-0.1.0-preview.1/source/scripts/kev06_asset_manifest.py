"""Shared integrity pin for the prepared Kev 0.6B Q4F16 asset manifest."""
import hashlib
import json

KEV06_MANIFEST_SHA256 = 'bc79c92a7011d2aa1aff4ede66c06cc365de81e744507256afb5c0449870716a'


def canonical_manifest_sha256(value):
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n'
    return hashlib.sha256(canonical.encode()).hexdigest()
