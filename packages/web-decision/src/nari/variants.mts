export const NARI_MODEL_ID = 'experimental-nari-decision-4b-q4f16';
export const NARI_MIXED_MODEL_ID = 'experimental-nari-decision-4b-mixed16-q4f16';

export const NARI_FULL = Object.freeze({
  model: NARI_MODEL_ID,
  metadataBytes: 443924,
  metadataSha256: '22f901a6d2b9d4fc6a92b279a8af010f2f87ca9d8e48449f7d2f31252ed92877',
  format: 'nari-q4-toq-range-assets-v1',
  bindings: 730
});

export const NARI_MIXED = Object.freeze({
  model: NARI_MIXED_MODEL_ID,
  metadataBytes: 448593,
  metadataSha256: '95b96f584cf46df4c9723661f6a9cdd6d4dbc84da16fd6493198fabcf391a084',
  format: 'nari-mixed16-toq-range-assets-v1',
  bindings: 606
});

export function nariVariant(model: unknown) {
  if (model === NARI_FULL.model) return NARI_FULL;
  if (model === NARI_MIXED.model) return NARI_MIXED;
  return undefined;
}
