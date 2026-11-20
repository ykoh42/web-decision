const baseInputs = Object.freeze([
  'inputs_embeds',
  'attention_mask',
  'position_ids',
  'option_indices',
  'temperatures',
  'n_slots'
]);
const prefixInputs = Object.freeze([...baseInputs, 'source_prefix']);

/** The input ABI and byte pins travel together when replacing the pending decoder. */
export const QWEN35_R19_TEXT = Object.freeze({
  manifest: {bytes: 2114, sha256: '6aa54189a087d199338ffe271e0469c9b719110ebac2363d637d2159419d7157'},
  fixture: {bytes: 202061, sha256: '7989413cb90ce9f60d01d25a8a96ad86a4ab74212e234d8f650e7ad3a891e7a9'},
  wasm: {bytes: 5772332, sha256: 'cbad94d32d5839215646e711be9adc8cdf98797beab663e1c2c90e929f3dade3'},
  parameterNamesSha256: '2bec9ebc149e5ef7389de06d8fea4615e30618f54ca6c226ed8c45fb509f487f',
  sourcePrefix: true as boolean,
  get inputNames(): readonly string[] {
    return this.sourcePrefix ? prefixInputs : baseInputs;
  }
});
