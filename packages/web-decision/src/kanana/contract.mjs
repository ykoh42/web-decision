export const KANANA_MODEL_ID = 'kanana-2-v-decide-3b-2611';
export const KANANA_REPOSITORY = 'kakaocorp/kanana-2-v-decide-3b-2611';
export const KANANA_REVISION = '1489062def4145ae3a0a9261fbac4f4d88a8c6d1';
export const KANANA_ASSETS_URL = `https://huggingface.co/${KANANA_REPOSITORY}/resolve/${KANANA_REVISION}/`;
export const KANANA_MODEL_MAX_LENGTH = 20_480;
export const KANANA_VOCAB_SIZE = 128_256;

export const KANANA_FILES = Object.freeze({
  'config.json': Object.freeze({
    bytes: 2057,
    sha256: '64f4a103accb8946bb7ded7dc5d200f9d6041df3eaa891c04124bb41fa06a69b'
  }),
  'tokenizer_config.json': Object.freeze({
    bytes: 334,
    sha256: '65ac2989bb33e0fe47b0ec6d80070eb8d2eed3d32271c8de52abb35c13095638'
  }),
  'tokenizer.json': Object.freeze({
    bytes: 10057457,
    sha256: '1c4be9ecf77c926456fb82d4cf07ff1218a91907f3408f44895d2b01e0f2b5ab'
  })
});

export const KANANA_TOTAL_BYTES = Object.values(KANANA_FILES).reduce((sum, file) => sum + file.bytes, 0);

export const KANANA_SOURCE_SPEC = Object.freeze({
  model: KANANA_MODEL_ID,
  repository: KANANA_REPOSITORY,
  revision: KANANA_REVISION,
  files: KANANA_FILES,
  manifest: Object.freeze({files: KANANA_FILES}),
  totalBytes: KANANA_TOTAL_BYTES,
  maxAssetBytes: Math.max(...Object.values(KANANA_FILES).map((file) => file.bytes)),
  maxLength: KANANA_MODEL_MAX_LENGTH,
  vocabSize: KANANA_VOCAB_SIZE
});
