export const TINYDECIDE_MODEL = 'experimental-tinydecide-10m-q4';
export const TINYDECIDE_REPOSITORY = 'TheREZOR/TinyDecide';
export const TINYDECIDE_REVISION = 'aa745b73a2db9a43bce731f0e8e5c90a6899fe2c';
export const TINYDECIDE_MAX_STATE_TOKENS = 127;
export const TINYDECIDE_MAX_QUESTION_TOKENS = 192;
export const TINYDECIDE_MAX_CHOICES = 32;
export const TINYDECIDE_QUESTIONS_PER_CHUNK = 8;

export const TINYDECIDE_FILES = Object.freeze({
  'meta.json': Object.freeze({
    bytes: 187362,
    sha256: '2fb895cfb0cdd6b11fcfb5573033e846bf4defadb3c78b9989cccfeefe06758f'
  }),
  'model.bin': Object.freeze({
    bytes: 6181984,
    sha256: '23fe9d93f2eafadbe4dc686a7dc5f309fbcba9ea1a4873d457e75d17c5bf3441'
  })
});

const TINYDECIDE_TOTAL_BYTES = Object.values(TINYDECIDE_FILES).reduce((sum, file) => sum + file.bytes, 0);
const TINYDECIDE_MANIFEST = Object.freeze({files: TINYDECIDE_FILES});
export const TINYDECIDE_SPEC = Object.freeze({
  files: TINYDECIDE_FILES,
  manifest: TINYDECIDE_MANIFEST,
  totalBytes: TINYDECIDE_TOTAL_BYTES
});
