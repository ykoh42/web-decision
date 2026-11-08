export const DECIMA_SMALL_MODEL = 'experimental-decima-small-int8';
const DECIMA_SMALL_REPOSITORY = 'amyrmahdy/decima-small';
const DECIMA_SMALL_REVISION = '480c4ea9e54f2946b947fb3ceb3c7cc7ece77522';
export const DECIMA_BASE_CLEAN_MODEL = 'experimental-decima-base-clean-int8';
const DECIMA_BASE_CLEAN_REPOSITORY = 'amyrmahdy/decima-base-clean';
const DECIMA_BASE_CLEAN_REVISION = 'a37e006fd4015c3c96176c341f3d6620f8a101f3';
export const DECIMA_AGENT_MODEL = 'experimental-decima-agent-int8';
const DECIMA_AGENT_REPOSITORY = 'amyrmahdy/decima-agent';
const DECIMA_AGENT_REVISION = '1acd2c4b1461c15ebc6b2b8f84dd9f8af18be245';
export const DECIMA_SMALL_MAX_STATE_TOKENS = 512;
const DECIMA_AGENT_MAX_STATE_TOKENS = 4096;
const DECIMA_AGENT_SOURCE_MAX_STATE_TOKENS = 8192;
export const DECIMA_SMALL_MAX_CHOICE_TOKENS = 64;
export const DECIMA_SMALL_TEMPERATURE = 0.9355568358032639;
export const DECIMA_AGENT_TEMPERATURE = 0.6124814821077146;

const smallFiles = Object.freeze({
  'encoder.onnx': Object.freeze({
    bytes: 122164246,
    sha256: '12fb5ae2b875893adf43e91ed1901c492360cf58107ea14f5e5080519db8e09f'
  }),
  'scorer.onnx': Object.freeze({
    bytes: 4915472,
    sha256: 'b37f6a8540647a2b62a1f72ebbf358683162c8483a4839e7b0a10d45238961d1'
  }),
  'decima.json': Object.freeze({
    bytes: 20151,
    sha256: '7cf3a87b7c9bebeff3834831b23ed2ff33f1834dc67d59602b61fbfe748ee25d'
  }),
  'tokenizer/tokenizer.json': Object.freeze({
    bytes: 17082987,
    sha256: '255f5e32cb320b4dc2b31e0f203832fecafe87ca4cc87c3c41eb8ccae73f441a'
  }),
  'tokenizer/tokenizer_config.json': Object.freeze({
    bytes: 596,
    sha256: 'b1aa8a3f4cecb46b952811c1c55c983348b88abc9c854743fa2fa47264a3a4ad'
  })
});

const baseCleanFiles = Object.freeze({
  'encoder.onnx': Object.freeze({
    bytes: 322447126,
    sha256: '69d699ea1387346cb248dca90ed5f7f1e3818a49d1dcc13a58fcdabdb0a4e498'
  }),
  'scorer.onnx': Object.freeze({
    bytes: 14436752,
    sha256: 'e2bdf1e8104ff15e329e622b8ef91e3a1a91340e67fda1c4ccce88d21404cf99'
  }),
  'decima.json': Object.freeze({
    bytes: 40198,
    sha256: 'f0c0210d6f9fb4d93ef1b428e2792d891b5a6720caaa0c765912dcdd437724b2'
  }),
  'tokenizer/tokenizer.json': Object.freeze({
    bytes: 34363440,
    sha256: 'd941237f7b665542273cd22102ae14fdacf54d3aaf0bd45da55022c46b4cc006'
  }),
  'tokenizer/tokenizer_config.json': Object.freeze({
    bytes: 735,
    sha256: '1701c4a91d2c2001df059773bbe4294218be0bba7ac8be04359a440ba0975df8'
  })
});

const agentFiles = Object.freeze({
  'encoder.onnx': Object.freeze({
    bytes: 322447126,
    sha256: 'fc4996f13640f3b43f27c073e4ac66ff58b544ae4599b425866ab2a66b647435'
  }),
  'scorer.onnx': Object.freeze({
    bytes: 14436752,
    sha256: 'a5832dd643cfa55c5750cf2fd21590232cbef5ac20b95caf51cc99a354dafc71'
  }),
  'decima.json': Object.freeze({
    bytes: 40245,
    sha256: '9569b6a41b85e68509ef8635b8de51099877cfd0d06810e2c5f9a8095d376ef0'
  }),
  'tokenizer/tokenizer.json': Object.freeze({
    bytes: 34363440,
    sha256: 'd941237f7b665542273cd22102ae14fdacf54d3aaf0bd45da55022c46b4cc006'
  }),
  'tokenizer/tokenizer_config.json': Object.freeze({
    bytes: 735,
    sha256: '1701c4a91d2c2001df059773bbe4294218be0bba7ac8be04359a440ba0975df8'
  })
});

function modelSpec({
  model,
  repository,
  revision,
  files,
  temperature,
  stateOverflow,
  choiceFit,
  maxStateTokens = DECIMA_SMALL_MAX_STATE_TOKENS,
  sourceMaxStateTokens = maxStateTokens,
  typedQuestionTypes = ['choice', 'score', 'noul', 'rank'],
  cpuWasmVerification = 'verified'
}) {
  const manifest = Object.freeze({files});
  return Object.freeze({
    model,
    repository,
    revision,
    files,
    manifest,
    maxAssetBytes: Math.max(...Object.values(files).map((file) => file.bytes)),
    totalBytes: Object.values(files).reduce((sum, file) => sum + file.bytes, 0),
    maxStateTokens,
    sourceMaxStateTokens,
    maxChoiceTokens: DECIMA_SMALL_MAX_CHOICE_TOKENS,
    temperature,
    stateOverflow,
    choiceFit,
    typedQuestionTypes: Object.freeze([...typedQuestionTypes]),
    cpuWasmVerification
  });
}

export const DECIMA_SMALL_SPEC = modelSpec({
  model: DECIMA_SMALL_MODEL,
  repository: DECIMA_SMALL_REPOSITORY,
  revision: DECIMA_SMALL_REVISION,
  files: smallFiles,
  temperature: DECIMA_SMALL_TEMPERATURE,
  stateOverflow: 'truncate',
  choiceFit: false
});

export const DECIMA_BASE_CLEAN_SPEC = modelSpec({
  model: DECIMA_BASE_CLEAN_MODEL,
  repository: DECIMA_BASE_CLEAN_REPOSITORY,
  revision: DECIMA_BASE_CLEAN_REVISION,
  files: baseCleanFiles,
  temperature: 0.6929793512269696,
  stateOverflow: 'reject',
  choiceFit: true
});

export const DECIMA_AGENT_SPEC = modelSpec({
  model: DECIMA_AGENT_MODEL,
  repository: DECIMA_AGENT_REPOSITORY,
  revision: DECIMA_AGENT_REVISION,
  files: agentFiles,
  temperature: DECIMA_AGENT_TEMPERATURE,
  stateOverflow: 'reject',
  choiceFit: true,
  maxStateTokens: DECIMA_AGENT_MAX_STATE_TOKENS,
  sourceMaxStateTokens: DECIMA_AGENT_SOURCE_MAX_STATE_TOKENS,
  typedQuestionTypes: ['choice', 'score', 'noul'],
  cpuWasmVerification: 'verified'
});

export const DECIMA_MODELS = Object.freeze({
  [DECIMA_SMALL_MODEL]: DECIMA_SMALL_SPEC,
  [DECIMA_BASE_CLEAN_MODEL]: DECIMA_BASE_CLEAN_SPEC,
  [DECIMA_AGENT_MODEL]: DECIMA_AGENT_SPEC
});
