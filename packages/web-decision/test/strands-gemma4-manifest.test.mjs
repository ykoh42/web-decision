import assert from 'node:assert/strict';
import test from 'node:test';
import {parseGemma4ShardPlan} from '../src/strands/gemma4-manifest.mjs';
import {uploadGemma4Shard} from '../src/strands/gemma4-upload.mjs';

const sha = (character) => character.repeat(64);

function fixture() {
  const expected = {
    model: 'StrandsAgents/strands-decider-E2B-gemma4-v1-2610',
    checkpointRevision: 'checkpoint',
    baseModel: 'google/gemma-4-E2B-it',
    baseRevision: 'base',
    sourceBaseSha256: sha('a'),
    graphSha256: sha('b'),
    inputSpecsSha256: sha('c'),
    cacheIndexSha256: sha('d'),
    cacheIndexBytes: 100,
    sourceMapSha256: sha('e'),
    sourceMapBytes: 200
  };
  const inputSpecs = Array.from({length: 1248}, (_, index) => ({
    kind: index < 1207 ? 'InputKind.PARAMETER' : index === 1207 ? 'InputKind.CONSTANT_TENSOR' : 'InputKind.BUFFER',
    target: index === 1207 ? 'torso.embed_tokens.embed_scale' : `target_${index}`,
    arg: `argument_${index}`
  }));
  inputSpecs.push(
    ...['token_ids', 'answer_index', 'option_indices'].map((arg) => ({kind: 'InputKind.USER_INPUT', target: null, arg}))
  );
  const sourceMap = Array.from({length: 1248}, (_, index) => ({
    name: `param_${index}`,
    target: index === 1207 ? 'torso.embed_tokens.embed_scale' : `target_${index}`,
    shape: index === 1207 ? [] : [1],
    dtype: 'float16',
    nbytes: 2,
    sourceSha256: sha('f'),
    ...(index === 1207
      ? {
          sourceKind: 'derived',
          sourceTensor: 'main_embed_scale',
          sourceValue: 39.1875,
          sourceDtype: 'float16',
          conversion: 'none'
        }
      : {})
  }));
  const order = Array.from({length: 1248}, (_, index) => (index + 1) % 1248);
  const records = [order.slice(0, 624), order.slice(624)].map((indices, shard) => ({
    dataPath: `params_shard_${shard}.bin`,
    format: 'raw-shard',
    nbytes: indices.length * 2,
    records: indices.map((index, offset) => ({
      name: `param_${index}`,
      shape: index === 1207 ? [] : [1],
      dtype: 'float16',
      format: 'raw',
      nbytes: 2,
      byteOffset: offset * 2
    }))
  }));
  const cacheManifest = {
    schemaVersion: 1,
    model: expected.model,
    checkpointRevision: expected.checkpointRevision,
    baseModel: expected.baseModel,
    baseRevision: expected.baseRevision,
    sourceBaseSha256: expected.sourceBaseSha256,
    sourceWeightDtype: 'bfloat16',
    computeWeightDtype: 'float16',
    weightConversion: 'bfloat16-to-float16-lossy',
    derivedBufferConversion: 'float32-to-float16-lossy',
    tensorSourceMap: 'tensor-source-map.json',
    files: [
      ...records.map((record, index) => ({
        path: record.dataPath,
        bytes: record.nbytes,
        sha256: sha(index ? '1' : '2')
      })),
      {path: 'tensor-cache.json', bytes: expected.cacheIndexBytes, sha256: expected.cacheIndexSha256},
      {path: 'tensor-source-map.json', bytes: expected.sourceMapBytes, sha256: expected.sourceMapSha256}
    ]
  };
  const cacheIndex = {
    metadata: {
      model: expected.model,
      revision: expected.checkpointRevision,
      baseModel: expected.baseModel,
      baseRevision: expected.baseRevision,
      baseWeightsSha256: expected.sourceBaseSha256,
      wasmSha256: expected.graphSha256,
      inputSpecsSha256: expected.inputSpecsSha256,
      sourceWeightDtype: 'bfloat16',
      computeWeightDtype: 'float16',
      weightConversion: 'bfloat16-to-float16-lossy',
      derivedBufferConversion: 'float32-to-float16-lossy',
      format: 'raw'
    },
    records
  };
  return {inputSpecs, cacheManifest, cacheIndex, sourceMap, expected};
}

test('builds an immutable ABI-ordered plan for physically reordered shards', () => {
  const input = fixture();
  const plan = parseGemma4ShardPlan(input);
  assert.equal(plan.shards.length, 2);
  assert.equal(plan.expectedParameters.length, 1248);
  assert.equal(plan.shards[0].expectedParameters[0].index, 1);
  assert.equal(plan.shards[1].expectedParameters.at(-1).index, 0);
  assert.deepEqual(plan.expectedParameters[1207].shape, []);
  input.sourceMap[1].shape[0] = 8;
  input.cacheIndex.records[0].records[0].byteOffset = 99;
  assert.deepEqual(plan.expectedParameters[1].shape, [1]);
  assert.equal(plan.shards[0].records[0].byteOffset, 0);
  assert.throws(() => plan.shards[0].records[0].shape.push(5), TypeError);
});

test('pins the v26 graph argument inventory and embedded scalar constant index', () => {
  const {inputSpecs, sourceMap} = fixture();
  const graphArguments = inputSpecs.slice(0, 1248);
  assert.equal(graphArguments.filter((item) => item.kind === 'InputKind.PARAMETER').length, 1207);
  assert.equal(graphArguments.filter((item) => item.kind === 'InputKind.BUFFER').length, 40);
  assert.equal(graphArguments.filter((item) => item.kind === 'InputKind.CONSTANT_TENSOR').length, 1);
  assert.deepEqual(
    inputSpecs.slice(1248).map(({arg}) => arg),
    ['token_ids', 'answer_index', 'option_indices']
  );
  assert.equal(inputSpecs[1207].target, 'torso.embed_tokens.embed_scale');
  assert.deepEqual(sourceMap[1207], {
    name: 'param_1207',
    target: 'torso.embed_tokens.embed_scale',
    shape: [],
    dtype: 'float16',
    nbytes: 2,
    sourceSha256: sha('f'),
    sourceKind: 'derived',
    sourceTensor: 'main_embed_scale',
    sourceValue: 39.1875,
    sourceDtype: 'float16',
    conversion: 'none'
  });
});

test('plan shard records and expectations connect to the existing uploader', async () => {
  const shard = parseGemma4ShardPlan(fixture()).shards[0];
  let writes = 0,
    syncs = 0;
  const uploaded = await uploadGemma4Shard({
    bytes: new Uint8Array(shard.bytes),
    records: shard.records,
    expectedParameters: shard.expectedParameters,
    parameterCount: 1248,
    allocate: ({index, shape}) => ({
      tensor: {shape, byteOffset: 0, getDataPtr: () => index},
      capacityBytes: 4
    }),
    context: {copyRawBytesToBuffer: () => writes++},
    sync: async () => {
      syncs++;
    }
  });
  assert.equal(uploaded.size, 624);
  assert.equal(writes, 624);
  assert.equal(syncs, 1);
});

test('rejects mismatched identities, graph hashes and independently verified metadata hashes', () => {
  const changes = [
    (f) => {
      f.cacheManifest.model = 'other';
    },
    (f) => {
      f.cacheIndex.metadata.revision = 'other';
    },
    (f) => {
      f.cacheIndex.metadata.baseRevision = 'other';
    },
    (f) => {
      f.cacheIndex.metadata.baseWeightsSha256 = sha('0');
    },
    (f) => {
      f.expected.graphSha256 = sha('0');
    },
    (f) => {
      f.expected.inputSpecsSha256 = sha('0');
    },
    (f) => {
      f.expected.cacheIndexSha256 = sha('0');
    },
    (f) => {
      f.expected.sourceMapBytes++;
    },
    (f) => {
      f.cacheIndex.metadata.format = 'other';
    }
  ];
  for (const change of changes) {
    const input = fixture();
    change(input);
    assert.throws(() => parseGemma4ShardPlan(input));
  }
});

test('rejects wrong or incomplete graph, source map and parameter metadata', () => {
  const changes = [
    (f) => {
      f.inputSpecs.pop();
    },
    (f) => {
      f.inputSpecs[1248].arg = 'answer_index';
    },
    (f) => {
      f.inputSpecs[0].kind = 'InputKind.USER_INPUT';
    },
    (f) => {
      f.inputSpecs[0].target = f.inputSpecs[1].target;
    },
    (f) => {
      f.inputSpecs[0].arg = f.inputSpecs[1].arg;
    },
    (f) => {
      f.sourceMap[0].name = 'param_1';
    },
    (f) => {
      f.sourceMap[0].target = 'other';
    },
    (f) => {
      f.sourceMap[0].sourceSha256 = 'bad';
    },
    (f) => {
      f.sourceMap[1].shape = [0];
    },
    (f) => {
      f.sourceMap[1].shape = [Number.MAX_SAFE_INTEGER, 3];
    },
    (f) => {
      f.cacheIndex.records[0].records[0].dtype = 'float32';
    },
    (f) => {
      f.cacheIndex.records[0].records[0].shape = [];
    },
    (f) => {
      f.cacheIndex.records[0].records[1].name = 'param_1';
    },
    (f) => {
      f.cacheIndex.records[0].records.pop();
    }
  ];
  for (const change of changes) {
    const input = fixture();
    change(input);
    assert.throws(() => parseGemma4ShardPlan(input));
  }
});

test('rejects unsafe shard paths, sizes, hashes, overlap and missing bytes', () => {
  const changes = [
    (f) => {
      f.cacheManifest.files[0].path = '../params_shard_0.bin';
    },
    (f) => {
      f.cacheManifest.files[0].sha256 = 'bad';
    },
    (f) => {
      f.cacheManifest.files[1].sha256 = f.cacheManifest.files[0].sha256;
    },
    (f) => {
      f.cacheManifest.files[0].bytes = Number.MAX_SAFE_INTEGER + 1;
    },
    (f) => {
      f.cacheManifest.files.push({...f.cacheManifest.files[0]});
    },
    (f) => {
      f.cacheManifest.files.push({path: 'params_shard_2.bin', bytes: 2, sha256: sha('3')});
    },
    (f) => {
      f.cacheIndex.records[1].dataPath = f.cacheIndex.records[0].dataPath;
    },
    (f) => {
      f.cacheIndex.records[0].nbytes++;
    },
    (f) => {
      f.cacheIndex.records[0].records[1].byteOffset = 1;
    },
    (f) => {
      f.cacheIndex.records[0].records[1].byteOffset = 3;
    },
    (f) => {
      f.cacheIndex.records[0].records[0].byteOffset = Number.MAX_SAFE_INTEGER;
    }
  ];
  for (const change of changes) {
    const input = fixture();
    change(input);
    assert.throws(() => parseGemma4ShardPlan(input));
  }
});
