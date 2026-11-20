import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {loadModel} from '../src/curated/adapters/strands-qwen35.ts';
import {kananaFloat} from '../src/curated/index.ts';
import type {ModelDescription} from '../src/curated/types.js';
import type {
  StrandsQwen35CandidateLoadOptions,
  StrandsQwen35CandidateModel,
  StrandsQwen35SystemOneRequest
} from '../src/curated/strands-qwen35-candidate-types.js';
import {createQwen35Session} from '../src/strands/qwen35-session.mts';
import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from '../src/strands/qwen35-graph-input.mts';
import {createQwen35PreEmbedSourceResolver} from '../src/strands/qwen35-pre-embed-sources.mts';
import {prepareQwen35R13GraphInputTuple} from '../src/strands/qwen35-r13-pre-embed-bridge.mts';
import type {Qwen35R13GraphInputTuple} from '../src/strands/qwen35-r13-pre-embed-bridge.mts';
import {loadPinnedQwen35Model} from '../src/strands/qwen35-model-loader.mts';
import {
  CACHE_NAME as candidateCacheName,
  clearModelCache as clearCandidateModelCache,
  describeModel as describeCandidateModel,
  loadDecisionModel as loadCandidateDecisionModel
} from './fixtures/qwen35-candidate-package-entry.mts';

const MODEL_ID = 'strands-decider-2b-qwen3.5-v1-2610';
const description = {modelId: MODEL_ID} as unknown as ModelDescription;

function tokenizerFixture() {
  const specialTokens = [
    ['<|vision_start|>', 248053],
    ['<|image_pad|>', 248056],
    ['<|vision_end|>', 248054],
    ['<|endoftext|>', 248044],
    ['<bos>', 248043]
  ] as const;
  const byteDecoder = Object.fromEntries(
    Array.from({length: 256}, (_, byte) => [String.fromCharCode(byte + 0x100), byte])
  );
  const addedTokens = new Set(specialTokens.map(([, id]) => id));
  return {
    decoder: {byte_decoder: byteDecoder},
    get_added_tokens_decoder: () => addedTokens,
    encode(text: string, {add_special_tokens}: {readonly add_special_tokens: boolean}) {
      const ids: number[] = [];
      const tokens: string[] = [];
      if (add_special_tokens) {
        ids.push(248043);
        tokens.push('<bos>');
      }
      let cursor = 0;
      while (cursor < text.length) {
        const special = specialTokens.find(([token]) => text.startsWith(token, cursor));
        if (special) {
          ids.push(special[1]);
          tokens.push(special[0]);
          cursor += special[0].length;
          continue;
        }
        let end = cursor + 1;
        while (end < text.length && !specialTokens.some(([token]) => text.startsWith(token, end))) end += 1;
        const bytes = new TextEncoder().encode(text.slice(cursor, end).normalize('NFC'));
        for (const byte of bytes) {
          ids.push(byte + 1);
          tokens.push(String.fromCharCode(byte + 0x100));
        }
        cursor = end;
      }
      return {ids, tokens};
    }
  };
}

test('candidate systemOne accepts the package float marker only inside structured content', async (t) => {
  const inputs: Qwen35GraphInputChunk[] = [];
  const candidate = await loadCandidateDecisionModel(
    {modelId: MODEL_ID},
    {
      async openCache() {
        return null;
      },
      async loadPinnedModel() {
        return createQwen35Session({
          modelName: MODEL_ID,
          tokenizer: tokenizerFixture(),
          runner: {
            async run(chunk) {
              inputs.push(chunk);
              return [[0.5, 0.5]];
            },
            async releaseCachedMemory() {},
            async dispose() {}
          }
        });
      }
    }
  );
  t.after(() => candidate.dispose());

  const one = kananaFloat(1);
  const question = {
    type: 'choice',
    instructions: {limit: kananaFloat(1e20)},
    criteria: {first: {cutoff: one}, second: {value: 1}}
  } as const;
  await candidate.systemOne({state: new Map([['values', [1, one, kananaFloat(-0)]]]), questions: {pick: question}});
  assert.equal(
    inputs[0]!.expandedStateText,
    '<state>\n{\n  "values": [\n    1,\n    1.0,\n    -0.0\n  ]\n}\n</state>\n'
  );
  assert.equal(
    inputs[0]!.questions[0]!.questionText,
    [
      '<question type="choice">',
      'Select exactly one option.',
      '{',
      '  "limit": 1e+20',
      '}',
      '<options>',
      '1. first — { "cutoff": 1.0 }',
      '2. second — { "value": 1 }',
      '</options>',
      '</question>',
      '<answer>'
    ].join('\n')
  );

  await assert.rejects(
    // @ts-expect-error A scalar state also fails runtime validation.
    candidate.systemOne({state: one, questions: {pick: question}}),
    /State must be text or JSON data/u
  );
  await assert.rejects(
    // @ts-expect-error A scalar instruction also fails runtime validation.
    candidate.systemOne({state: '', questions: {pick: {...question, instructions: one}}}),
    /Question pick instructions must be text or JSON data/u
  );
  await assert.rejects(
    // @ts-expect-error A scalar option description also fails runtime validation.
    candidate.systemOne({state: '', questions: {pick: {...question, criteria: {first: one, second: null}}}}),
    /Choice descriptions must be text or JSON data/u
  );
  assert.equal(inputs.length, 1);
});

test('candidate API clears only the library asset cache and handles unavailable CacheStorage', async () => {
  const previousCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  try {
    const deletedNames: string[] = [];
    Object.defineProperty(globalThis, 'caches', {
      configurable: true,
      value: {
        async delete(cacheName: string) {
          deletedNames.push(cacheName);
          return true;
        }
      }
    });

    assert.equal(await clearCandidateModelCache(), true);
    assert.deepEqual(deletedNames, [candidateCacheName]);

    Object.defineProperty(globalThis, 'caches', {configurable: true, value: undefined});
    assert.equal(await clearCandidateModelCache(), false);
  } finally {
    if (previousCaches) Object.defineProperty(globalThis, 'caches', previousCaches);
    else Reflect.deleteProperty(globalThis, 'caches');
  }
});

test('public candidate load decorates a frozen session through its mutable Qwen facade', async () => {
  const calls: unknown[][] = [];
  let disposeCount = 0;
  const session = Object.freeze({
    identity: 'frozen-qwen-session',
    async systemOne(this: {readonly identity: string}, request: unknown, options?: unknown) {
      calls.push([this.identity, request, options]);
      return 'forwarded';
    },
    async releaseCachedMemory(this: {readonly identity: string}) {
      calls.push([this.identity, 'release']);
    },
    async dispose(this: {readonly identity: string}) {
      assert.equal(this.identity, 'frozen-qwen-session');
      disposeCount++;
    }
  });
  const loaded = await loadCandidateDecisionModel(
    {modelId: MODEL_ID},
    {
      async openCache() {
        return null;
      },
      async loadPinnedModel() {
        return session as unknown as Awaited<ReturnType<typeof loadPinnedQwen35Model>>;
      }
    }
  );

  assert.ok(Object.isFrozen(session));
  assert.ok(Object.isExtensible(loaded));
  assert.equal(Object.isFrozen(loaded), false);
  assert.equal(loaded.modelId, MODEL_ID);
  assert.equal(loaded.description, describeCandidateModel(MODEL_ID));
  const request = {state: 'bound call', questions: {decision: {type: 'noul', instructions: 'Is it bound?'}}} as const;
  const options = {strictWindow: true};
  assert.equal(await loaded.systemOne(request, options), 'forwarded');
  await loaded.releaseCachedMemory();
  assert.deepEqual(calls, [
    ['frozen-qwen-session', request, options],
    ['frozen-qwen-session', 'release']
  ]);

  const firstDispose = loaded.dispose();
  const secondDispose = loaded.dispose();
  assert.equal(firstDispose, secondDispose);
  await Promise.all([firstDispose, secondDispose]);
  assert.equal(disposeCount, 1);
});

test('Qwen adapter disposes the pinned session if facade creation fails', async () => {
  let disposeCount = 0;
  const session = Object.freeze({
    get systemOne(): never {
      throw new Error('systemOne binding failed');
    },
    async releaseCachedMemory() {},
    async dispose() {
      disposeCount++;
    }
  });

  await assert.rejects(
    loadModel({cache: false}, description, {
      async openCache() {
        return null;
      },
      async loadPinnedModel() {
        return session as unknown as Awaited<ReturnType<typeof loadPinnedQwen35Model>>;
      }
    }),
    /systemOne binding failed/u
  );
  assert.equal(disposeCount, 1);
});

test('Qwen adapter exposes a mutable public facade around the pinned session', async () => {
  const controller = new AbortController();
  const progress = () => {};
  const systemOneCalls: Array<{readonly request: unknown; readonly options: unknown}> = [];
  let disposeCount = 0;
  const session = Object.freeze({
    async systemOne(request: unknown, options?: unknown) {
      systemOneCalls.push({request, options});
      return {
        model: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
        answers: {},
        usage: {input_tokens: 0, output_tokens: 0}
      };
    },
    async releaseCachedMemory() {},
    async dispose() {
      disposeCount++;
    }
  });
  let cacheEnabled: boolean | undefined;
  let cacheSignal: AbortSignal | null | undefined;
  let loaderOptions: Parameters<typeof loadPinnedQwen35Model>[0] | undefined;
  const dependencies = {
    async openCache(enabled: boolean | undefined, signal?: AbortSignal | null) {
      cacheEnabled = enabled;
      cacheSignal = signal;
      return null;
    },
    async loadPinnedModel(options: Parameters<typeof loadPinnedQwen35Model>[0]) {
      loaderOptions = options;
      return session as Awaited<ReturnType<typeof loadPinnedQwen35Model>>;
    }
  };
  const options = {
    cache: false,
    offline: false,
    signal: controller.signal,
    onProgress: progress
  } satisfies StrandsQwen35CandidateLoadOptions;

  const loaded = await loadModel(options, description, dependencies);
  const candidate: StrandsQwen35CandidateModel = loaded;

  assert.equal(cacheEnabled, false);
  assert.equal(cacheSignal, controller.signal);
  assert.deepEqual(loaderOptions, {
    cache: null,
    offline: false,
    signal: controller.signal,
    onProgress: progress
  });
  assert.equal(Object.hasOwn(loaderOptions!, 'assetsUrl'), false);
  assert.notEqual(loaded, session);
  assert.ok(Object.isExtensible(loaded));
  assert.equal(loaded.modelId, MODEL_ID);
  assert.equal(loaded.description, description);

  const request: StrandsQwen35SystemOneRequest = {
    state: 'Strict context window.',
    questions: {decision: {type: 'noul', instructions: 'Does the statement hold?'}}
  };
  const systemOneOptions = {strictWindow: true} as const;
  await candidate.systemOne(request, systemOneOptions);
  assert.deepEqual(systemOneCalls, [{request, options: systemOneOptions}]);
  await loaded.releaseCachedMemory();
  const firstDispose = loaded.dispose();
  const secondDispose = loaded.dispose();
  assert.equal(firstDispose, secondDispose);
  await Promise.all([firstDispose, secondDispose]);
  assert.equal(disposeCount, 1);
});

test('public Qwen systemOne renders custom text and structured Noul criteria in false/true slot order', async () => {
  const questionTexts: string[] = [];
  const runner = {
    async run(chunk: Qwen35GraphInputChunk) {
      questionTexts.push(...chunk.questions.map(({questionText}) => questionText));
      return chunk.questions.map(({name}) => (name === 'false-text-true-structured' ? [0.8, 0.2] : [0.3, 0.7]));
    },
    async releaseCachedMemory() {},
    async dispose() {}
  };
  const session = createQwen35Session({
    tokenizer: tokenizerFixture(),
    runner,
    modelName: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610'
  });
  const candidate = await loadModel({}, description, {
    async openCache() {
      return null;
    },
    async loadPinnedModel() {
      return session;
    }
  });

  const request: StrandsQwen35SystemOneRequest = {
    state: 'The record supports the claim.',
    questions: {
      'false-text-true-structured': {
        type: 'noul',
        instructions: 'Does the record support the claim?',
        criteria: {
          false: 'Reject when the record contradicts the claim.',
          true: {evidence: ['receipt', 'timestamp'], interpretation: 'direct support'}
        }
      },
      'false-structured-true-text': {
        type: 'noul',
        instructions: 'Does the claim hold?',
        criteria: {
          false: {reason: 'unsupported', signals: ['missing', 'contradicted']},
          true: 'Accept when the record supports the claim.'
        }
      }
    }
  };

  try {
    const result = await candidate.systemOne(request);
    assert.deepEqual(questionTexts, [
      '<question type="noul">\nDecide whether the statement is true of the state.\nDoes the record support the claim?\n<options>\n1. false — Reject when the record contradicts the claim.\n2. true — { "evidence": [ "receipt", "timestamp" ], "interpretation": "direct support" }\n</options>\n</question>\n<answer>',
      '<question type="noul">\nDecide whether the statement is true of the state.\nDoes the claim hold?\n<options>\n1. false — { "reason": "unsupported", "signals": [ "missing", "contradicted" ] }\n2. true — Accept when the record supports the claim.\n</options>\n</question>\n<answer>'
    ]);
    assert.equal(result.answers['false-text-true-structured']?.type, 'noul');
    assert.equal(result.answers['false-text-true-structured']?.noul, 0.2);
    assert.equal(result.answers['false-structured-true-text']?.type, 'noul');
    assert.equal(result.answers['false-structured-true-text']?.noul, 0.7);
  } finally {
    await candidate.dispose();
  }
});

test('candidate Qwen adapter carries ordered image features into the pointer graph ABI', async () => {
  const image = Buffer.from(
    await readFile(new URL('./fixtures/qwen35-image-input/single-pixel.png', import.meta.url))
  ).toString('base64');
  let visionCalls = 0;
  let visionFeatureRows: readonly Float32Array[] = [];
  let inputTuple: Qwen35R13GraphInputTuple | undefined;
  let graphChunk: Qwen35GraphInputChunk | undefined;
  let graphInput: Qwen35GraphInputRecord | undefined;
  const resolver = createQwen35PreEmbedSourceResolver({
    embeddings: {
      async lookupTokenEmbeddings(tokenIds) {
        assert.ok(
          tokenIds.every((tokenId) => tokenId !== 248056),
          'image placeholders use vision features'
        );
        return new Float32Array(tokenIds.length * 2048).fill(0.25);
      }
    },
    vision: {
      async run(input) {
        visionCalls++;
        assert.equal(input.imageCount, 2);
        assert.deepEqual(
          input.visionGeometry.map(({featureRows}) => featureRows),
          [64, 64]
        );
        visionFeatureRows = input.visionGeometry.map(({featureRows}, imageIndex) =>
          new Float32Array(featureRows * 2048).fill(imageIndex === 0 ? 0.125 : 0.375)
        );
        return visionFeatureRows;
      }
    }
  });
  const session = createQwen35Session({
    tokenizer: tokenizerFixture(),
    modelName: 'StrandsAgents/strands-decider-2B-qwen3.5-v1-2610',
    runner: {
      async run(chunk, input, signal) {
        graphChunk = chunk;
        graphInput = input;
        const sources = await resolver(chunk, input, signal);
        inputTuple = await prepareQwen35R13GraphInputTuple(chunk, input, sources, {signal});
        return chunk.questions.map(() => [0.2, 0.8]);
      },
      async releaseCachedMemory() {},
      async dispose() {}
    }
  });
  const candidate = await loadModel({}, description, {
    async openCache() {
      return null;
    },
    async loadPinnedModel() {
      return session;
    }
  });

  try {
    const result = await candidate.systemOne(
      {
        state: 'Compare both images in source order.',
        images: [image, image],
        questions: {decision: {type: 'noul', instructions: 'Does the second image match the first?'}}
      },
      {maxImages: 2}
    );

    assert.equal(visionCalls, 1, 'the prepared request is evaluated once before pointer graph dispatch');
    assert.equal(inputTuple?.length, 6);
    const [inputsEmbeds] = inputTuple!;
    assert.ok(graphChunk);
    assert.ok(graphInput);
    assert.deepEqual(
      graphInput.visionGeometry.map(({featureRows}) => featureRows),
      [64, 64]
    );
    assert.equal(inputsEmbeds.shape[0], 1);
    assert.equal(inputsEmbeds.shape[2], 2048);
    assert.equal(
      inputsEmbeds.values.length,
      inputsEmbeds.shape.reduce((total, dimension) => total * dimension, 1)
    );
    const imageStarts: number[] = [];
    const prefix = graphChunk.prefixInputIds;
    for (let index = 0; index < graphChunk.mmTokenTypeIds.length; index += 1)
      if (graphChunk.mmTokenTypeIds[index] === 1 && (index === 0 || graphChunk.mmTokenTypeIds[index - 1] === 0))
        imageStarts.push(index);
    assert.deepEqual(
      imageStarts.map((start) => prefix[start]),
      [248056, 248056]
    );
    assert.equal(visionFeatureRows.length, 2);
    for (const [imageIndex, value] of [0.125, 0.375].entries()) {
      const start = imageStarts[imageIndex]! * 2048;
      const end = start + visionFeatureRows[imageIndex]!.length;
      assert.ok(inputsEmbeds.values.subarray(start, end).every((entry) => entry === value));
    }
    assert.equal(result.answers.decision?.type, 'noul');
    if (result.answers.decision?.type === 'noul') assert.equal(result.answers.decision.noul, 0.8);
  } finally {
    await candidate.dispose();
  }
});

test('pending curated Qwen adapter rejects caller asset URLs and offline mode without caching', async () => {
  let cacheOpens = 0;
  const dependencies = {
    async openCache() {
      cacheOpens++;
      return null;
    },
    async loadPinnedModel() {
      throw new Error('unexpected pinned loader call');
    }
  };

  await assert.rejects(loadModel({assetsUrl: 'https://assets.example.test/qwen/'}, description, dependencies), {
    code: 'INVALID_INPUT'
  });
  await assert.rejects(loadModel({cache: false, offline: true}, description, dependencies), {code: 'INVALID_INPUT'});
  assert.equal(cacheOpens, 0);
});

test('pending Qwen adapter enables the origin cache by default and passes it to the pinned loader', async () => {
  const cache = {
    async match() {
      return null;
    },
    async put() {}
  } as unknown as Cache;
  let cacheOpen: {readonly enabled: boolean | undefined; readonly signal: AbortSignal | null | undefined} | undefined;
  let loaderOptions: Parameters<typeof loadPinnedQwen35Model>[0] | undefined;
  const session = {
    async systemOne() {
      return {model: MODEL_ID, answers: {}, usage: {input_tokens: 0, output_tokens: 0}};
    },
    async releaseCachedMemory() {},
    async dispose() {}
  } as Awaited<ReturnType<typeof loadPinnedQwen35Model>>;

  const loaded = await loadModel({}, description, {
    async openCache(enabled, signal) {
      cacheOpen = {enabled, signal};
      return cache;
    },
    async loadPinnedModel(options) {
      loaderOptions = options;
      return session;
    }
  });

  assert.deepEqual(cacheOpen, {enabled: true, signal: undefined});
  assert.deepEqual(loaderOptions, {cache, offline: false});
  assert.notEqual(loaded, session);
  assert.equal(loaded.modelId, MODEL_ID);
});

test('pending Qwen adapter preserves offline cache and abort options without an asset URL', async () => {
  const controller = new AbortController();
  const cache = {
    async match() {
      return null;
    },
    async put() {}
  } as unknown as Cache;
  let cacheOpen: {readonly enabled: boolean | undefined; readonly signal: AbortSignal | null | undefined} | undefined;
  let loaderOptions: Parameters<typeof loadPinnedQwen35Model>[0] | undefined;
  const session = {
    async systemOne() {
      return {model: MODEL_ID, answers: {}, usage: {input_tokens: 0, output_tokens: 0}};
    },
    async releaseCachedMemory() {},
    async dispose() {}
  } as Awaited<ReturnType<typeof loadPinnedQwen35Model>>;

  await loadModel({offline: true, signal: controller.signal}, description, {
    async openCache(enabled, signal) {
      cacheOpen = {enabled, signal};
      return cache;
    },
    async loadPinnedModel(options) {
      loaderOptions = options;
      return session;
    }
  });

  assert.deepEqual(cacheOpen, {enabled: true, signal: controller.signal});
  assert.deepEqual(loaderOptions, {cache, offline: true, signal: controller.signal});
  assert.equal(Object.hasOwn(loaderOptions!, 'assetsUrl'), false);
});
