import assert from 'node:assert/strict';
import test from 'node:test';
import {createQwen35PreEmbedSourceResolver} from '../src/strands/qwen35-pre-embed-sources.mts';
import type {Qwen35GraphInputChunk, Qwen35GraphInputRecord} from '../src/strands/qwen35-graph-input.mts';
import type {Qwen35VisionFeatureRunner} from '../src/strands/qwen35-vision-runner.mts';

const chunk = {} as unknown as Qwen35GraphInputChunk;

test('resolves source token embeddings for text without opening the vision graph', async () => {
  const controller = new AbortController();
  const calls: Array<{ids: Int32Array; signal: AbortSignal | null | undefined}> = [];
  const resolver = createQwen35PreEmbedSourceResolver({
    embeddings: {
      async lookupTokenEmbeddings(ids, options) {
        calls.push({ids: new Int32Array(ids), signal: options?.signal});
        return Float32Array.of(...ids);
      }
    },
    vision: null
  });

  const sources = await resolver(chunk, {imageCount: 0} as Qwen35GraphInputRecord, controller.signal);
  const ids = Int32Array.of(2025, 42);
  const rows = await sources.lookupTokenEmbeddings(ids);

  assert.deepEqual(rows, Float32Array.of(2025, 42));
  assert.deepEqual(calls, [{ids, signal: controller.signal}]);
  assert.deepEqual(sources.imageFeatures, []);
});

test('resolves source-order vision features and forwards inference cancellation', async () => {
  const controller = new AbortController();
  const input = {imageCount: 2} as Qwen35GraphInputRecord;
  const features = [Float32Array.of(1), Float32Array.of(2)];
  const visionCalls: Array<{input: Qwen35GraphInputRecord; signal: AbortSignal | null | undefined}> = [];
  const embeddingSignals: Array<AbortSignal | null | undefined> = [];
  const vision: Qwen35VisionFeatureRunner = {
    async run(value, signal) {
      visionCalls.push({input: value, signal});
      return features;
    }
  };
  const resolver = createQwen35PreEmbedSourceResolver({
    embeddings: {
      async lookupTokenEmbeddings(ids, options) {
        embeddingSignals.push(options?.signal);
        return new Float32Array(ids.length);
      }
    },
    vision,
    visionActivationPrecision: 'float32'
  });

  const sources = await resolver(chunk, input, controller.signal);
  await sources.lookupTokenEmbeddings(Int32Array.of(7), {signal: controller.signal});

  assert.deepEqual(sources.imageFeatures, features);
  assert.equal(sources.visionActivationPrecision, 'float32');
  assert.deepEqual(visionCalls, [{input, signal: controller.signal}]);
  assert.deepEqual(embeddingSignals, [controller.signal]);
});

test('shares vision features only for the same prepared request and clears failed work', async () => {
  let visionCalls = 0;
  const resolver = createQwen35PreEmbedSourceResolver({
    embeddings: {
      async lookupTokenEmbeddings(ids) {
        return new Float32Array(ids.length);
      }
    },
    vision: {
      async run() {
        visionCalls += 1;
        if (visionCalls === 1) throw new Error('vision request aborted');
        return [Float32Array.of(visionCalls)];
      }
    }
  });
  const firstInput = {imageCount: 1} as Qwen35GraphInputRecord;

  await assert.rejects(resolver(chunk, firstInput), /vision request aborted/u);
  const first = await resolver(chunk, firstInput);
  const second = await resolver(chunk, firstInput);

  assert.equal(visionCalls, 2, 'a failed request is retried, then reused for later graph chunks');
  assert.strictEqual(first.imageFeatures, second.imageFeatures);
  assert.deepEqual(first.imageFeatures, [Float32Array.of(2)]);

  const nextInput = {imageCount: 1} as Qwen35GraphInputRecord;
  const nextRequest = await resolver(chunk, nextInput);
  assert.equal(visionCalls, 3, 'a distinct prepared request performs fresh vision work');
  assert.notStrictEqual(nextRequest.imageFeatures, first.imageFeatures);
  assert.deepEqual(nextRequest.imageFeatures, [Float32Array.of(3)]);

  const controller = new AbortController();
  controller.abort(new Error('request cancelled before cached features are consumed'));
  await assert.rejects(
    resolver(chunk, nextInput, controller.signal),
    /request cancelled before cached features are consumed/u
  );
  assert.equal(visionCalls, 3, 'cancellation is checked even when features are already cached');
});

test('rejects image input without a vision runner and stops when already aborted', async () => {
  const resolver = createQwen35PreEmbedSourceResolver({
    embeddings: {
      async lookupTokenEmbeddings(ids) {
        return new Float32Array(ids.length);
      }
    },
    vision: null
  });
  const input = {imageCount: 1} as Qwen35GraphInputRecord;
  await assert.rejects(Promise.resolve(resolver(chunk, input)), /image input needs its pinned TVM vision runner/);

  const controller = new AbortController();
  controller.abort(new Error('cancelled before vision'));
  await assert.rejects(Promise.resolve(resolver(chunk, input, controller.signal)), /cancelled before vision/);
});
