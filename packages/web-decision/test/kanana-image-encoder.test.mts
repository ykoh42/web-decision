import assert from 'node:assert/strict';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';

const outputDirectory = process.env.KANANA_CORE_DIST;
if (!outputDirectory) throw new Error('Compile the Kanana image encoder before running its focused tests.');
const {createKananaTvmImageEmbeddingEncoder} = await import(
  pathToFileURL(join(outputDirectory, 'image-encoder.js')).href
);

test('encodes image patches once in prompt order and concatenates merged embeddings', async () => {
  const decodedSources: unknown[] = [];
  const visionGrids: number[][] = [];
  const encoder = createKananaTvmImageEmbeddingEncoder({
    async decode(source: unknown) {
      decodedSources.push(source);
      return {rgb: new Uint8Array([source === 'first' ? 1 : 2]), width: source === 'first' ? 1 : 2, height: 1};
    },
    preprocess(_rgb: Uint8Array, width: number) {
      return {
        imageTokenThw: width === 1 ? [1, 1, 1] : [1, 1, 2],
        pixelValues: Float32Array.from([width]),
        visionGridThw: width === 1 ? [1, 2, 2] : [1, 2, 4]
      };
    },
    vision: {
      async infer(_pixels: ArrayLike<number>, grid: readonly [number, number, number]) {
        visionGrids.push([...grid]);
        const tokens = (grid[1] * grid[2]) / 4;
        const value = grid[2] === 2 ? 0.25 : 0.75;
        return {
          shape: [tokens, 2_560],
          dtype: 'float32' as const,
          values: new Float32Array(tokens * 2_560).fill(value)
        };
      }
    }
  });

  const embeddings = await encoder(
    ['first', 'second'],
    [
      [1, 1, 1],
      [1, 1, 2]
    ]
  );

  assert.deepEqual(decodedSources, ['first', 'second']);
  assert.deepEqual(visionGrids, [
    [1, 2, 2],
    [1, 2, 4]
  ]);
  assert.deepEqual(embeddings.shape, [3, 2_560]);
  assert.equal(embeddings.values[0], 0.25);
  assert.equal(embeddings.values[2_560], 0.75);
});

test('rejects image grid and vision output mismatches without changing prompt order', async () => {
  const encoder = createKananaTvmImageEmbeddingEncoder({
    decode: async (_source: unknown) => ({rgb: new Uint8Array(3), width: 1, height: 1}),
    preprocess: (_rgb: Uint8Array, _width: number, _height: number) => ({
      imageTokenThw: [1, 1, 2],
      pixelValues: new Float32Array(1),
      visionGridThw: [1, 2, 2]
    }),
    vision: {
      async infer(_pixels: ArrayLike<number>, _grid: readonly [number, number, number]) {
        return {shape: [1, 2_560], dtype: 'float32' as const, values: new Float32Array(2_560)};
      }
    }
  });

  await assert.rejects(encoder(['first'], [[1, 1, 1]]), /prompt token grid/);
});

test('forwards distinct AbortSignals to concurrent vision inference calls', async () => {
  const firstController = new AbortController();
  const secondController = new AbortController();
  const signals: AbortSignal[] = [];
  const resolutions = new Map<
    AbortSignal,
    (result: {shape: readonly number[]; dtype: 'float32'; values: Float32Array}) => void
  >();
  let resolveBothStarted!: () => void;
  const bothStarted = new Promise<void>((resolve) => {
    resolveBothStarted = resolve;
  });
  let decoderCount = 0;
  let resolveDecoders!: () => void;
  const bothDecoded = new Promise<void>((resolve) => {
    resolveDecoders = resolve;
  });

  const encoder = createKananaTvmImageEmbeddingEncoder({
    async decode(source: unknown) {
      decoderCount++;
      if (decoderCount === 2) resolveDecoders();
      await bothDecoded;
      return {rgb: new Uint8Array([source === 'first' ? 1 : 2]), width: 1, height: 1};
    },
    preprocess(rgb: Uint8Array) {
      return {
        imageTokenThw: [1, 1, 1],
        pixelValues: Float32Array.from([rgb[0]!]),
        visionGridThw: [1, 2, 2]
      };
    },
    vision: {
      infer(_pixels: ArrayLike<number>, _grid: readonly [number, number, number], signal?: AbortSignal) {
        if (!signal) throw new Error('vision inference did not receive a request signal');
        signals.push(signal);
        const pending = new Promise<{shape: readonly number[]; dtype: 'float32'; values: Float32Array}>((resolve) =>
          resolutions.set(signal, resolve)
        );
        if (signals.length === 2) resolveBothStarted();
        return pending;
      }
    }
  });

  const first = encoder(['first'], [[1, 1, 1]], firstController.signal);
  const second = encoder(['second'], [[1, 1, 1]], secondController.signal);
  await bothStarted;

  assert.deepEqual(signals, [firstController.signal, secondController.signal]);
  firstController.abort(new Error('first request cancelled'));
  for (const resolve of resolutions.values())
    resolve({shape: [1, 2_560], dtype: 'float32', values: new Float32Array(2_560)});

  await assert.rejects(first, /first request cancelled/);
  await assert.doesNotReject(second);
});

test('aborts before starting the next image and rejects unsupported aggregate token counts', async () => {
  const controller = new AbortController();
  const encoder = createKananaTvmImageEmbeddingEncoder({
    async decode(_source: unknown, _signal?: AbortSignal) {
      controller.abort(new Error('cancelled'));
      return {rgb: new Uint8Array(3), width: 1, height: 1};
    },
    preprocess: (_rgb: Uint8Array, _width: number, _height: number) => ({
      imageTokenThw: [1, 1, 1],
      pixelValues: new Float32Array(1),
      visionGridThw: [1, 2, 2]
    }),
    vision: {
      async infer(_pixels: ArrayLike<number>, _grid: readonly [number, number, number]) {
        throw new Error('vision should not start after abort');
      }
    }
  });

  await assert.rejects(encoder(['first'], [[1, 1, 1]], controller.signal), /cancelled/);
  await assert.rejects(encoder(['first'], [[1, 20_481, 1]]), /model sequence limit/);
});
