import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {TINYDECIDE_FILES} from '../src/tinydecide/contract.mjs';
import {TinyDecide, type TinyDecideMetadata, type TinyDecidePrototype} from '../src/tinydecide/engine.ts';

type EngineRequest = Parameters<TinyDecide['answer']>[0];
type EnginePrototypes = Parameters<TinyDecide['answer']>[1];
type EngineResult = ReturnType<TinyDecide['answer']>;
type EngineInstance = Pick<TinyDecide, 'W' | 'answer'>;
type UpstreamModule = {TinyDecide: new (metadata: TinyDecideMetadata, buffer: ArrayBuffer) => EngineInstance};

const engineSourceSha256 = '577fb33b169b370711248b94bf4d9c6f61eda5f4f158710f177a96ffe74990d1';

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

async function readPinnedAsset(directory: string, name: keyof typeof TINYDECIDE_FILES): Promise<Buffer> {
  const bytes = await readFile(join(directory, name));
  const expected = TINYDECIDE_FILES[name];
  assert.equal(bytes.byteLength, expected.bytes, `${name} size`);
  assert.equal(sha256(bytes), expected.sha256, `${name} SHA-256`);
  return bytes;
}

function makePrototype(optionCount: number, dimensions: number, offset: number): TinyDecidePrototype {
  return {
    vec: Float32Array.from({length: optionCount * dimensions}, (_, index) => Math.sin(index + offset) * 0.1),
    cnt: Int32Array.from({length: optionCount}, (_, index) => index + 1),
    center: null,
    lam: 0.75
  };
}

async function main(): Promise<void> {
  const assetDirectory = process.argv[2];
  if (!assetDirectory)
    throw new Error('Usage: node --import tsx test/tinydecide-engine-parity.mts <pinned-model-directory>');

  const [metadataBytes, modelBytes] = await Promise.all([
    readPinnedAsset(assetDirectory, 'meta.json'),
    readPinnedAsset(assetDirectory, 'model.bin')
  ]);
  const metadata = JSON.parse(metadataBytes.toString('utf8')) as TinyDecideMetadata;
  const modelBuffer = new Uint8Array(modelBytes.byteLength);
  modelBuffer.set(modelBytes);
  const buffer = modelBuffer.buffer;
  const fixtureUrl = new URL('./fixtures/vendor/tinydecide-engine.js.txt', import.meta.url);
  const originalSource = await readFile(fixtureUrl);
  assert.equal(sha256(originalSource), engineSourceSha256, 'upstream engine fixture SHA-256');

  const temporaryDirectory = await mkdtemp(join(tmpdir(), 'tinydecide-parity-'));
  const temporaryEngine = join(temporaryDirectory, 'tinydecide-engine.mjs');
  try {
    await writeFile(temporaryEngine, originalSource);
    const upstream = (await import(pathToFileURL(temporaryEngine).href)) as UpstreamModule;
    const currentModel = new TinyDecide(metadata, buffer);
    const upstreamModel = new upstream.TinyDecide(metadata, buffer);

    for (const entry of metadata.tensors) {
      assert.deepEqual(currentModel.W.get(entry.name).shape, upstreamModel.W.get(entry.name).shape, entry.name);
      assert.deepEqual(currentModel.W.get(entry.name).data, upstreamModel.W.get(entry.name).data, entry.name);
    }

    const request: EngineRequest = {
      state: 'The story took place in Paris in 2024.',
      questions: [
        {type: 'choice', text: 'Where did it take place?', options: ['Paris', 'London', 'Tokyo']},
        {type: 'score', text: 'How positive is the tone?', options: ['low', 'medium', 'high']},
        {type: 'noul', text: 'Is there a safety concern?'},
        {type: 'span', text: 'What year is mentioned?'}
      ]
    };
    const dimensions = metadata.cfg.dh_head;
    const prototypes: EnginePrototypes = [
      makePrototype(3, dimensions, 1),
      makePrototype(3, dimensions, 4),
      makePrototype(2, dimensions, 7),
      undefined
    ];
    const current = currentModel.answer(request, prototypes);
    const reference = upstreamModel.answer(request, prototypes);
    assert.deepEqual(compareableResult(current), compareableResult(reference));
    process.stdout.write(
      `TinyDecide TS parity passed: ${metadata.tensors.length} tensors and ${current.answers.length} typed heads; ` +
        `meta=${TINYDECIDE_FILES['meta.json'].sha256}, model=${TINYDECIDE_FILES['model.bin'].sha256}\n`
    );
  } finally {
    await rm(temporaryDirectory, {recursive: true, force: true});
  }
}

function compareableResult(result: EngineResult): Omit<EngineResult, 'ms'> {
  const {ms: _ms, ...comparable} = result;
  return comparable;
}

await main();
