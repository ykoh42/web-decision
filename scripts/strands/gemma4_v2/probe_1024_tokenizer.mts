import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {loadGemma4V2Model} from '../../../packages/web-decision/src/strands/gemma4-v2-loader.mts';
import type {readVerifiedAsset} from '../../../packages/web-decision/src/assets.js';
import type {openPinnedGemma4V2Source} from '../../../packages/web-decision/src/strands/gemma4-v2-source.mts';

const root = resolve(process.argv[2] ?? '../e2b-v2-text-1024-reference');
const checkpoint = resolve(process.argv[3] ?? '../strands-e2b-v2');
const request = JSON.parse(await readFile(resolve(root, 'request.json'), 'utf8'));
const expected = JSON.parse(await readFile(resolve(root, 'tokens.json'), 'utf8')) as {
  state: number[];
  questions: number[][];
};
const readAsset = (async (_base, manifest, name) => {
  const bytes = await readFile(resolve(checkpoint, name));
  assert.equal(bytes.length, manifest.files[name]!.bytes);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), manifest.files[name]!.sha256);
  return Uint8Array.from(bytes).buffer;
}) as typeof readVerifiedAsset;
const source = {dispose() {}} as Awaited<ReturnType<typeof openPinnedGemma4V2Source>>;
let calls = 0;
const session = await loadGemma4V2Model(
  {},
  {
    readAsset,
    openSource: async () => source,
    createRunner: async () => ({
      async run(input) {
        calls++;
        assert.deepEqual(input.stateIds, expected.state);
        assert.deepEqual(input.questionIds, expected.questions[0]);
        return [0, 0];
      },
      releaseCachedMemory() {},
      dispose() {}
    })
  }
);
const result = await session.systemOne(request);
assert.equal(result.usage.input_tokens, 1024);
assert.equal(calls, 1);
await session.dispose();
console.log('1024 token IDs match the pinned official tokenizer; runner is a CPU test double.');
