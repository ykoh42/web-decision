import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {loadGemma4V2Model} from '../../../packages/web-decision/src/strands/gemma4-v2-loader.mts';
import type {
  Gemma4V2OwnedRunner,
  Gemma4V2Request
} from '../../../packages/web-decision/src/strands/gemma4-v2-session.mts';
import type {readVerifiedAsset} from '../../../packages/web-decision/src/assets.ts';
import type {openPinnedGemma4V2Source} from '../../../packages/web-decision/src/strands/gemma4-v2-source.mts';
const fixtures = JSON.parse(await readFile('../e2b-v2-prompt-reference.json', 'utf8')) as Array<{
  state_ids: number[];
  questions: Record<string, {question_ids: number[]}>;
  request: Gemma4V2Request;
}>;
let index = 0;
const runner: Gemma4V2OwnedRunner = {
  async run(input) {
    const expected = fixtures[index]!;
    assert.deepEqual(input.stateIds, expected.state_ids);
    assert.deepEqual(input.questionIds, Object.values(expected.questions)[0]!.question_ids);
    return [0, 0];
  },
  releaseCachedMemory() {},
  dispose() {}
};
const readAsset = (async (_base, manifest, name) => {
  const data = await readFile(resolve('../strands-e2b-v2', name));
  assert.equal(data.length, manifest.files[name]!.bytes);
  assert.equal(createHash('sha256').update(data).digest('hex'), manifest.files[name]!.sha256);
  return Uint8Array.from(data).buffer;
}) as typeof readVerifiedAsset;
const source = {dispose() {}} as Awaited<ReturnType<typeof openPinnedGemma4V2Source>>;
const model = await loadGemma4V2Model(
  {},
  {
    readAsset,
    openSource: async () => source,
    createRunner: async () => runner,
    prepareImage: async () => {
      throw Error('unexpected image');
    }
  }
);
for (const fixture of fixtures.slice(0, 2)) {
  await model.systemOne(fixture.request);
  index++;
}
await model.dispose();
console.log('Pinned assets, real tokenizer, and loader preserve both upstream token sequences');
