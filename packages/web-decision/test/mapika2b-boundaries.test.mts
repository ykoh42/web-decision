import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import test from 'node:test';
import {Tokenizer} from '@huggingface/tokenizers';
import {
  MAPIKA2B_DECIDER_REVISION,
  MAPIKA2B_REPOSITORY,
  MAPIKA2B_REVISION,
  MAPIKA2B_TOKENIZER_CONFIG_SHA256,
  MAPIKA2B_TOKENIZER_SHA256
} from '../src/mapika2b/contract.mts';
import {createMapika2bLabels, prepareMapika2bRequest} from '../src/mapika2b/protocol.mts';

const snapshot = process.env.WEB_DECISION_MAPIKA2B_SNAPSHOT;
test('request boundaries match the pinned Mapika source', {skip: !snapshot}, async () => {
  async function tokenizerFile(name: string, digest: string) {
    const bytes = await readFile(join(snapshot!, name));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), digest);
    return JSON.parse(bytes.toString('utf8')) as object;
  }
  const [data, config, reference] = await Promise.all([
    tokenizerFile('tokenizer.json', MAPIKA2B_TOKENIZER_SHA256),
    tokenizerFile('tokenizer_config.json', MAPIKA2B_TOKENIZER_CONFIG_SHA256),
    readFile(new URL('../../../scripts/mapika/reference-boundaries.json', import.meta.url), 'utf8')
  ]);
  const fixture = JSON.parse(reference) as {
    source: {repository: string; revision: string; deciderRevision: string};
    cases: {id: string; status: 'accepted' | 'rejected'; request: unknown; usage?: {input_tokens: number}}[];
  };
  assert.deepEqual(
    [fixture.source.repository, fixture.source.revision, fixture.source.deciderRevision],
    [MAPIKA2B_REPOSITORY, MAPIKA2B_REVISION, MAPIKA2B_DECIDER_REVISION]
  );
  const tokenizer = new Tokenizer(data, config);
  const labels = createMapika2bLabels(tokenizer);
  for (const item of fixture.cases) {
    if (item.status === 'rejected') {
      assert.throws(
        () => prepareMapika2bRequest(item.request, tokenizer, labels),
        (cause: unknown) => cause instanceof Error && 'code' in cause && cause.code === 'INVALID_INPUT',
        item.id
      );
    } else {
      const result = prepareMapika2bRequest(item.request, tokenizer, labels);
      assert.equal(result.inputTokenCount, item.usage?.input_tokens, item.id);
    }
  }
});
