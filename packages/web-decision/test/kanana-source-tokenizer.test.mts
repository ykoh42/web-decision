import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {createKananaTokenizer} from '../src/kanana/tokenizer-asset.mjs';
import {openKananaSource} from '../src/kanana/source.mjs';
import {
  KANANA_ASSETS_URL,
  KANANA_FILES,
  KANANA_MODEL_ID,
  KANANA_REPOSITORY,
  KANANA_REVISION,
  KANANA_SOURCE_SPEC
} from '../src/kanana/contract.mjs';
import {candidates, protocol, tokenizer as tokenizerCore, truncation} from './kanana-core-modules.mts';

type TestSourceOptions = {
  cacheEnabled?: boolean;
  fetchImpl?: (url: string | URL) => Promise<Response>;
};
const openTestSource = openKananaSource as unknown as (
  options: TestSourceOptions
) => Promise<{read(name: string): Promise<ArrayBuffer | Uint8Array>}>;

const configuredSnapshot = process.env.KANANA_TOKENIZER_SNAPSHOT;
const snapshotDirectory = resolve(
  configuredSnapshot ??
    join(homedir(), '.cache/huggingface/hub/models--kakaocorp--kanana-2-v-decide-3b-2611/snapshots', KANANA_REVISION)
);
const snapshotHasFiles = Object.keys(KANANA_FILES).every((name) => existsSync(join(snapshotDirectory, name)));
const skipWithoutSnapshot =
  !configuredSnapshot && !snapshotHasFiles ? 'pinned Kanana tokenizer is not cached locally' : false;
const fixturePath = join(fileURLToPath(new URL('.', import.meta.url)), 'fixtures/kanana-tokenizer-python-ids.json');
const reference = JSON.parse(await readFile(fixturePath, 'utf8'));
const systemOneFixturePath = join(
  fileURLToPath(new URL('.', import.meta.url)),
  'fixtures/kanana-system-one-python-ids.json'
);
const systemOneReference = JSON.parse(await readFile(systemOneFixturePath, 'utf8'));

async function readSnapshotAsset(name: string) {
  return new Uint8Array(await readFile(join(snapshotDirectory, name)));
}

async function loadPinnedTokenizer() {
  const [configBytes, tokenizerConfigBytes, tokenizerBytes] = await Promise.all([
    readSnapshotAsset('config.json'),
    readSnapshotAsset('tokenizer_config.json'),
    readSnapshotAsset('tokenizer.json')
  ]);
  const config = JSON.parse(new TextDecoder().decode(configBytes));
  const tokenizerConfig = JSON.parse(new TextDecoder().decode(tokenizerConfigBytes));
  const tokenizerJson = JSON.parse(new TextDecoder().decode(tokenizerBytes));
  return createKananaTokenizer(tokenizerJson, tokenizerConfig, config);
}

function hashPrompt(prompt: string) {
  return createHash('sha256').update(prompt, 'utf8').digest('hex');
}

function expandReferenceIds(ids: (number | [number, number])[]) {
  return ids.flatMap((entry) => (Array.isArray(entry) ? Array(entry[1]).fill(entry[0]) : [entry]));
}

test(
  'pinned Kanana source reads only the revision-pinned tokenizer and metadata assets',
  {skip: skipWithoutSnapshot},
  async () => {
    assert.equal(KANANA_MODEL_ID, 'kanana-2-v-decide-3b-2611');
    assert.equal(KANANA_REPOSITORY, 'kakaocorp/kanana-2-v-decide-3b-2611');
    assert.equal(KANANA_SOURCE_SPEC.totalBytes, 10059848);
    assert.deepEqual(Object.keys(KANANA_FILES), ['config.json', 'tokenizer_config.json', 'tokenizer.json']);

    const requests: string[] = [];
    const source = await openTestSource({
      cacheEnabled: false,
      async fetchImpl(url: string | URL) {
        requests.push(String(url));
        const asset = new URL(url).pathname.split('/').at(-1);
        const bytes = await readSnapshotAsset(asset!);
        return new Response(bytes.slice().buffer as ArrayBuffer);
      }
    });

    for (const name of Object.keys(KANANA_FILES) as (keyof typeof KANANA_FILES)[]) {
      const bytes = await source.read(name);
      assert.equal(bytes.byteLength, KANANA_FILES[name].bytes);
      assert.deepEqual(new Uint8Array(bytes), await readSnapshotAsset(name));
    }
    assert.deepEqual(
      requests,
      Object.keys(KANANA_FILES).map((name) => new URL(name, KANANA_ASSETS_URL).href)
    );
    await assert.rejects(source.read('model.safetensors'), /not listed in manifest/);
  }
);

test('pinned Kanana source rejects wrong byte counts and SHA-256 values', {skip: skipWithoutSnapshot}, async (t) => {
  const original = await readSnapshotAsset('tokenizer_config.json');
  const cases: [string, Uint8Array][] = [
    ['short response', original.slice(0, original.byteLength - 1)],
    ['wrong digest', Uint8Array.from(original, (byte, index) => (index === 0 ? byte ^ 1 : byte))]
  ];

  for (const [label, bytes] of cases) {
    await t.test(label, async () => {
      const source = await openTestSource({
        cacheEnabled: false,
        async fetchImpl() {
          return new Response(bytes.slice().buffer as ArrayBuffer);
        }
      });
      await assert.rejects(source.read('tokenizer_config.json'), /Integrity check failed/);
    });
  }
});

test(
  'Kanana tokenizer IDs match the pinned Python tokenizer for query, image, and document prompts',
  {
    skip: skipWithoutSnapshot
  },
  async () => {
    assert.equal(reference.model, KANANA_REPOSITORY);
    assert.equal(reference.revision, KANANA_REVISION);
    assert.equal(reference.reference.api, 'transformers.AutoTokenizer.encode(prompt)');

    const tokenizer = await loadPinnedTokenizer();
    const choiceRequest = protocol.prepareKananaTextSystemOne({
      state: {parcel: {arrived: 'late'}},
      questions: {
        delivery: {
          type: 'choice',
          instructions: 'Did the parcel arrive late?',
          criteria: new Map([
            ['late', 'Arrived after the promised date'],
            ['on_time', 'Arrived by the promised date']
          ])
        }
      }
    });
    const choice = tokenizerCore.encodeKananaQuery(
      tokenizer,
      protocol.renderKananaQueryMessages(choiceRequest, 'delivery')
    );

    const imageRequest = protocol.prepareKananaSystemOne({
      state: 'A red mug sits beside the laptop.',
      images: ['local-image-placeholder'],
      questions: {visible: {type: 'noul', instructions: 'Is there a red mug?'}}
    });
    const image = tokenizerCore.encodeKananaQuery(
      tokenizer,
      protocol.renderKananaQueryMessages(imageRequest, 'visible'),
      [[1, 4, 16]]
    );

    const document = truncation.truncateKananaDocument(tokenizer, 'A parcel arrived after the promised date.');
    const actual = {
      choice_query: choice,
      image_query: image,
      candidate_document: document
    };
    for (const [name, result] of Object.entries(actual)) {
      const expected = reference.cases[name];
      assert.equal(hashPrompt(result.prompt), expected.promptSha256, `${name} prompt changed`);
      assert.deepEqual(result.inputIds, expandReferenceIds(expected.inputIds), `${name} token IDs changed`);
    }

    assert.deepEqual(tokenizer.ids, {
      '<|begin_of_text|>': 128000,
      '<|end_of_text|>': 128001,
      '<|im_start|>': 128009,
      '<|im_end|>': 128010,
      '<|vision_start|>': 128017,
      '<|vision_end|>': 128018,
      '<|image_pad|>': 128020,
      '<|video_pad|>': 128021,
      '<think>': 128032,
      '</think>': 128033
    });
  }
);

test(
  'Kanana System One prepares the same 13 query and candidate token IDs as the pinned Python model',
  {skip: skipWithoutSnapshot},
  async () => {
    assert.equal(systemOneReference.model.id, KANANA_REPOSITORY);
    assert.equal(systemOneReference.model.revision, KANANA_REVISION);
    assert.equal(systemOneReference.reference.api, 'transformers.AutoTokenizer.encode(prompt)');

    const tokenizer = await loadPinnedTokenizer();
    const prepared = protocol.prepareKananaTextSystemOne(systemOneReference.request);
    const preparedCandidates = candidates.prepareKananaCandidates(tokenizer, prepared);
    const actual: {id: string; inputIds: number[]}[] = [];

    for (const entry of prepared.questionEntries) {
      const query = candidates.truncateKananaQueryWithCandidateFallback(tokenizer, prepared, entry.id);
      actual.push({id: `${entry.id}:query`, inputIds: query.inputIds});

      const candidateEntry = preparedCandidates.find(({id}) => id === entry.id);
      assert.ok(candidateEntry, `missing candidate documents for ${entry.id}`);
      for (const document of candidateEntry.candidateDocuments)
        actual.push({id: `${entry.id}:${document.key}`, inputIds: document.inputIds});
    }

    assert.deepEqual(actual, systemOneReference.inputs);
  }
);
