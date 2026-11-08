import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import test from 'node:test';
import {
  SKEJJA_ARCHIVE,
  SKEJJA_INT8_PROFILE,
  SKEJJA_INT8DYN_PROFILE,
  SKEJJA_INT8_MODEL,
  SKEJJA_INT8DYN_MODEL
} from '../src/skejja/contract.mjs';
import {fitSkejjaPlatt} from '../src/skejja/calibration.mjs';
import {loadSkejja, validateSkejjaLoadOptions} from '../src/skejja/loader.mjs';
import {openSkejjaSource, parseSkejjaArchive} from '../src/skejja/source.mjs';
import {describeSkejjaModel} from '../src/skejja-preview-registry.mjs';
import {createSkejjaSession} from '../src/skejja/session.mjs';
import {
  asSkejjaText,
  finishSkejjaDecision,
  finishSkejjaSystemOne,
  prepareSkejjaDecision,
  prepareSkejjaSystemOne
} from '../src/skejja/protocol.mjs';
import {createSkejjaPrompt, createSkejjaTokenizer, SKEJJA_TRUNCATION_MARKER} from '../src/skejja/tokenizer.mjs';

class CharacterTokenizer {
  encode(text) {
    return Array.from(text, (character) => character.codePointAt(0));
  }

  decode(ids) {
    return String.fromCodePoint(...ids);
  }
}

function bytes(value) {
  return new TextEncoder().encode(JSON.stringify(value)).buffer;
}

function digest(value) {
  return createHash('sha256').update(value).digest('hex');
}

function tarHeader(path, size, type = '\0', corruptChecksum = false) {
  const header = Buffer.alloc(512);
  const octal = (value, width) => `${value.toString(8).padStart(width - 1, '0')}\0`;
  Buffer.from(path).copy(header, 0);
  Buffer.from(octal(0o644, 8)).copy(header, 100);
  Buffer.from(octal(0, 8)).copy(header, 108);
  Buffer.from(octal(0, 8)).copy(header, 116);
  Buffer.from(octal(size, 12)).copy(header, 124);
  Buffer.from(octal(0, 12)).copy(header, 136);
  header.fill(0x20, 148, 156);
  header[156] = type === '\0' ? 0 : type.charCodeAt(0);
  Buffer.from('ustar\0').copy(header, 257);
  Buffer.from('00').copy(header, 263);
  let checksum = 0;
  for (const byte of header) checksum += byte;
  const checksumText = `${(checksum + (corruptChecksum ? 1 : 0)).toString(8).padStart(6, '0')}\0 `;
  Buffer.from(checksumText).copy(header, 148);
  return header;
}

function makeTarGzip(files, {corruptChecksum = false} = {}) {
  const root = `${SKEJJA_ARCHIVE.root}/`;
  const chunks = [tarHeader(root, 0, '5')];
  for (const [index, [name, value]] of files.entries()) {
    const payload = Buffer.from(value);
    chunks.push(tarHeader(`${root}${name}`, payload.byteLength, '\0', corruptChecksum && index === 0));
    chunks.push(payload);
    const padding = (512 - (payload.byteLength % 512)) % 512;
    if (padding) chunks.push(Buffer.alloc(padding));
  }
  chunks.push(Buffer.alloc(10 * 1024));
  return gzipSync(Buffer.concat(chunks));
}

function makeSmallArchiveFixture() {
  const files = {
    'tokenizer.json': Buffer.from('tokenizer'),
    'edge-v32k-ft6.int8.onnx': Buffer.from('graph'),
    'edge-v32k-ft6.int8.onnx.data': Buffer.from('weights'),
    'archive-note.txt': Buffer.from('not selected')
  };
  const compressed = makeTarGzip(Object.entries(files));
  const profileFiles = Object.fromEntries(
    Object.entries(files)
      .filter(([name]) => name !== 'archive-note.txt')
      .map(([name, value]) => [name, {bytes: value.byteLength, sha256: digest(value)}])
  );
  const profile = {
    files: profileFiles,
    totalBytes: Object.values(profileFiles).reduce((sum, file) => sum + file.bytes, 0)
  };
  const archiveSpec = {
    filename: 'skejja-test.tar.gz',
    bytes: compressed.byteLength,
    sha256: digest(compressed),
    root: SKEJJA_ARCHIVE.root,
    maxExpandedBytes: 64 * 1024,
    files: Object.fromEntries(Object.entries(files).map(([name, value]) => [name, value.byteLength]))
  };
  return {files, compressed, profile, archiveSpec};
}

class TestOrtTensor {
  constructor(type, data, dims) {
    Object.assign(this, {type, data, dims});
  }
}

function makeLoadDependencies(profile, logits = [0.2, -0.1, 0.5]) {
  const released = {count: 0};
  const runInputs = [];
  const values = {
    'tokenizer.json': bytes({model: {type: 'BPE', vocab: {'<bos>': 2, '<eos>': 1}}}),
    'tokenizer_config.json': bytes({
      tokenizer_class: 'PreTrainedTokenizerFast',
      add_bos_token: true,
      bos_token: '<bos>',
      eos_token: '<eos>',
      max_length: 4096
    }),
    'calibration.json': bytes({
      [profile.onnxFile]: {
        temperature: profile.temperature,
        platt_a: profile.plattA,
        platt_b: profile.plattB
      }
    }),
    [profile.onnxFile]: new ArrayBuffer(4),
    [profile.dataFile]: new ArrayBuffer(4)
  };
  class TestTokenizer extends CharacterTokenizer {
    constructor() {
      super();
    }

    token_to_id(token) {
      return {'<<LABEL>>': 32765, '<<SEP>>': 32766}[token];
    }

    encode(text) {
      return {ids: [2, ...super.encode(text), 1]};
    }

    decode(ids) {
      return super.decode(ids.filter((id) => id !== 1 && id !== 2));
    }
  }
  const ort = {
    Tensor: TestOrtTensor,
    InferenceSession: {
      async create(_graph, options) {
        assert.deepEqual(options.executionProviders, ['wasm']);
        assert.equal(options.intraOpNumThreads, 1);
        assert.equal(options.externalData[0].path, profile.dataFile);
        return {
          inputNames: ['input_ids', 'attention_mask', 'label_slots'],
          outputNames: ['logits'],
          async run(inputs) {
            runInputs.push(inputs);
            const count = inputs.label_slots.dims[0];
            const row = Array.from({length: count}, (_, index) => logits[index] ?? 0);
            return {logits: new TestOrtTensor('float32', Float32Array.from(row), [1, count])};
          },
          async release() {
            released.count++;
          }
        };
      }
    }
  };
  return {
    released,
    runInputs,
    dependencies: {
      ort,
      Tokenizer: TestTokenizer,
      async openSource(_options, selected) {
        assert.equal(selected, profile);
        return {
          async read(name) {
            assert.ok(name in values);
            return values[name];
          }
        };
      }
    }
  };
}

test('Skejja profiles pin both official ONNX variants and their external weights', () => {
  assert.equal(SKEJJA_INT8_MODEL, 'experimental-skejja-edge32k-ft6-int8');
  assert.equal(SKEJJA_INT8DYN_MODEL, 'experimental-skejja-edge32k-ft6-int8dyn');
  assert.equal(SKEJJA_INT8_PROFILE.totalBytes, 103141152);
  assert.equal(SKEJJA_INT8DYN_PROFILE.totalBytes, 127490939);
  assert.equal(SKEJJA_INT8_PROFILE.files['edge-v32k-ft6.int8.onnx.data'].bytes, 100674628);
  assert.equal(SKEJJA_INT8DYN_PROFILE.files['edge-v32k-ft6.int8dyn.onnx.data'].bytes, 124288004);
  assert.equal(
    SKEJJA_INT8_PROFILE.files['tokenizer.json'].sha256,
    SKEJJA_INT8DYN_PROFILE.files['tokenizer.json'].sha256
  );
  assert.equal(SKEJJA_ARCHIVE.filename, 'skejja-edge32k-ft6_onnx-int8.tar.gz');
  assert.equal(SKEJJA_ARCHIVE.bytes, 208260729);
  assert.equal(SKEJJA_ARCHIVE.sha256, '751e243f1fba39421d36d44a832fc42a34bb74c6bec68f15d956e219e1131f42');
});

test('Skejja ID-only source shares a verified archive parse between concurrent reads', async () => {
  const fixture = makeSmallArchiveFixture();
  const calls = {cache: 0, archive: 0};
  const progress = [];
  const source = await openSkejjaSource(
    {
      base: new URL('https://example.test/revision/'),
      cacheEnabled: true,
      offline: true,
      useArchive: true,
      onProgress: (value) => progress.push(value)
    },
    fixture.profile,
    {
      archiveSpec: fixture.archiveSpec,
      async openAssetCache(enabled) {
        assert.equal(enabled, true);
        calls.cache++;
        return {name: 'existing-asset-cache'};
      },
      async readVerifiedAsset(base, manifest, name, options) {
        assert.equal(base.href, 'https://example.test/revision/');
        assert.deepEqual(manifest.files[name], {
          bytes: fixture.archiveSpec.bytes,
          sha256: fixture.archiveSpec.sha256
        });
        assert.equal(name, fixture.archiveSpec.filename);
        assert.equal(options.cache.name, 'existing-asset-cache');
        assert.equal(options.offline, true);
        assert.equal(options.maxAssetBytes, fixture.archiveSpec.bytes);
        calls.archive++;
        return fixture.compressed.buffer.slice(
          fixture.compressed.byteOffset,
          fixture.compressed.byteOffset + fixture.compressed.byteLength
        );
      }
    }
  );

  const [tokenizer, graph, data] = await Promise.all([
    source.read('tokenizer.json'),
    source.read('edge-v32k-ft6.int8.onnx'),
    source.read('edge-v32k-ft6.int8.onnx.data')
  ]);
  assert.equal(calls.cache, 1);
  assert.equal(calls.archive, 1);
  assert.equal(Buffer.from(tokenizer).toString(), 'tokenizer');
  assert.equal(Buffer.from(graph).toString(), 'graph');
  assert.equal(Buffer.from(data).toString(), 'weights');
  assert.deepEqual(progress, [
    {loadedBytes: fixture.archiveSpec.bytes, totalBytes: fixture.archiveSpec.bytes, fraction: 1}
  ]);

  await source.release(['tokenizer.json', 'edge-v32k-ft6.int8.onnx', 'edge-v32k-ft6.int8.onnx.data']);
  await assert.rejects(source.read('LICENSE'), {code: 'INVALID_ASSET'});
});

test('Skejja archive parser extracts only pinned-profile files and rejects malformed tar data', async () => {
  const fixture = makeSmallArchiveFixture();
  const parsed = await parseSkejjaArchive(fixture.compressed, fixture.profile, {archiveSpec: fixture.archiveSpec});
  assert.equal(Buffer.from(parsed.get('tokenizer.json')).toString(), 'tokenizer');
  assert.equal(parsed.has('archive-note.txt'), false);

  const corrupt = makeTarGzip(Object.entries(fixture.files), {corruptChecksum: true});
  const corruptSpec = {...fixture.archiveSpec, bytes: corrupt.byteLength};
  await assert.rejects(parseSkejjaArchive(corrupt, fixture.profile, {archiveSpec: corruptSpec}), {
    code: 'INVALID_ASSET'
  });

  const duplicate = makeTarGzip([
    ['tokenizer.json', fixture.files['tokenizer.json']],
    ['tokenizer.json', fixture.files['tokenizer.json']],
    ...Object.entries(fixture.files).filter(([name]) => name !== 'tokenizer.json')
  ]);
  await assert.rejects(
    parseSkejjaArchive(duplicate, fixture.profile, {
      archiveSpec: {...fixture.archiveSpec, bytes: duplicate.byteLength}
    }),
    {code: 'INVALID_ASSET'}
  );

  const truncated = fixture.compressed.subarray(0, fixture.compressed.byteLength - 4);
  await assert.rejects(
    parseSkejjaArchive(truncated, fixture.profile, {
      archiveSpec: {...fixture.archiveSpec, bytes: truncated.byteLength}
    }),
    {code: 'INVALID_ASSET'}
  );

  const aborted = new AbortController();
  aborted.abort(new DOMException('Cancelled', 'AbortError'));
  await assert.rejects(
    parseSkejjaArchive(fixture.compressed, fixture.profile, {archiveSpec: fixture.archiveSpec, signal: aborted.signal}),
    {
      name: 'AbortError'
    }
  );
});

test('Skejja archive hashing aborts promptly while a digest is still pending', async () => {
  const fixture = makeSmallArchiveFixture();
  const controller = new AbortController();
  const abortReason = new DOMException('Cancelled during file verification', 'AbortError');
  let startHash;
  const hashStarted = new Promise((resolve) => {
    startHash = resolve;
  });
  let settleDigest;
  let digestSettled = false;
  const pendingDigest = new Promise((resolve) => {
    settleDigest = resolve;
  }).then((value) => {
    digestSettled = true;
    return value;
  });
  const parsing = parseSkejjaArchive(fixture.compressed, fixture.profile, {
    archiveSpec: fixture.archiveSpec,
    signal: controller.signal,
    hashBytes() {
      startHash();
      return pendingDigest;
    }
  });

  await hashStarted;
  controller.abort(abortReason);
  let timeout;
  const outcome = await Promise.race([
    parsing.then(
      () => ({type: 'resolved'}),
      (error) => ({type: 'rejected', error})
    ),
    new Promise((resolve) => {
      timeout = setTimeout(() => resolve({type: 'timeout'}), 1000);
    })
  ]);
  clearTimeout(timeout);
  assert.equal(outcome.type, 'rejected');
  assert.equal(outcome.error, abortReason);
  assert.equal(digestSettled, false);
  settleDigest('');
});

test('Skejja preview metadata separates row limits from the upstream question default', () => {
  for (const model of [SKEJJA_INT8_MODEL, SKEJJA_INT8DYN_MODEL]) {
    const limits = describeSkejjaModel(model).features.textChoice.limits;
    const tokenLimit = limits.find(({resource}) => resource === 'tokens');
    const questionLimit = limits.find(({resource}) => resource === 'questions');
    const optionLimit = limits.find(({resource}) => resource === 'options');

    assert.deepEqual([tokenLimit.scope, tokenLimit.max], ['question', 512]);
    assert.equal(questionLimit.min, 1);
    assert.equal(questionLimit.max, undefined);
    assert.match(questionLimit.description, /upstream server defaults to 256/i);
    assert.deepEqual([optionLimit.scope, optionLimit.min, optionLimit.max], ['question', 2, 255]);
  }
});

test('Skejja tokenizer checks its BPE markers and adds the original truncation rule', () => {
  const created = createSkejjaTokenizer(
    class {
      constructor() {}
      token_to_id(token) {
        return {'<<LABEL>>': 32765, '<<SEP>>': 32766}[token];
      }
      encode(text) {
        return {ids: [2, ...Array.from(text, (character) => character.codePointAt(0)), 1]};
      }
      decode(ids) {
        return String.fromCodePoint(...ids.filter((id) => id !== 1 && id !== 2));
      }
    },
    {model: {type: 'BPE', vocab: {'<bos>': 2, '<eos>': 1}}},
    {
      tokenizer_class: 'PreTrainedTokenizerFast',
      add_bos_token: true,
      bos_token: '<bos>',
      eos_token: '<eos>',
      max_length: 4096
    }
  );
  assert.ok(createSkejjaPrompt(created, ['on', 'off'], 'state').tokenCount < 512);
  assert.equal(SKEJJA_TRUNCATION_MARKER, '\n[…]\n');
  assert.throws(() => createSkejjaTokenizer(class {}, {model: {type: 'WordPiece'}}, {}), {code: 'INVALID_ASSET'});
});

test('Skejja formats JSON with upstream spacing and rejects cyclic values', () => {
  assert.equal(
    asSkejjaText({intent: ['set alarm', 1], enabled: true}),
    '{"intent": ["set alarm", 1], "enabled": true}'
  );
  const cyclic = {};
  cyclic.self = cyclic;
  assert.throws(() => asSkejjaText(cyclic), {code: 'INVALID_INPUT'});
});

test('Skejja uses the upstream Choice, Noul, and Score labels and calibration', () => {
  const tokenizer = new CharacterTokenizer();
  const prepared = prepareSkejjaSystemOne(
    {utterance: 'spegnere le luci', urgent: false},
    {
      route: {type: 'choice', instructions: 'Choose an intent.', criteria: {lights_off: 'turn off lights', other: ''}},
      reminder: {type: 'noul', instructions: 'Does the user want to set a reminder?'},
      impact: {type: 'score', instructions: 'How serious is it?', criteria: ['low', 'medium', 'high']}
    },
    tokenizer,
    SKEJJA_INT8_PROFILE
  );
  const result = finishSkejjaSystemOne(prepared, [[2, 0], [1], [0, 2, 0]], SKEJJA_INT8_PROFILE, new Map());
  assert.equal(result.model, SKEJJA_INT8_MODEL);
  assert.deepEqual(Object.keys(result.answers), ['route', 'reminder', 'impact']);
  assert.equal(result.answers.route.type, 'choice');
  assert.equal(result.answers.route.choice, 'lights_off');
  assert.ok(
    Math.abs(result.answers.route.probabilities.lights_off - 0.5 - result.answers.route.confidence / 2) < 1e-12
  );
  assert.ok(result.answers.reminder.noul > 0.5);
  assert.equal(result.answers.impact.type, 'score');
  assert.equal(result.answers.impact.legend['2'], 'high');
  assert.equal(result.answers.impact.score, 1);
  assert.equal(result.truncated, undefined);
  assert.ok(result.tokenCount > 0);
});

test('Skejja shortening preserves the beginning and end and fits the complete 512-token sequence', () => {
  const tokenizer = new CharacterTokenizer();
  const prepared = prepareSkejjaSystemOne(
    'start ' + 'middle '.repeat(180) + ' end',
    {
      route: {type: 'choice', criteria: {yes: 'yes', no: 'no'}}
    },
    tokenizer
  );
  const prompt = prepared.questions[0];
  assert.equal(prompt.truncated, true);
  assert.ok(prompt.tokenCount <= 512);
  assert.ok(prompt.inputIds.length <= 512);
  assert.equal(prepared.truncated, true);
});

test('Skejja applies the 512-token limit to long bodies and labels before inference', async () => {
  const {dependencies, runInputs} = makeLoadDependencies(SKEJJA_INT8_PROFILE);
  const model = await loadSkejja({model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/skejja/'}, dependencies);
  const longText = 'x'.repeat(200_001);
  const result = await model.systemOne(longText, {
    route: {type: 'choice', instructions: longText, criteria: {yes: 'yes', no: 'no'}}
  });
  assert.equal(result.truncated, true);
  assert.ok(result.tokenCount <= 512);
  assert.equal(runInputs.length, 1);

  const exactLabelBudget = await model.systemOne('', {
    route: {type: 'choice', criteria: {first: 'x'.repeat(452), second: 'y'}}
  });
  assert.ok(exactLabelBudget.tokenCount <= 512);
  assert.equal(runInputs.length, 2);

  await assert.rejects(
    model.systemOne('', {route: {type: 'choice', criteria: {first: 'x'.repeat(453), second: 'y'}}}),
    (error) => error.code === 'INVALID_INPUT' && error.message.includes('label text exceeds')
  );
  assert.equal(runInputs.length, 2);
  await model.dispose();
});

test('Skejja accepts long calibration states and instructions through 512-token truncation', async () => {
  const {dependencies, runInputs} = makeLoadDependencies(SKEJJA_INT8_PROFILE);
  const model = await loadSkejja({model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/skejja/'}, dependencies);
  const longText = 'x'.repeat(200_001);
  const calibration = await model.calibrateQuestion(
    {type: 'noul', instructions: longText, criteria: {true: 'Does the user want to continue?'}},
    Array.from({length: 8}, (_, index) => ({state: index === 0 ? longText : `example ${index}`, answer: index < 4}))
  );

  assert.equal(calibration.examples, 8);
  assert.equal(calibration.positives, 4);
  assert.ok(Number.isFinite(calibration.a));
  assert.ok(Number.isFinite(calibration.b));
  assert.equal(runInputs.length, 8);
  assert.ok(runInputs.every(({input_ids}) => input_ids.dims[1] <= 512));
  await model.dispose();
});

test('Skejja enforces typed limits and preserves decision option order', () => {
  const tokenizer = new CharacterTokenizer();
  assert.throws(
    () => prepareSkejjaSystemOne('state', {bad: {type: 'score', criteria: Array(11).fill('level')}}, tokenizer),
    {code: 'INVALID_INPUT'}
  );
  assert.throws(() => prepareSkejjaSystemOne('state', {bad: {type: 'choice', criteria: {only: 'one'}}}, tokenizer), {
    code: 'INVALID_INPUT'
  });
  const prepared = prepareSkejjaDecision(
    {
      state: 'turn off the lights',
      questions: [{instructions: 'Select one.', options: ['lights_on', 'lights_off']}]
    },
    tokenizer
  );
  const result = finishSkejjaDecision(prepared, [[0, 2]], SKEJJA_INT8_PROFILE);
  assert.equal(result.answers[0].selected, 'lights_off');
  assert.deepEqual(
    result.answers[0].probabilities.map(({option}) => option),
    ['lights_on', 'lights_off']
  );
  assert.throws(() => prepareSkejjaSystemOne('state', {bad: {type: 'noul', instructions: '   '}}, tokenizer), {
    code: 'INVALID_INPUT'
  });
});

test('Skejja accepts 257 generic and typed questions and preserves row order', async () => {
  const tokenizer = new CharacterTokenizer();
  const calls = [];
  const session = createSkejjaSession({
    tokenizer,
    runner: {
      async run(inputIds, labelCount) {
        calls.push({text: tokenizer.decode(inputIds), labelCount});
        return Array.from({length: labelCount}, (_, index) => index);
      },
      async releaseCachedMemory() {},
      async dispose() {}
    }
  });
  const genericQuestions = Array.from({length: 257}, (_, index) => ({
    instructions: `question ${index}`,
    options: [`no ${index}`, `yes ${index}`]
  }));
  const generic = await session.decide({state: 'shared state', questions: genericQuestions});

  assert.equal(generic.answers.length, 257);
  assert.deepEqual(
    generic.answers.map(({selected}) => selected),
    genericQuestions.map(({options}) => options[1])
  );
  assert.equal(calls.length, 257);
  assert.ok(calls.every(({labelCount}) => labelCount === 2));
  assert.match(calls[0].text, /no 0/);
  assert.match(calls.at(-1).text, /no 256/);

  calls.length = 0;
  const ids = Array.from({length: 257}, (_, index) => `question-${String(index).padStart(3, '0')}`);
  const typedQuestions = Object.fromEntries(
    ids.map((id, index) => [id, {type: 'choice', criteria: {[`no-${index}`]: 'no', [`yes-${index}`]: 'yes'}}])
  );
  const typed = await session.systemOne('shared state', typedQuestions);

  assert.deepEqual(Object.keys(typed.answers), ids);
  assert.equal(calls.length, 257);
  assert.ok(calls.every(({labelCount}) => labelCount === 2));
  assert.match(calls[0].text, /no/);
  assert.match(calls.at(-1).text, /yes/);
  await session.dispose();
});

test('Skejja Platt calibration is finite, anchored and rejects one-class data', () => {
  const fitted = fitSkejjaPlatt([-2, -1, -0.5, 0.2, 1, 2], [0, 0, 0, 1, 1, 1], [0.84, 0.4]);
  assert.ok(Number.isFinite(fitted.a));
  assert.ok(Number.isFinite(fitted.b));
  assert.ok(fitted.a > 0);
  assert.throws(() => fitSkejjaPlatt([0, 1], [1, 2], [1, 0]), {code: 'INVALID_INPUT'});
});

test('Skejja load options are strict and require a matching HTTPS directory', () => {
  assert.equal(
    validateSkejjaLoadOptions({model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/models'}).base.href,
    'https://example.test/models/'
  );
  const defaults = validateSkejjaLoadOptions({model: SKEJJA_INT8_MODEL});
  assert.equal(
    defaults.base.href,
    'https://huggingface.co/vincyber/skejja-edge32k-ft6/resolve/a076ad7d698b11b15dc70fe067d15c3d1140761f/'
  );
  assert.equal(defaults.useArchive, true);
  assert.equal(
    validateSkejjaLoadOptions({model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/models'}).useArchive,
    false
  );
  for (const options of [
    {model: SKEJJA_INT8DYN_MODEL, assetsUrl: 'https://example.test/'},
    {model: SKEJJA_INT8_MODEL, assetsUrl: 'file:///tmp/'},
    {model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/', maxPrefetchBytes: 1},
    {model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/', other: true},
    {model: SKEJJA_INT8_MODEL, assetsUrl: 'https://user:pass@example.test/'},
    {model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/?unsafe'}
  ])
    assert.throws(() => validateSkejjaLoadOptions(options), {code: 'INVALID_INPUT'});
});

test('Skejja loads its pinned WASM route, serves typed and generic decisions, then releases ONNX', async () => {
  const {dependencies, released, runInputs} = makeLoadDependencies(SKEJJA_INT8_PROFILE);
  const model = await loadSkejja({model: SKEJJA_INT8_MODEL, assetsUrl: 'https://example.test/skejja/'}, dependencies);
  const generic = await model.decide({state: 'state', questions: [{instructions: 'choose', options: ['no', 'yes']}]});
  assert.equal(generic.answers[0].selected, 'no');
  const typed = await model.systemOne('state', {
    choice: {type: 'choice', criteria: {one: 'first', two: 'second'}},
    binary: {type: 'noul', instructions: 'Does the user want to turn it on?'},
    score: {type: 'score', criteria: ['low', 'high']}
  });
  assert.equal(typed.model, SKEJJA_INT8_MODEL);
  assert.deepEqual(Object.keys(typed.answers), ['choice', 'binary', 'score']);
  const calibration = await model.calibrateQuestion(
    {type: 'noul', instructions: 'Does the user want to continue?'},
    Array.from({length: 8}, (_, index) => ({state: `example ${index}`, answer: index < 4}))
  );
  assert.equal(calibration.examples, 8);
  assert.equal(calibration.positives, 4);
  assert.ok(Number.isFinite(calibration.a));
  await model.releaseCachedMemory();
  await model.dispose();
  assert.equal(released.count, 1);
  assert.equal(runInputs.length, 12);
  await assert.rejects(model.decide({state: 'x', questions: [{instructions: '', options: ['a', 'b']}]}), {
    code: 'SESSION_CLOSED'
  });
});

test('Skejja also selects the separate dynamic-int8 graph without changing route semantics', async () => {
  const {dependencies} = makeLoadDependencies(SKEJJA_INT8DYN_PROFILE);
  const model = await loadSkejja(
    {model: SKEJJA_INT8DYN_MODEL, assetsUrl: 'https://example.test/skejja/'},
    dependencies
  );
  const result = await model.systemOne('state', {
    check: {type: 'noul', instructions: 'Does the user want to proceed?'}
  });
  assert.equal(result.model, SKEJJA_INT8DYN_MODEL);
  assert.ok(result.answers.check.noul > 0 && result.answers.check.noul < 1);
  await model.dispose();
});
