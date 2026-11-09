import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {selectModelFamily} from '../src/model-family.js';
import {packDebertaRequest, unpackDebertaScores} from '../src/deberta/packing.mjs';
import {validateDebertaTensors} from '../src/deberta/validate.mjs';
import {ExperimentalDebertaSession} from '../src/deberta/model.mjs';
import {readDebertaManifestResponse, validateDebertaManifest} from '../src/deberta/contract.mjs';
import {responseFromReader} from './stream-helpers.mjs';

const tokenizer = {
  encode(text: string) {
    return {
      ids:
        text === '[STATE]'
          ? [128001]
          : text
              .trim()
              .split(/\s+/)
              .filter(Boolean)
              .map(() => 7)
    };
  }
};
const request = () => ({
  state: 'a state',
  questions: [
    {instructions: 'choose color', options: ['red', 'blue']},
    {instructions: 'choose direction', options: ['left', 'right', 'center']}
  ]
});

test('selects explicit model families and preserves the Kev default', () => {
  assert.equal(selectModelFamily({assetsUrl: '/kev'} as unknown as {model?: unknown}), 'kev');
  assert.equal(selectModelFamily({model: 'kev-0.6b-q4f16'}), 'kev');
  assert.equal(selectModelFamily({model: 'open-jev-deberta-v3-large-q4f16'}), 'deberta');
  for (const value of [null, '', 'other', 42])
    assert.throws(() => selectModelFamily({model: value}), {code: 'INVALID_INPUT'});
});

test('packs multiple questions, snapshots options and applies the model temperature', () => {
  const submitted = request(),
    packed = packDebertaRequest(tokenizer, submitted);
  submitted.questions[0]!.options[0] = 'changed';
  assert.equal(packed.pairs, 5);
  assert.equal(packed.chunks.length, 2);
  assert.equal(packed.optionGroups[0]![0], 'red');
  const result = unpackDebertaScores(packed, [1, 0, 0, 1, 2]);
  assert.equal(result.answers[0]!.selected, 'red');
  assert.equal(result.answers[1]!.selected, 'center');
  assert.ok(
    Math.abs(
      result.answers[0]!.probabilities[0]!.probability / result.answers[0]!.probabilities[1]!.probability -
        Math.exp(1 / 1.05)
    ) < 1e-12
  );
});

test('rejects reserved markers, duplicate options and excess input without truncating', () => {
  const reserved = request();
  reserved.state = '[STATE]';
  assert.throws(() => packDebertaRequest(tokenizer, reserved), /reserved/);
  const duplicate = request();
  duplicate.questions[0]!.options[1] = 'red';
  assert.throws(() => packDebertaRequest(tokenizer, duplicate), /duplicate/);
  const longState = request();
  longState.state = 'token '.repeat(257);
  assert.throws(() => packDebertaRequest(tokenizer, longState), /state exceeds/);
  const longQuestion = request();
  longQuestion.questions[0]!.instructions = 'token '.repeat(512);
  assert.throws(() => packDebertaRequest(tokenizer, longQuestion), /Packed input exceeds/);
});

test('uses the packed-token budget for large question and pair requests', async () => {
  const questions = Array.from({length: 513}, (_, index) => ({
    instructions: `question ${index}`,
    options: ['yes', 'no']
  }));
  const x = session();
  await assert.rejects(x.model.decide({state: '', questions}), (error) => {
    assert.equal((error as {code: string}).code, 'INVALID_INPUT');
    assert.match((error as {cause?: {message: string}}).cause?.message ?? '', /Packed input exceeds 512 tokens/);
    return true;
  });

  const pairs = packDebertaRequest(tokenizer, {
    state: 'state',
    questions: [
      {
        instructions: 'choose',
        options: Array.from({length: 253}, (_, index) => `o${index}`)
      }
    ]
  });
  assert.equal(pairs.realTokens, 512);
  assert.equal(pairs.pairs, 253);
  assert.equal(pairs.chunks.length, 85);
  assert.equal(pairs.chunks.at(-1)!.count, 1);

  assert.throws(
    () =>
      packDebertaRequest(tokenizer, {
        state: 'state',
        questions: [{instructions: 'choose', options: Array.from({length: 254}, (_, index) => `o${index}`)}]
      }),
    /Packed input exceeds 512 tokens/
  );
});

test('tensor validation bounds segment bookkeeping by the sequence axis', () => {
  assert.throws(
    () =>
      validateDebertaTensors({
        groups: Array(513).fill(1),
        inputs: {
          input_ids: [7],
          attention_mask: [1],
          seg: [0],
          pair_q: Array(513).fill(513),
          pair_opt: Array.from({length: 513}, (_, index) => index)
        }
      }),
    /question and option segments cannot exceed sequence length/
  );
});

test('pins the DeBERTa manifest and stops waiting on a cancelled body', async () => {
  const pinned = JSON.parse(readFileSync(new URL('../src/deberta/contract.json', import.meta.url), 'utf8'));
  assert.equal(validateDebertaManifest(pinned).model, 'open-jev-deberta-v3-large-q4f16');
  const altered = structuredClone(pinned);
  altered.files['model.wasm'].sha256 = '0'.repeat(64);
  assert.throws(() => validateDebertaManifest(altered), {code: 'INVALID_ASSET'});
  let rejectBody!: (reason?: unknown) => void, started!: () => void;
  const begun = new Promise<void>((resolve) => {
    started = resolve;
  });
  const controller = new AbortController(),
    reason = Error('stop');
  const pending = readDebertaManifestResponse(
    responseFromReader(() => {
      started();
      return new Promise((_, reject) => {
        rejectBody = reject;
      });
    }) as unknown as Response,
    controller.signal
  );
  await begun;
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  rejectBody(Error('late'));
  await new Promise((resolve) => setImmediate(resolve));
});

function session() {
  let scopes = 1;
  let releaseGate: Promise<unknown> | undefined;
  const events: string[] = [],
    lifecycle = {disposed: false, loss: null};
  const runtime = {
    beginScope() {
      scopes++;
    },
    endScope() {
      scopes--;
    },
    dispose() {
      lifecycle.disposed = true;
      events.push('disposed');
    },
    cpu() {
      return 'cpu';
    },
    getGlobalFunc() {
      return () => events.push('clear');
    },
    empty() {
      return {
        copyFrom() {
          return this;
        },
        toRawBytes() {
          return new Uint8Array([0, 60, 0, 64, 0, 66]);
        }
      };
    }
  };
  const gpu = {
      pushErrorScope() {},
      async popErrorScope() {
        return null;
      }
    },
    device = {
      async sync() {
        if (releaseGate) await releaseGate;
      }
    };
  const model = new ExperimentalDebertaSession({
    runtime,
    device,
    gpu,
    encoder: () => ({}),
    head: () => ({}),
    encoderWeights: [],
    headWeights: [],
    tokenizer,
    uncaptured: [],
    lifecycle
  } as unknown as ConstructorParameters<typeof ExperimentalDebertaSession>[0]);
  return {
    model,
    events,
    get scopes() {
      return scopes;
    },
    hold(promise: Promise<unknown>) {
      releaseGate = promise;
    }
  };
}

test('serializes decisions, pool release and disposal using submitted snapshots', async () => {
  const x = session();
  let resume!: () => void;
  x.hold(
    new Promise<void>((resolve) => {
      resume = resolve;
    })
  );
  const r = request(),
    first = x.model.decide(r);
  r.questions[0]!.options[0] = 'changed';
  const release = x.model.releaseCachedMemory(),
    dispose = x.model.dispose();
  await assert.rejects(x.model.decide(request()), {code: 'SESSION_CLOSED'});
  await Promise.resolve();
  assert.deepEqual(x.events, []);
  resume();
  const [answer] = await Promise.all([first, release, dispose]);
  assert.equal(answer.answers[0]!.probabilities[0]!.option, 'red');
  assert.deepEqual(x.events, ['clear', 'disposed']);
  assert.equal(x.scopes, 0);
});
