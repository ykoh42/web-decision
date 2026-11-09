import assert from 'node:assert/strict';
import test from 'node:test';
import {protocol, tokenizer, truncation} from './kanana-core-modules.mjs';

const {prepareKananaSystemOne, renderKananaQueryInput} = protocol;
const {
  KANANA_DOCUMENT_INSTRUCTION,
  KANANA_MAX_LENGTH,
  truncateKananaDocument,
  truncateKananaQuery,
  truncateKananaQuestionCandidate
} = truncation;
const {encodeKananaQuery} = tokenizer;

const IMAGE_TOKEN = '<|image_pad|>';
const SPECIAL_TOKENS = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', IMAGE_TOKEN, '<|vision_end|>'];
const CHAR_ID_OFFSET = 0x110000;
const SPECIAL_IDS = new Map(SPECIAL_TOKENS.map((token, index) => [token, index + 10]));
const TOKENS_BY_ID = new Map([...SPECIAL_IDS].map(([token, id]) => [id, token]));
const BOS_ID = 1;
TOKENS_BY_ID.set(BOS_ID, '<|bos|>');

function charTokenizer() {
  return {
    encode(text: string, options: {add_special_tokens?: boolean} = {}) {
      const ids = [];
      for (let offset = 0; offset < text.length;) {
        const special = SPECIAL_TOKENS.find((token) => text.startsWith(token, offset));
        if (special) {
          ids.push(SPECIAL_IDS.get(special)!);
          offset += special.length;
          continue;
        }

        const codePoint = text.codePointAt(offset)!;
        const character = String.fromCodePoint(codePoint);
        ids.push(CHAR_ID_OFFSET + codePoint);
        offset += character.length;
      }
      if (options.add_special_tokens !== false) ids.unshift(BOS_ID);
      return {ids};
    },
    decode(ids: number[]) {
      return ids.map((id) => TOKENS_BY_ID.get(id) ?? String.fromCodePoint(id - CHAR_ID_OFFSET)).join('');
    }
  };
}

function boundaryTokenizer() {
  const tokens = new Map<string | number, string | number>([...SPECIAL_IDS, [100, '[u]\n'], [101, 'xb'], [102, 'ab']]);
  const tokensByValue = new Map([...tokens].map(([id, token]) => [token, id]));
  const boundaryTokens = ['[u]\n', 'xb', 'ab'];

  return {
    encode(text: string, options: {add_special_tokens?: boolean} = {}) {
      const ids = [];
      for (let offset = 0; offset < text.length;) {
        const token = [...boundaryTokens, ...SPECIAL_TOKENS].find((value) => text.startsWith(value, offset));
        if (token) {
          ids.push((tokensByValue.get(token) ?? SPECIAL_IDS.get(token)) as number);
          offset += token.length;
          continue;
        }

        const codePoint = text.codePointAt(offset)!;
        const character = String.fromCodePoint(codePoint);
        ids.push(CHAR_ID_OFFSET + codePoint);
        offset += character.length;
      }
      if (options.add_special_tokens !== false) ids.unshift(BOS_ID);
      return {ids};
    },
    decode(ids: number[]) {
      return ids.map((id) => tokens.get(id) ?? String.fromCodePoint(id - CHAR_ID_OFFSET)).join('');
    }
  };
}

function queryInput({
  state = '',
  images = [],
  instructions = 'Select a route'
}: {state?: string; images?: string[]; instructions?: string} = {}) {
  const prepared = prepareKananaSystemOne({
    state,
    images,
    questions: {
      route: {
        type: 'choice',
        instructions,
        criteria: new Map([
          ['billing', 'Billing'],
          ['support', 'Support']
        ])
      }
    }
  });
  return renderKananaQueryInput(prepared, 'route');
}

function candidateRequest({
  state = '',
  candidate,
  images = []
}: {state?: string; candidate?: unknown; images?: string[]} = {}) {
  return prepareKananaSystemOne({
    state,
    images,
    questions: {
      route: {
        type: 'choice',
        instructions: {candidate, source: 'structured'},
        criteria: new Map([
          ['billing', 'Billing'],
          ['support', 'Support']
        ])
      }
    }
  });
}

function fit(
  input: ReturnType<typeof queryInput>,
  tokenizer: ReturnType<typeof charTokenizer>,
  imageTokenThw: number[][] = [],
  options: Parameters<typeof truncateKananaQuery>[3] = {}
) {
  return truncateKananaQuery(tokenizer, input.messages, imageTokenThw, {
    editableUserTextIndexes: input.editableUserTextIndexes,
    ...options
  });
}

function stateText(result: ReturnType<typeof truncateKananaQuery>, index: number | undefined) {
  return (result.messages[1]!.content as {text: string}[])[index!]!.text;
}

test('accepts an exact model-length query without changing its state', () => {
  const tokenizer = charTokenizer();
  const empty = queryInput();
  const overhead = fit(empty, tokenizer).tokenCount;
  const input = queryInput({state: 'x'.repeat(KANANA_MAX_LENGTH - overhead)});
  const result = fit(input, tokenizer);

  assert.equal(result.tokenCount, KANANA_MAX_LENGTH);
  assert.equal(result.truncated, false);
  assert.equal(stateText(result, input.editableUserTextIndexes[0]).length, KANANA_MAX_LENGTH - overhead);
});

test('truncates a query one token over the limit by preserving the state head and tail', () => {
  const tokenizer = charTokenizer();
  const empty = queryInput();
  const overhead = fit(empty, tokenizer).tokenCount;
  const state = `${'a'.repeat(KANANA_MAX_LENGTH - overhead)}Z`;
  const input = queryInput({state});
  const result = fit(input, tokenizer);
  const outputState = stateText(result, input.editableUserTextIndexes[0]);

  assert.equal(result.tokenCount, KANANA_MAX_LENGTH);
  assert.equal(result.truncated, true);
  assert.equal(outputState.length, state.length - 1);
  assert.ok(outputState.startsWith('a'));
  assert.ok(outputState.endsWith('Z'));
});

test('truncates adjacent message text parts using their original token boundaries', () => {
  const tokenizer = boundaryTokenizer();
  const prepared = prepareKananaSystemOne({
    stateFormat: 'messages',
    state: [
      {
        role: 'u',
        content: [
          {type: 'text', text: 'x'},
          {type: 'text', text: 'a'},
          {type: 'text', text: 'b'}
        ]
      }
    ],
    questions: {route: {type: 'choice', criteria: {only: 'Only'}}}
  });
  const input = renderKananaQueryInput(prepared, 'route');
  const editable = input.editableUserTextIndexes;
  const editableText = editable.map((index) => (input.messages[1].content[index] as {text: string}).text);
  const emptyMessages = structuredClone(input.messages);
  for (const index of editable) (emptyMessages[1].content[index] as {text: string}).text = '';
  const protectedTokens = encodeKananaQuery(tokenizer, emptyMessages).tokenCount;
  const original = encodeKananaQuery(tokenizer, input.messages);

  assert.deepEqual(editableText, ['[u]\n', 'x', 'a', 'b']);
  assert.equal(original.tokenCount, protectedTokens + 3);

  const result = truncateKananaQuery(tokenizer, input.messages, [], {
    editableUserTextIndexes: editable,
    maxLength: protectedTokens + 2
  });

  assert.equal(result.tokenCount, protectedTokens + 2);
  assert.equal(result.retainedStateTokens, 3);
  assert.deepEqual(
    editable.map((index) => (result.messages[1]!.content as {text: string}[])[index]!.text),
    ['[u]\n', 'x', '', 'b']
  );
  assert.ok(result.prompt.includes('[u]\nxb'));
});

test('rejects an overlength prompt when pinned truncation is disabled', () => {
  const tokenizer = charTokenizer();
  const empty = queryInput();
  const overhead = fit(empty, tokenizer).tokenCount;
  const input = queryInput({state: 'x'.repeat(KANANA_MAX_LENGTH - overhead + 1)});

  assert.throws(
    () => fit(input, tokenizer, [], {truncateInputs: false}),
    (error) => {
      assert.deepEqual(Object.keys(error as object), ['name', 'code', 'maxLength', 'tokenCount']);
      assert.equal((error as {code: string}).code, 'input_too_long');
      return true;
    }
  );
});

test('preserves every expanded image token while trimming state text', () => {
  const tokenizer = charTokenizer();
  const grid = [1, 2, 2];
  const empty = queryInput({images: ['photo.png']});
  const overhead = fit(empty, tokenizer, [grid]).tokenCount;
  const input = queryInput({state: 'x'.repeat(KANANA_MAX_LENGTH - overhead + 1), images: ['photo.png']});
  const result = fit(input, tokenizer, [grid]);

  assert.equal(result.tokenCount, KANANA_MAX_LENGTH);
  assert.equal(result.inputIds.filter((id) => id === SPECIAL_IDS.get(IMAGE_TOKEN)).length, 4);
  assert.equal(result.prompt.includes(`${IMAGE_TOKEN.repeat(4)}`), true);
  assert.equal((result.messages[1]!.content as {type: string}[]).filter((part) => part.type === 'image').length, 1);
});

test('errors when protected options and images exceed the limit after state is removed', () => {
  const tokenizer = charTokenizer();
  const images = Array.from({length: 10}, (_, index) => `photo-${index}.png`);
  const imageTokenThw = images.map(() => [1, 40, 40]);
  const input = queryInput({state: 'short state', images, instructions: 'p'.repeat(5_000)});
  const emptyState = queryInput({images, instructions: 'p'.repeat(5_000)});

  assert.throws(() => fit(emptyState, tokenizer, imageTokenThw, {editableUserTextIndexes: []}), {
    code: 'input_too_long'
  });
  assert.throws(() => fit(input, tokenizer, imageTokenThw), {code: 'input_too_long'});
});

test('honors the supplied model config length without a smaller adapter cap', () => {
  const tokenizer = charTokenizer();
  const input = queryInput({state: 'x'.repeat(25_000)});
  const result = fit(input, tokenizer, [], {maxLength: 30_000});

  assert.equal(result.tokenCount, 25_000 + fit(queryInput(), tokenizer).tokenCount);
  assert.equal(result.truncated, false);
});

test('accepts an exact-length structured question candidate unchanged', () => {
  const tokenizer = charTokenizer();
  const prepared = candidateRequest({candidate: 'candidate text'});
  const input = renderKananaQueryInput(prepared, 'route');
  const original = encodeKananaQuery(tokenizer, input.messages);
  const result = truncateKananaQuestionCandidate(tokenizer, prepared, 'route', [], {
    maxLength: original.tokenCount
  });

  assert.equal(result.tokenCount, original.tokenCount);
  assert.equal(result.truncated, false);
  assert.ok(result.prompt.includes('"candidate":"candidate text"'));
});

test('truncates an overlength structured candidate by preserving its head and tail', () => {
  const tokenizer = charTokenizer();
  const imageTokenThw = [[1, 2, 2]];
  const empty = candidateRequest({candidate: '', images: ['photo.png']});
  const emptyInput = renderKananaQueryInput(empty, 'route');
  const overhead = encodeKananaQuery(tokenizer, emptyInput.messages, imageTokenThw).tokenCount;
  const candidate = `${'a'.repeat(KANANA_MAX_LENGTH - overhead)}Z`;
  const prepared = candidateRequest({candidate, images: ['photo.png']});
  const input = renderKananaQueryInput(prepared, 'route');
  const original = encodeKananaQuery(tokenizer, input.messages, imageTokenThw);
  const result = truncateKananaQuestionCandidate(tokenizer, prepared, 'route', imageTokenThw);

  assert.equal(original.tokenCount, KANANA_MAX_LENGTH + 1);
  assert.equal(result.tokenCount, KANANA_MAX_LENGTH);
  assert.equal(result.truncated, true);
  assert.equal(result.inputIds.filter((id) => id === SPECIAL_IDS.get(IMAGE_TOKEN)).length, 4);
  assert.equal(result.prompt.includes(`${IMAGE_TOKEN.repeat(4)}`), true);
  assert.ok(result.prompt.includes('Z","source":"structured"'));
  assert.ok(result.prompt.includes('\n\nOptions:\nA: billing: Billing\nB: support: Support'));
  assert.ok(result.prompt.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'));
  assert.ok(result.prompt.includes('"source":"structured"'));
});

test('candidate fallback obeys its flag and requires a structured candidate field', () => {
  const tokenizer = charTokenizer();
  const prepared = candidateRequest({candidate: 'c'.repeat(21_000)});

  assert.throws(
    () => truncateKananaQuestionCandidate(tokenizer, prepared, 'route', [], {truncateQuestionCandidates: false}),
    {code: 'input_too_long'}
  );
  assert.throws(() => truncateKananaQuestionCandidate(tokenizer, prepared, 'route', [], {truncateInputs: false}), {
    code: 'input_too_long'
  });
  const unstructured = prepareKananaSystemOne({
    state: '',
    questions: {
      route: {
        type: 'choice',
        instructions: 'c'.repeat(21_000),
        criteria: new Map([
          ['billing', 'Billing'],
          ['support', 'Support']
        ])
      }
    }
  });
  assert.throws(() => truncateKananaQuestionCandidate(tokenizer, unstructured, 'route'), {code: 'input_too_long'});
});

test('candidate fallback errors when protected state and options still exceed the limit', () => {
  const tokenizer = charTokenizer();
  const prepared = candidateRequest({state: 's'.repeat(21_000), candidate: 'short'});

  assert.throws(() => truncateKananaQuestionCandidate(tokenizer, prepared, 'route'), {code: 'input_too_long'});
});

test('candidate fallback rejects a zero-token candidate budget at the exact protected boundary', () => {
  const tokenizer = charTokenizer();
  const empty = candidateRequest({candidate: ''});
  const protectedLength = encodeKananaQuery(tokenizer, renderKananaQueryInput(empty, 'route').messages).tokenCount;
  const prepared = candidateRequest({candidate: 'one token'});

  assert.throws(() => truncateKananaQuestionCandidate(tokenizer, prepared, 'route', [], {maxLength: protectedLength}), {
    code: 'input_too_long',
    message: 'No token budget remains for the Question candidate.'
  });
});

test('accepts an exact-length candidate document unchanged', () => {
  const tokenizer = charTokenizer();
  const original = truncateKananaDocument(tokenizer, 'A candidate document');
  const exact = truncateKananaDocument(tokenizer, 'A candidate document', [], {maxLength: original.tokenCount});

  assert.equal(exact.tokenCount, original.tokenCount);
  assert.equal(exact.truncated, false);
  assert.equal((exact.messages[0]!.content as {text: string}[])[0]!.text, KANANA_DOCUMENT_INSTRUCTION);
  assert.equal((exact.messages[1]!.content as {text: string}[])[0]!.text, 'A candidate document');
});

test('truncates an overlength candidate document to its longest fitting prefix', () => {
  const tokenizer = charTokenizer();
  const overhead = truncateKananaDocument(tokenizer, 'x').tokenCount - 1;
  const value = `${'a'.repeat(KANANA_MAX_LENGTH - overhead)}Z`;
  const result = truncateKananaDocument(tokenizer, value);

  assert.equal(result.tokenCount, KANANA_MAX_LENGTH);
  assert.equal(result.truncated, true);
  assert.equal((result.messages[1]!.content as {text: string}[])[0]!.text, value.slice(0, -1));
  assert.equal((result.messages[0]!.content as {text: string}[])[0]!.text, KANANA_DOCUMENT_INSTRUCTION);
});

test('candidate documents preserve the protected instruction and honor configured limits', () => {
  const tokenizer = charTokenizer();
  const longInstruction = 'i'.repeat(KANANA_MAX_LENGTH + 1);
  const longDocument = 'd'.repeat(KANANA_MAX_LENGTH + 1);

  assert.throws(() => truncateKananaDocument(tokenizer, 'candidate', [], {instruction: longInstruction}), {
    code: 'input_too_long'
  });
  assert.throws(() => truncateKananaDocument(tokenizer, longDocument, [], {truncateInputs: false}), {
    code: 'input_too_long'
  });
  const value = 'd'.repeat(25_000);
  const result = truncateKananaDocument(tokenizer, value, [], {maxLength: 30_000});
  assert.equal((result.messages[1]!.content as {text: string}[])[0]!.text, value);
  assert.equal(result.truncated, false);
  assert.ok(result.tokenCount > KANANA_MAX_LENGTH);
});
