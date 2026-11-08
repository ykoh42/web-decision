import assert from 'node:assert/strict';
import test from 'node:test';
import {prepareKananaSystemOne} from '../src/kanana/protocol.mjs';
import {KANANA_DOCUMENT_INSTRUCTION, KANANA_MAX_LENGTH} from '../src/kanana/truncation.mjs';
import {prepareKananaCandidates, truncateKananaQueryWithCandidateFallback} from '../src/kanana/candidates.mjs';
import {renderKananaChatTemplate} from '../src/kanana/chat-template.mjs';

const SPECIAL_TOKENS = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|image_pad|>', '<|vision_end|>'];
const CHAR_ID_OFFSET = 0x110000;
const SPECIAL_IDS = new Map(SPECIAL_TOKENS.map((token, index) => [token, index + 10]));
const TOKENS_BY_ID = new Map([...SPECIAL_IDS].map(([token, id]) => [id, token]));
const BOS_ID = 1;
TOKENS_BY_ID.set(BOS_ID, '<|bos|>');

function charTokenizer() {
  return {
    encode(text, options = {}) {
      const ids = [];
      for (let offset = 0; offset < text.length;) {
        const special = SPECIAL_TOKENS.find((token) => text.startsWith(token, offset));
        if (special) {
          ids.push(SPECIAL_IDS.get(special));
          offset += special.length;
          continue;
        }

        const codePoint = text.codePointAt(offset);
        const character = String.fromCodePoint(codePoint);
        ids.push(CHAR_ID_OFFSET + codePoint);
        offset += character.length;
      }
      if (options.add_special_tokens !== false) ids.unshift(BOS_ID);
      return {ids};
    },
    decode(ids) {
      return ids.map((id) => TOKENS_BY_ID.get(id) ?? String.fromCodePoint(id - CHAR_ID_OFFSET)).join('');
    }
  };
}

function preparedQuestions(questions) {
  return prepareKananaSystemOne({state: 'state', questions});
}

function singleCandidate(text) {
  return {
    questionEntries: [{id: 'route', candidateKeys: ['route'], candidateTexts: [text]}]
  };
}

function candidateRequest({state = '', candidate}) {
  return prepareKananaSystemOne({
    state,
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

function countedTokenizer() {
  const tokenizer = charTokenizer();
  let encodeCalls = 0;
  return {
    encodeCalls: () => encodeCalls,
    encode(...args) {
      encodeCalls++;
      return tokenizer.encode(...args);
    },
    decode(...args) {
      return tokenizer.decode(...args);
    }
  };
}

test('prepares each question and candidate in its existing order', () => {
  const tokenizer = charTokenizer();
  const prepared = preparedQuestions([
    [
      '10',
      {
        type: 'choice',
        criteria: new Map([
          ['2', 'second'],
          ['10', 'tenth']
        ])
      }
    ],
    ['2', {type: 'score', criteria: ['low', 'middle', 'high']}]
  ]);

  const result = prepareKananaCandidates(tokenizer, prepared);

  assert.deepEqual(
    result.map(({id}) => id),
    ['10', '2']
  );
  assert.deepEqual(result[0].candidateKeys, ['2', '10']);
  assert.deepEqual(result[0].candidateTexts, ['2: second', '10: tenth']);
  assert.deepEqual(
    result[0].candidateDocuments.map(({key, text}) => [key, text]),
    [
      ['2', '2: second'],
      ['10', '10: tenth']
    ]
  );
  assert.deepEqual(result[1].candidateKeys, ['0', '1', '2']);
  assert.deepEqual(result[1].candidateTexts, ['low', 'middle', 'high']);
  assert.equal(result[0].candidateDocuments[0].messages[0].content[0].text, KANANA_DOCUMENT_INSTRUCTION);
  assert.equal(result[0].candidateDocuments[0].inputIds.length, result[0].candidateDocuments[0].tokenCount);
  assert.equal(result[0].candidateDocuments[0].inputIds[0], BOS_ID);
  assert.equal(renderKananaChatTemplate(result[0].candidateDocuments[0].messages).includes('2: second'), true);
});

test('truncates query state before considering candidate truncation', () => {
  const tokenizer = charTokenizer();
  const state = 's'.repeat(20_000);
  const candidate = 'c'.repeat(1_000);
  const prepared = candidateRequest({state, candidate});

  const result = truncateKananaQueryWithCandidateFallback(tokenizer, prepared, 'route');

  assert.equal(result.truncated, true);
  assert.ok(result.messages[1].content[1].text.length < state.length);
  assert.ok(result.prompt.includes(`"candidate":"${candidate}"`));
});

test('shortens the candidate only after state truncation cannot fit protected query content', () => {
  const tokenizer = charTokenizer();
  const candidate = `${'a'.repeat(15_000)}Z${'b'.repeat(6_000)}`;
  const prepared = candidateRequest({state: 'short state', candidate});

  const result = truncateKananaQueryWithCandidateFallback(tokenizer, prepared, 'route');

  assert.equal(result.tokenCount, KANANA_MAX_LENGTH);
  assert.equal(result.truncated, true);
  assert.equal(result.messages[1].content[1].text, 'short state');
  assert.ok(result.prompt.includes(`Z${'b'.repeat(40)}`));
  assert.ok(result.prompt.includes(`${'b'.repeat(40)}","source":"structured"`));
  assert.ok(!result.prompt.includes(`"candidate":"${candidate}"`));
});

test('does not attempt candidate fallback when either truncation flag is disabled', () => {
  const candidate = 'c'.repeat(KANANA_MAX_LENGTH + 500);
  const prepared = candidateRequest({candidate});
  const strictTokenizer = countedTokenizer();
  const candidateStrictTokenizer = countedTokenizer();

  assert.throws(
    () =>
      truncateKananaQueryWithCandidateFallback(strictTokenizer, prepared, 'route', [], {
        truncateInputs: false
      }),
    {code: 'input_too_long'}
  );
  assert.equal(strictTokenizer.encodeCalls(), 1);

  assert.throws(
    () =>
      truncateKananaQueryWithCandidateFallback(candidateStrictTokenizer, prepared, 'route', [], {
        truncateQuestionCandidates: false
      }),
    {code: 'input_too_long'}
  );
  assert.equal(candidateStrictTokenizer.encodeCalls(), 2);
});

test('returns an empty candidate document list without inventing entries', () => {
  const result = prepareKananaCandidates(charTokenizer(), {
    questionEntries: [{id: 'empty', candidateKeys: [], candidateTexts: []}]
  });

  assert.equal(result.length, 1);
  assert.deepEqual(result[0].candidateDocuments, []);
  assert.deepEqual(result[0].candidateKeys, []);
  assert.deepEqual(result[0].candidateTexts, []);
});

test('rejects whitespace-only candidates using Python str.strip whitespace', () => {
  const tokenizer = charTokenizer();
  const whitespaceOnly = ['\u0085', '\u001c', '\u001d', '\u001e', '\u001f', '\u00a0', '\u2003', '\u3000'];

  for (const text of whitespaceOnly) {
    assert.throws(
      () => prepareKananaCandidates(tokenizer, singleCandidate(text)),
      {code: 'empty_candidate'},
      `expected ${JSON.stringify(text)} to be treated as Python whitespace`
    );
  }
});

test('preserves Python whitespace around nonempty candidate documents', () => {
  const tokenizer = charTokenizer();
  const text = '\u0085\u001c\u2003candidate\u00a0\u3000';
  const result = prepareKananaCandidates(tokenizer, singleCandidate(text));

  assert.equal(result[0].candidateDocuments[0].text, text);
  assert.equal(result[0].candidateDocuments[0].messages[1].content[0].text, text);
});

test('accepts a candidate document exactly at max_length and truncates one token over', () => {
  const tokenizer = charTokenizer();
  const documentOverhead = documentTokenCount(tokenizer, 'x') - 1;
  const exactText = 'a'.repeat(KANANA_MAX_LENGTH - documentOverhead);
  const exact = prepareKananaCandidates(tokenizer, singleCandidate(exactText));

  assert.equal(exact[0].candidateDocuments[0].tokenCount, KANANA_MAX_LENGTH);
  assert.equal(exact[0].candidateDocuments[0].truncated, false);
  assert.equal(exact[0].candidateDocuments[0].messages[1].content[0].text, exactText);

  const overText = `${exactText}Z`;
  const over = prepareKananaCandidates(tokenizer, singleCandidate(overText));
  const document = over[0].candidateDocuments[0];

  assert.equal(document.tokenCount, KANANA_MAX_LENGTH);
  assert.equal(document.truncated, true);
  assert.equal(document.messages[1].content[0].text, exactText);
  assert.deepEqual(over[0].candidateTexts, [overText]);
});

function documentTokenCount(tokenizer, text) {
  const messages = [
    {role: 'system', content: [{type: 'text', text: KANANA_DOCUMENT_INSTRUCTION}]},
    {role: 'user', content: [{type: 'text', text}]}
  ];
  return tokenizer.encode(renderKananaChatTemplate(messages)).ids.length;
}

test('uses the configured length without imposing the model default as a lower cap', () => {
  const tokenizer = charTokenizer();
  const text = 'd'.repeat(25_000);
  const result = prepareKananaCandidates(tokenizer, singleCandidate(text), {maxLength: 30_000});
  const document = result[0].candidateDocuments[0];

  assert.equal(document.messages[1].content[0].text, text);
  assert.equal(document.truncated, false);
  assert.ok(document.tokenCount > KANANA_MAX_LENGTH);
});

test('rejects duplicate document token IDs when truncation is disabled', () => {
  const collisionTokenizer = {encode: () => ({ids: [BOS_ID, 7]})};
  const prepared = {
    questionEntries: [
      {
        id: 'route',
        candidateKeys: ['first', 'second'],
        candidateTexts: ['first text', 'different text']
      }
    ]
  };

  assert.throws(() => prepareKananaCandidates(collisionTokenizer, prepared, {truncateInputs: false}), {
    code: 'candidate_token_collision'
  });
});

test('preserves duplicate token sequences when truncation is enabled', () => {
  const collisionTokenizer = {encode: () => ({ids: [BOS_ID, 7]})};
  const prepared = {
    questionEntries: [
      {
        id: 'route',
        candidateKeys: ['first', 'second'],
        candidateTexts: ['first text', 'different text']
      }
    ]
  };
  const result = prepareKananaCandidates(collisionTokenizer, prepared, {truncateInputs: true});

  assert.deepEqual(
    result[0].candidateDocuments.map(({key, inputIds}) => [key, inputIds]),
    [
      ['first', [BOS_ID, 7]],
      ['second', [BOS_ID, 7]]
    ]
  );
});

test('rejects mismatched candidate key and text lists without dropping either side', () => {
  assert.throws(
    () =>
      prepareKananaCandidates(charTokenizer(), {
        questionEntries: [{id: 'route', candidateKeys: ['first'], candidateTexts: []}]
      }),
    {code: 'candidate_count'}
  );
});
