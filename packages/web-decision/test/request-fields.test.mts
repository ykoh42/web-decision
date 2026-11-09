import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {validateDecisionFields, validateRequestFields} from '../src/request-fields.mjs';
import {KEV_MAX_OPTIONS, packKevRequest} from '../src/kev.js';
import {packKev08Request} from '../src/kev08.js';
import {packMapikaRequest} from '../src/mapika.js';
import {packDebertaRequest} from '../src/deberta/packing.mjs';
import {snapshotChoiceQuestion} from '../src/strands/protocol.mjs';
import type {KevTokenizer} from '../src/kev.js';

type TestQuestion = {instructions: string; options: string[]; [field: string]: unknown};
type TestRequest = {state: string; questions: TestQuestion[]; [field: string]: unknown};
const makeRequest = (): TestRequest => ({state: 'text', questions: [{instructions: 'Choose', options: ['A', 'B']}]});
const isInvalid = {code: 'INVALID_INPUT'};

test('documented ordinary, frozen, null-prototype and cross-realm requests remain accepted', () => {
  const req = makeRequest();
  validateRequestFields(req);
  validateRequestFields(Object.freeze({state: 'text', questions: Object.freeze([Object.freeze(req.questions[0])])}));
  validateRequestFields(Object.assign(Object.create(null), req));
  validateRequestFields(vm.runInNewContext('({state:"text",questions:[{instructions:"Choose",options:["A","B"]}]})'));
  validateDecisionFields(undefined);
  validateDecisionFields(null);
  validateDecisionFields({signal: new AbortController().signal});
});

test('unsupported request, question, and decision fields are rejected before evaluating their values', () => {
  for (const key of ['image', 'images', 'audio', 'video', 'type', 'criteria', 'unknown']) {
    const req = makeRequest();
    Object.defineProperty(req, key, {
      get() {
        throw Error('getter invoked');
      },
      enumerable: false
    });
    assert.throws(() => validateRequestFields(req), isInvalid);
    const q = makeRequest();
    Object.defineProperty(q.questions[0]!, key, {
      get() {
        throw Error('getter invoked');
      },
      enumerable: false
    });
    assert.throws(() => validateRequestFields(q), isInvalid);
    assert.throws(() => validateDecisionFields({[key]: true}), isInvalid);
  }
  const symbol = makeRequest();
  (symbol.questions[0] as Record<PropertyKey, unknown>)[Symbol('ignored')] = 1;
  assert.throws(() => validateRequestFields(symbol), isInvalid);
  const inherited = makeRequest();
  inherited.questions[0] = Object.assign(Object.create({type: 'score'}), inherited.questions[0]!);
  assert.throws(() => validateRequestFields(inherited), isInvalid);
  const hidden = makeRequest();
  const proto = {};
  Object.defineProperty(proto, 'criteria', {value: 'x'});
  hidden.questions[0] = Object.assign(Object.create(proto), hidden.questions[0]!);
  assert.throws(() => validateRequestFields(hidden), isInvalid);
  const sparse = makeRequest();
  sparse.questions.length = 2;
  assert.throws(() => validateRequestFields(sparse), isInvalid);
});

test('checked questions and option arrays are snapshotted before the packer reads them', () => {
  const first = [{instructions: 'Choose', options: ['A', 'B']}],
    replacement = [{...first[0], type: 'score'}];
  let reads = 0;
  const request = {
    state: 'text',
    get questions() {
      return ++reads === 1 ? first : replacement;
    }
  };
  const checked = validateRequestFields(request);
  first[0]!.options[0] = 'changed';
  assert.equal(reads, 1);
  assert.deepEqual(checked, {state: 'text', questions: [{instructions: 'Choose', options: ['A', 'B']}]});
});

test('question count is captured before reading question entries', () => {
  const questions = [
    {instructions: 'Choose', options: ['A', 'B']},
    {instructions: 'Choose', options: ['C', 'D']}
  ];
  Object.defineProperty(questions, '0', {
    get() {
      questions.length = 1;
      return {instructions: 'Choose', options: ['A', 'B']};
    }
  });
  assert.throws(() => validateRequestFields({state: 'text', questions}), isInvalid);
});

test('data-only snapshots preserve captured own and inherited values without invoking accessors', () => {
  const validation = {dataPropertiesOnly: true, allowInheritedDataProperties: true};
  const inheritedQuestion = Object.create({instructions: 'Choose', options: ['A', 'B']});
  Object.defineProperty(Object.getPrototypeOf(inheritedQuestion), Symbol('metadata'), {value: true});
  const inheritedRequest = Object.create({state: 'text'});
  inheritedRequest.questions = [inheritedQuestion];
  assert.deepEqual(
    validateRequestFields(inheritedRequest, Infinity, Infinity, Infinity, Infinity, Infinity, validation),
    {
      state: 'text',
      questions: [{instructions: 'Choose', options: ['A', 'B']}]
    }
  );

  const ownOnlyRequest = Object.create({state: 'inherited'});
  ownOnlyRequest.questions = [{instructions: 'Choose', options: ['A', 'B']}];
  assert.equal(
    validateRequestFields(ownOnlyRequest, Infinity, Infinity, Infinity, Infinity, Infinity, {
      dataPropertiesOnly: true
    }).state,
    undefined
  );

  let getterCalls = 0;
  const accessorPrototype = {};
  Object.defineProperty(accessorPrototype, 'state', {
    enumerable: true,
    get() {
      getterCalls++;
      return 'unsafe';
    }
  });
  const accessorRequest = Object.create(accessorPrototype);
  accessorRequest.questions = [{instructions: 'Choose', options: ['A', 'B']}];
  assert.throws(
    () => validateRequestFields(accessorRequest, Infinity, Infinity, Infinity, Infinity, Infinity, validation),
    isInvalid
  );
  assert.equal(getterCalls, 0);

  const unsupportedPrototype = {};
  Object.defineProperty(unsupportedPrototype, 'criteria', {value: 'hidden'});
  const unsupportedRequest = Object.create(unsupportedPrototype);
  unsupportedRequest.state = 'text';
  unsupportedRequest.questions = [{instructions: 'Choose', options: ['A', 'B']}];
  assert.throws(
    () => validateRequestFields(unsupportedRequest, Infinity, Infinity, Infinity, Infinity, Infinity, validation),
    isInvalid
  );
});

test('data-only snapshots reject symbols, accessors, missing descriptors, and cyclic prototypes', () => {
  const validation = {dataPropertiesOnly: true};
  const request = {state: 'text', questions: [{instructions: 'Choose', options: ['A', 'B']}]};
  assert.throws(
    () =>
      validateRequestFields(
        {state: 'text', [Symbol('extra')]: true},
        Infinity,
        Infinity,
        Infinity,
        Infinity,
        Infinity,
        validation
      ),
    isInvalid
  );

  let getterCalls = 0;
  const accessorRequest = {...request};
  Object.defineProperty(accessorRequest, 'state', {
    enumerable: true,
    get() {
      getterCalls++;
      return 'text';
    }
  });
  assert.throws(
    () => validateRequestFields(accessorRequest, Infinity, Infinity, Infinity, Infinity, Infinity, validation),
    isInvalid
  );
  assert.equal(getterCalls, 0);

  const target = {...request};
  const missingDescriptor = new Proxy(target, {
    getOwnPropertyDescriptor(value, key) {
      return key === 'state' ? undefined : Reflect.getOwnPropertyDescriptor(value, key);
    }
  });
  assert.throws(
    () => validateRequestFields(missingDescriptor, Infinity, Infinity, Infinity, Infinity, Infinity, validation),
    isInvalid
  );

  let cyclicPrototype: object;
  cyclicPrototype = new Proxy({}, {getPrototypeOf: () => cyclicPrototype});
  const cyclicRequest = Object.assign(Object.create(cyclicPrototype), request);
  assert.throws(
    () => validateRequestFields(cyclicRequest, Infinity, Infinity, Infinity, Infinity, Infinity, validation),
    isInvalid
  );

  const absentDescriptor = new Proxy({}, {ownKeys: () => ['ghost'], getOwnPropertyDescriptor: () => undefined});
  const inheritedRequest = Object.assign(Object.create(absentDescriptor), request);
  assert.deepEqual(
    validateRequestFields(inheritedRequest, Infinity, Infinity, Infinity, Infinity, Infinity, {
      dataPropertiesOnly: true,
      allowInheritedDataProperties: true
    }),
    request
  );
});

test('data-only preflight enforces exact token, character, option, and question boundaries before values', () => {
  const validation = {dataPropertiesOnly: true};
  const exact = validateRequestFields(
    {state: 's', questions: [{instructions: 'i', options: ['a', 'b']}]},
    1,
    4,
    9,
    2,
    2,
    validation
  );
  assert.equal(exact.questions[0]!.options.length, 2);

  // oxlint-disable-next-line unicorn/no-new-array -- Preserve sparse entries for the validation fixture.
  const options = new Array(3);
  let optionReads = 0;
  Object.defineProperty(options, '0', {
    get() {
      optionReads++;
      return 'a';
    }
  });
  const markerOverage = {state: 's', questions: [{instructions: 'i', options}]};
  assert.throws(() => validateRequestFields(markerOverage, Infinity, Infinity, 8, 3, Infinity, validation), isInvalid);
  assert.equal(optionReads, 0);
  assert.throws(() => validateRequestFields(markerOverage, 0, Infinity, Infinity, 3, Infinity, validation), isInvalid);
  assert.equal(optionReads, 0);
  assert.throws(
    () => validateRequestFields(markerOverage, Infinity, Infinity, Infinity, 2, Infinity, validation),
    isInvalid
  );
  assert.equal(optionReads, 0);

  const optionsOverCharacters = {state: 's', questions: [{instructions: 'i', options: ['a', 'b']}]};
  assert.throws(
    () => validateRequestFields(optionsOverCharacters, Infinity, 3, Infinity, 2, Infinity, validation),
    isInvalid
  );
  assert.throws(
    () =>
      validateRequestFields(
        {state: 's', questions: [{instructions: 'i', options: ['a', 'b']}]},
        Infinity,
        Infinity,
        Infinity,
        2,
        1,
        validation
      ),
    isInvalid
  );

  const nonArray = validateRequestFields(
    {state: 's', questions: [{instructions: 'i', options: 'later validation'}]},
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    validation
  );
  assert.equal(nonArray.questions[0].options, 'later validation');

  assert.throws(
    () =>
      validateRequestFields(
        {state: 's', questions: 'not an array'},
        Infinity,
        Infinity,
        Infinity,
        Infinity,
        Infinity,
        validation
      ),
    isInvalid
  );
  assert.throws(
    () => validateRequestFields([], Infinity, Infinity, Infinity, Infinity, Infinity, validation),
    isInvalid
  );
  assert.throws(
    () =>
      validateRequestFields(
        {state: 's', questions: [{instructions: 'i', options: ['a']}]},
        Infinity,
        Infinity,
        4,
        Infinity
      ),
    isInvalid
  );
  assert.throws(
    () =>
      validateRequestFields(
        {state: 's', questions: [{instructions: 'i', options: ['a', 'b', 'c']}]},
        Infinity,
        Infinity,
        8
      ),
    isInvalid
  );
  assert.equal(
    validateRequestFields({state: 's', questions: [{instructions: 'i', options: null}]}).questions[0].options,
    null
  );
});

test('model-specific question limit is checked before reading entries', () => {
  const question = {instructions: 'Choose', options: ['A', 'B']};
  const allowed = {state: 'text', questions: Array.from({length: 512}, () => question)};
  assert.equal(validateRequestFields(allowed, 512).questions.length, 512);
  const over = {state: 'text', questions: Array.from({length: 513}, () => question)};
  let reads = 0;
  Object.defineProperty(over.questions, '0', {
    get() {
      reads++;
      throw Error('question getter invoked');
    }
  });
  assert.throws(() => validateRequestFields(over, 512), isInvalid);
  assert.equal(reads, 0);
  assert.equal(
    validateRequestFields({state: 'text', questions: Array.from({length: 513}, () => question)}).questions.length,
    513
  );
});

test('Kev does not apply a fixed question count or aggregate token cap', () => {
  const question = {instructions: '', options: ['']};
  const questions = Array.from({length: 300}, () => question);
  const request = {state: '', questions};
  const checked = validateRequestFields(request, Infinity, Infinity, Infinity, KEV_MAX_OPTIONS);
  const packed = packKevRequest(
    {encode: () => ({ids: []})} as unknown as KevTokenizer,
    checked,
    {maxStateTokens: 8192},
    {
      allowEmptyInstructions: true,
      allowEmptyOptions: true
    }
  );
  assert.equal(packed.groups.length, 300);
  assert.equal(packed.ids.length, 1201);
});

test('Kev applies an explicitly configured character safety cap before tokenization', () => {
  const maxInputCharacters = 65_536;
  const request = {
    state: 's'.repeat(maxInputCharacters - 4) + '😀',
    questions: [{instructions: 'i', options: ['o']}]
  };
  const checked = validateRequestFields(request, Infinity, maxInputCharacters, Infinity, KEV_MAX_OPTIONS);
  assert.equal(
    checked.state.length + checked.questions[0]!.instructions.length + checked.questions[0]!.options[0]!.length,
    maxInputCharacters
  );

  request.questions[0]!.options[0] = '😀';
  assert.throws(
    () => validateRequestFields(request, Infinity, maxInputCharacters, Infinity, KEV_MAX_OPTIONS),
    isInvalid
  );
});

test('Kev rejects option arrays whose delimiter minimum already exceeds the packed request limit', () => {
  const first = {instructions: '', options: Array.from({length: KEV_MAX_OPTIONS}, () => '')};
  const secondOptions = Array.from({length: KEV_MAX_OPTIONS}, () => '');
  let reads = 0;
  Object.defineProperty(secondOptions, '0', {
    get() {
      reads++;
      throw Error('option getter invoked');
    }
  });
  assert.throws(
    () =>
      validateRequestFields(
        {state: '', questions: [first, {instructions: '', options: secondOptions}]},
        Infinity,
        Infinity,
        1024,
        KEV_MAX_OPTIONS
      ),
    isInvalid
  );
  assert.equal(reads, 0);
});

test('Kev preserves the upstream 255-option limit and checks packed marker budget', () => {
  const question = {instructions: '', options: Array.from({length: KEV_MAX_OPTIONS}, () => '')};
  const request = {state: '', questions: [question]};
  const checked = validateRequestFields(request, Infinity, Infinity, 1024, KEV_MAX_OPTIONS);
  const packed = packKevRequest(
    {encode: () => ({ids: []})} as unknown as KevTokenizer,
    checked,
    {maxStateTokens: 8192, maxBranchTokens: 8192, maxInputTokens: 1024, maxOptions: KEV_MAX_OPTIONS},
    {allowEmptyInstructions: true, allowEmptyOptions: true, allowDuplicateOptionsFor: new Set([0])}
  );
  assert.equal(packed.groups[0]!.options.length, KEV_MAX_OPTIONS);
  assert.equal(packed.ids.length, 3 + KEV_MAX_OPTIONS * 2);

  const tooMany = {
    state: '',
    questions: [{instructions: '', options: Array.from({length: KEV_MAX_OPTIONS + 1}, () => '')}]
  };
  let reads = 0;
  Object.defineProperty(tooMany.questions[0]!.options, '0', {
    get() {
      reads++;
      throw Error('option getter invoked');
    }
  });
  assert.throws(() => validateRequestFields(tooMany, Infinity, Infinity, 1024, KEV_MAX_OPTIONS), isInvalid);
  assert.equal(reads, 0);
});

test('Kev does not impose a character cap unless the caller configures one', () => {
  const state = 's'.repeat(65_537);
  const checked = validateRequestFields({state, questions: [{instructions: 'choose', options: ['yes']}]});
  assert.equal(checked.state.length, state.length);
});

test('sparse option entries are rejected even if Array.prototype supplies a value', () => {
  const request = makeRequest();
  // oxlint-disable-next-line unicorn/no-new-array -- Preserve sparse entries for the validation fixture.
  request.questions[0]!.options = new Array(3);
  request.questions[0]!.options[0] = 'A';
  request.questions[0]!.options[2] = 'C';
  const previous = Object.getOwnPropertyDescriptor(Array.prototype, '1');
  try {
    Object.defineProperty(Array.prototype, '1', {configurable: true, writable: true, value: 'B'});
    assert.throws(() => validateRequestFields(request), isInvalid);
  } finally {
    if (previous) Object.defineProperty(Array.prototype, '1', previous);
    else delete Array.prototype[1];
  }
});

test('model-specific option limits are checked before reading entries', () => {
  for (const count of [2, 255]) {
    const choices = Array.from({length: count}, (_, i) => `option ${i}`);
    const request = makeRequest();
    request.questions[0]!.options = choices;
    const checked = validateRequestFields(request, Infinity, Infinity, Infinity, 255);
    assert.deepEqual(checked.questions[0].options, choices);
    assert.notEqual(checked.questions[0].options, choices);
  }
  const choices = Array.from({length: 256}, (_, i) => `option ${i}`);
  let reads = 0;
  Object.defineProperty(choices, '0', {
    get() {
      reads++;
      throw Error('option getter invoked');
    }
  });
  const request = makeRequest();
  request.questions[0]!.options = choices;
  assert.throws(() => validateRequestFields(request, Infinity, Infinity, Infinity, 255), isInvalid);
  assert.equal(reads, 0);
});

test('the shared request snapshot does not impose an option cap when a model declares none', () => {
  const options = Array.from({length: 256}, (_, index) => `option ${index}`);
  const request = makeRequest();
  request.questions[0]!.options = options;
  assert.equal(validateRequestFields(request).questions[0]!.options.length, 256);
});

test('guarded existing packers keep ordinary packed output unchanged; unsupported fields fail before tokenizer', () => {
  let calls = 0;
  const tokenizer = {
    encode(text: string) {
      calls++;
      return {ids: [text === 'A' ? 3 : text === 'B' ? 4 : 5]};
    }
  };
  const labels = {
    names: Array.from({length: 255}, (_, i) => String(i)),
    ids: Array.from({length: 255}, (_, i) => i + 10),
    open: [1]
  };
  const packers: Array<(request: TestRequest) => unknown> = [
    (request) =>
      packKevRequest(tokenizer as unknown as KevTokenizer, request, {maxStateTokens: 384, maxInputTokens: 1024}),
    (request) =>
      packKev08Request(tokenizer as unknown as KevTokenizer, request, {maxStateTokens: 128, maxInputTokens: 128}),
    (request) =>
      packMapikaRequest(tokenizer, labels, request, {maxStateTokens: 512, maxInputTokens: 512, controlIds: [999]}),
    (request) => packDebertaRequest(tokenizer, request),
    (request) => snapshotChoiceQuestion(request.questions[0]!)
  ];
  for (const pack of packers) {
    const request = makeRequest(),
      before = pack(request),
      snapshot = validateRequestFields(request);
    assert.deepEqual(pack(snapshot), before);
    for (const bad of [
      {...request, image: 'ignored'},
      {...request, questions: [{...request.questions[0]!, type: 'score', criteria: 'x'}]}
    ]) {
      const count = calls;
      assert.throws(() => {
        validateRequestFields(bad);
        pack(bad);
      }, isInvalid);
      assert.equal(calls, count);
    }
  }
});
