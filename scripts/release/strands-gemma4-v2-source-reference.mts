import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

export const GEMMA_V2_CPU_REFERENCE_PATH = 'tests/fixtures/strands-gemma4-v2-upstream-cpu-reference.json';
export const GEMMA_V2_CPU_REFERENCE_SHA256 = '396e7be2c54225de08e8bf38a9a5105f14f98af3f6f257a7b4448ae44d768a2c';
const pinned = {
  'text-1024': {
    tokens: 1024,
    request: 'd7573e5b15f1469efe2dcd80105050843f37f432451bf4b05b617bc6d65aec7b',
    tokenIds: '9769f2d65c2da1b2cbad97cb03ca91155700344cc802732f884c59055a453635',
    result: 'c69ed29a897605d47061a76abf9772e1470c3a4b6d61408332b7a98e0f36a3a2'
  },
  'text-32768': {
    tokens: 32768,
    request: 'ae56406dcf0d61f44040bc9f39b1a476812b187cad74ef436ea3f2c4c95ba9d4',
    tokenIds: '40d00eff85c5d558348cc124ac1bc1742f5132b40dc4c746739533c274cd6194',
    result: '698422709834f58bea9e96d72a191391899b5c4b72d75cf141f8826671053991'
  }
} as const;
const hash = (value: string | Uint8Array) => createHash('sha256').update(value).digest('hex');

const capturedPins = {
  'image-single': {
    tokens: 337,
    request: 'bde1b1f65ae9f540e0691b0ea139c75f108ee351e0cabd0d46c3792f49a9d8db',
    tokenIds: '3283f212bf0880fe9675e7f7875847176c2b3a25d835b888bd7a75ac8bab74cc',
    result: '1ab4ef42df19fccc148cb5156460db0bd038a8150ac20cf7657091e21c5e8165',
    images: ['4f600563fc21c334bf79b2edbf703922ed3c796e257e2662429c7c35ac11b7cf']
  },
  'image-multiple': {
    tokens: 598,
    request: 'b8111c2fb2d5f565d8b65d7801f02553518ec02780a0fa0d97b5dc941073be9e',
    tokenIds: 'f5d5194e0f50242a2bda4acd75e7661d04e2bef6b8d34e7212ef008bf4ff6e50',
    result: 'de534568cc4b58cb481e0ba518a005bf0762f9d5890188bf99b18c7f847395cf',
    images: [
      '7c523339849eb22f17b912ea25b65de69ce6236e8a597ccc896790a923e24685',
      '3545e07c4e1e17714411dd6f5cd96d0a55de0269cd7345937bb08d394aa3deb9'
    ]
  },
  'multi-kind': {
    tokens: 227,
    request: '3c8969f80371c3d81acd90e0ceac9af64f4d728d2aaf846cb7bd675c79e80737',
    tokenIds: '78fdc0dd5719b36b0fa5a708a942dbf88e5e78c44cfaeaa9aec4c801b2194071',
    result: '476d1cc579816928f26ed45ee78d06d8a3aedba74bfbc3960781350a3ca7f8ff',
    images: []
  }
} as const;

export type CpuCase = {
  requestSha256: string;
  requestJson: string;
  questionName: string;
  imageCount: number;
  inputTokens: number;
  outputTokens: number;
  result: unknown;
};

type CapturedCase = {
  inputTokens: number;
  requestSha256: string;
  requestJson: string;
  tokensSha256: string;
  tokensJson: string;
  resultSha256: string;
  resultJson: string;
  images: {sha256: string; bytes: number}[];
};

function validateCapturedImages(
  entry: CapturedCase,
  request: {images?: string[]},
  tokens: {state: number[]; images: {start: number; end: number}[]},
  expected: readonly string[]
) {
  assert.equal(tokens.images.length, expected.length);
  assert.equal(entry.images.length, expected.length);
  assert.equal((request.images ?? []).length, expected.length);
  for (const [index, digest] of expected.entries()) {
    const image = Buffer.from(request.images![index]!, 'base64');
    assert.equal(hash(image), digest);
    assert.equal(entry.images[index]!.sha256, digest);
    assert.equal(entry.images[index]!.bytes, image.length);
    assert.ok(tokens.images[index]!.start >= 0 && tokens.images[index]!.end <= tokens.state.length);
    assert.ok(tokens.images[index]!.start < tokens.images[index]!.end);
  }
}

function validateCapturedCases(cases: Record<string, CapturedCase>) {
  assert.deepEqual(Object.keys(cases).sort(), Object.keys(capturedPins).sort());
  const reference: Record<string, CpuCase> = {};
  for (const [name, expected] of Object.entries(capturedPins)) {
    const entry = cases[name]!;
    assert.equal(entry.inputTokens, expected.tokens);
    for (const [actual, data, digest] of [
      [entry.requestSha256, entry.requestJson, expected.request],
      [entry.tokensSha256, entry.tokensJson, expected.tokenIds],
      [entry.resultSha256, entry.resultJson, expected.result]
    ]) {
      assert.equal(actual, digest);
      assert.equal(hash(data!), digest);
    }
    const request = JSON.parse(entry.requestJson) as {images?: string[]; questions: Record<string, unknown>};
    const tokens = JSON.parse(entry.tokensJson) as {
      state: number[];
      questions: number[][];
      images: {start: number; end: number}[];
    };
    const result = JSON.parse(entry.resultJson) as {
      answers: Record<string, {type: string}>;
      usage: {input_tokens: number; output_tokens: number};
    };
    const questionNames = Object.keys(request.questions);
    assert.equal(
      tokens.state.length + tokens.questions.reduce((sum, question) => sum + question.length, 0),
      expected.tokens
    );
    assert.equal(tokens.questions.length, questionNames.length);
    assert.equal(result.usage.input_tokens, expected.tokens);
    assert.equal(result.usage.output_tokens, questionNames.length);
    validateCapturedImages(entry, request, tokens, expected.images);
    for (const question of questionNames) {
      const id =
        name === 'multi-kind' ? ({refund: 'noul', urgency: 'score'} as Record<string, string>)[question] : name;
      if (!id) continue;
      assert.equal(result.answers[question]?.type, name === 'multi-kind' ? id : 'choice');
      reference[id] = {
        requestSha256: entry.requestSha256,
        requestJson: entry.requestJson,
        questionName: question,
        imageCount: entry.images.length,
        inputTokens: entry.inputTokens,
        outputTokens: questionNames.length,
        result: result.answers[question]
      };
    }
  }
  return reference;
}

/** The exact upstream Python JSON encoding for these two archived long-input requests. */
export function gemmaV2CpuRequest(repetitions: number): string {
  assert.ok(Number.isSafeInteger(repetitions) && repetitions >= 0);
  return `{"state": "${' apple'.repeat(repetitions)}", "questions": {"q": {"type": "choice", "instructions": "Choose the relevant label.", "criteria": {"yes": "yes", "no": "no"}}}}\n`;
}

/** Check collected CPU references; this is not a model or release approval. */
export function validateGemmaV2CpuReferences(bytes: Uint8Array): Record<string, CpuCase> {
  assert.equal(hash(bytes), GEMMA_V2_CPU_REFERENCE_SHA256, 'Gemma v2 CPU reference fixture differs.');
  const fixture = JSON.parse(new TextDecoder().decode(bytes)) as {
    status: string;
    sourceRevision: string;
    checkpointSha256: string;
    device: string;
    dtype: string;
    python: string;
    packages: Record<string, string>;
    imageRunReportSha256: string;
    multiKindRunReportSha256: string;
    capturedCases: Record<string, CapturedCase>;
    tokenLayout: {prefix: number[]; repeatedToken: number; suffix: number[]; question: number[]};
    cases: Record<
      string,
      {
        tokens: number;
        stateRepeatCount: number;
        requestSha256: string;
        tokensSha256: string;
        resultSha256: string;
        resultJson: string;
      }
    >;
  };
  assert.equal(fixture.status, 'collected-pending-release-review');
  assert.equal(fixture.sourceRevision, 'c3c845782095b31d438963ac2afcdfe41b49f757');
  assert.equal(fixture.checkpointSha256, '593a28a67754653c7ffffff80a5186f18966cc6262c2f4f097414c187e5fd48c');
  assert.equal(fixture.device, 'cpu');
  assert.equal(fixture.dtype, 'bfloat16');
  assert.equal(fixture.python, '3.13.13');
  assert.deepEqual(fixture.packages, {torch: '2.15.0', transformers: '5.17.0', numpy: '2.5.5', safetensors: '0.8.0'});
  assert.deepEqual(Object.keys(fixture.cases).sort(), Object.keys(pinned).sort());
  assert.equal(fixture.imageRunReportSha256, '7dda7e2d7445854305232573a06bf07681adece8441e0aaf7da98e303426d4b9');
  assert.equal(fixture.multiKindRunReportSha256, '6f21c28d09800fb1130ff82c038e2bcf63b296f6946ed1d93c00f71c31c8e080');
  const captured = validateCapturedCases(fixture.capturedCases);
  const longCases = Object.fromEntries(
    Object.entries(pinned).map(([name, expected]) => {
      const entry = fixture.cases[name]!;
      assert.equal(entry.tokens, expected.tokens);
      assert.equal(entry.stateRepeatCount, expected.tokens - 69);
      assert.equal(entry.requestSha256, expected.request);
      const requestJson = gemmaV2CpuRequest(entry.stateRepeatCount);
      assert.equal(hash(requestJson), expected.request);
      assert.equal(entry.tokensSha256, expected.tokenIds);
      const {prefix, repeatedToken, suffix, question} = fixture.tokenLayout;
      const state = [...prefix, ...Array.from({length: entry.stateRepeatCount - 1}, () => repeatedToken), ...suffix];
      assert.equal(state.length + question.length, expected.tokens);
      const tokenJson = `{"state": [${state.join(', ')}], "questions": [[${question.join(', ')}]]}\n`;
      assert.equal(hash(tokenJson), expected.tokenIds, `${name} token IDs differ from the captured source.`);
      assert.equal(entry.resultSha256, expected.result);
      assert.equal(hash(entry.resultJson), expected.result);
      const result = JSON.parse(entry.resultJson);
      assert.equal(result.usage?.input_tokens, expected.tokens);
      assert.equal(result.usage?.output_tokens, 1);
      assert.equal(result.answers?.q?.type, 'choice');
      const reference: CpuCase = {
        requestSha256: entry.requestSha256,
        requestJson,
        questionName: 'q',
        imageCount: 0,
        inputTokens: entry.tokens,
        outputTokens: 1,
        result: result.answers.q
      };
      return [name, reference];
    })
  );
  return {...longCases, ...captured};
}

/** Bind the observed CPU cases without approving the other release cases. */
export function assertGemmaV2LongReferences(
  approvedReference: unknown,
  cpuReference: ReturnType<typeof validateGemmaV2CpuReferences>
) {
  assert.ok(approvedReference && typeof approvedReference === 'object');
  const cases = (approvedReference as {cases?: Record<string, unknown>}).cases;
  assert.ok(cases && typeof cases === 'object');
  for (const [name, expected] of Object.entries(cpuReference)) {
    const actual = cases[name] as {requestSha256?: unknown; inputTokens?: unknown; result?: unknown} | undefined;
    assert.ok(actual, `Missing ${name} source reference.`);
    assert.equal(actual.requestSha256, expected.requestSha256, `${name} request differs from the CPU source.`);
    assert.equal(actual.inputTokens, expected.inputTokens, `${name} input length differs from the CPU source.`);
    assert.deepEqual(actual.result, expected.result, `${name} result differs from the CPU source.`);
  }
}
