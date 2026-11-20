import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {copyFile, link, mkdir, mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import test from 'node:test';
import {fileURLToPath} from 'node:url';
import {installQwen35TvmDispatchCapture} from '../../tests/browser/qwen35-tvm-dispatch-capture.mts';
import {canonicalSha256, type ArchiveInfo} from './verify-release-evidence.ts';
import {
  assertQwen35BrowserArchive,
  assertQwen35ProductionArchive,
  createQwen35ProvenancePins,
  validateQwen35NativeBuild
} from './pin-qwen35-provenance.ts';
import {
  assertQwen35EvidenceRecord,
  assertQwen35OfflineResults,
  assertQwen35OfflineParity,
  assertQwen35PassedSourceReference,
  assertQwen35SourceCase,
  assertQwen35TvmMainCallSites,
  assertQwen35VisionImageCountCoverage,
  assertQwen35B3S4096SourceReference,
  buildQwen35ResultParityDiagnostic,
  compareQwen35Result,
  assertQwen35MixedVisionSourceReference,
  loadQwen35MixedVisionSourceReference,
  QWEN35_MIXED_VISION_REFERENCE,
  QWEN35_OFFICIAL_VISION_REFERENCE,
  loadQwen35CandidateInputs,
  loadQwen35B3S4096SourceReference,
  QWEN35_RELEASE_MODEL,
  writeQwen35OnlineDiagnostic
} from './strands-qwen35-release-evidence.ts';

test('Qwen release source reference pins public batch-three full-context inference', async () => {
  const source = await loadQwen35B3S4096SourceReference();
  assert.equal(source.batchSize, 3);
  assert.equal(source.sequenceLength, QWEN35_RELEASE_MODEL.maxLength);
  assert.equal(source.graphInputTokens, 12_288);
  assert.deepEqual(source.expectedUsage, {input_tokens: 4_250, output_tokens: 3});
  assert.deepEqual(source.questionNames, ['runtime', 'runtime_copy_2', 'runtime_copy_3']);
  assert.deepEqual(source.labels, ['Node.js', 'TVM WebGPU', 'WebGPU']);
  assert.deepEqual(source.expectedChoices, ['TVM WebGPU', 'TVM WebGPU', 'TVM WebGPU']);
  assert.equal(source.expectedProbabilities.length, 3);
  assert.ok(source.expectedProbabilities.every((row) => row.length === 3));
  assert.equal(source.sourceReference.sha256, 'd18bf9ef341447d34c519da817d140dc818f9a270df4ca06142fe9991d687900');

  const protocolReport = JSON.parse(
    (
      await readFile(
        new URL('../strands/qwen35/reports/qwen35-source-reference-cpu-r3-mixed-2026-11-15.json', import.meta.url),
        'utf8'
      )
    ).replace(/(^|[\s:,\u005B])(-Infinity|Infinity|NaN)(?=$|[\s,\]}])/gmu, '$1null')
  ) as {request: {questions: Array<{name: string; slotLabelsByPosition: string[]}>}};
  const protocolRuntime = protocolReport.request.questions.find(({name}) => name === 'runtime');
  assert.deepEqual(protocolRuntime?.slotLabelsByPosition, ['WebGPU', 'Node.js', 'TVM WebGPU']);
  assert.notDeepEqual(protocolRuntime?.slotLabelsByPosition, source.labels);

  const reducedBatch = structuredClone(source.rawReference) as Record<string, any>;
  reducedBatch.request.questions.pop();
  assert.throws(() => assertQwen35B3S4096SourceReference(reducedBatch), /three questions|batch-three/u);

  const reorderedLabels = structuredClone(source.rawReference) as Record<string, any>;
  const originalCriteria = reorderedLabels.request.questions[0].question.criteria as Record<string, string>;
  reorderedLabels.request.questions[0].question.criteria = {
    WebGPU: originalCriteria.WebGPU,
    'Node.js': originalCriteria['Node.js'],
    'TVM WebGPU': originalCriteria['TVM WebGPU']
  };
  assert.throws(() => assertQwen35B3S4096SourceReference(reorderedLabels), /option label order/u);

  const reducedContext = structuredClone(source.rawReference) as Record<string, any>;
  reducedContext.sourceDerivedForward.actualSequence = QWEN35_RELEASE_MODEL.maxLength - 1;
  assert.throws(() => assertQwen35B3S4096SourceReference(reducedContext), /4096/u);

  const changedOutput = structuredClone(source.rawReference) as Record<string, any>;
  changedOutput.candidateOutputs.probabilitiesShape[0] = 2;
  assert.throws(() => assertQwen35B3S4096SourceReference(changedOutput), /3 × 3|3-by-3/u);
});

test('Qwen exact-package static evidence requires R19 and vision VM main calls plus TVM dispatch code', () => {
  const textRunnerSource = ['runtime.createVirtualMachine(device).getFunction("main")', 'await main(...args)'].join(
    ';'
  );
  const visionRunnerSource = [
    'runtime.createVirtualMachine(runtime.webgpu()).getFunction("main")',
    'runImage(runtime, main, input)'
  ].join(';');
  const runtimeSource = 'compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2])';
  assert.deepEqual(assertQwen35TvmMainCallSites(textRunnerSource, visionRunnerSource, runtimeSource), {
    r19: true,
    vision: true,
    tvmRuntimeDispatch: true
  });
  assert.throws(
    () => assertQwen35TvmMainCallSites('await main(...args)', visionRunnerSource, runtimeSource),
    /R19 TVM VM main/u
  );
  assert.throws(
    () =>
      assertQwen35TvmMainCallSites(
        textRunnerSource,
        visionRunnerSource.replace('runImage(runtime, main, input)', ''),
        runtimeSource
      ),
    /vision inputs/u
  );
  assert.throws(
    () => assertQwen35TvmMainCallSites(textRunnerSource, visionRunnerSource, 'no GPU call here'),
    /WebGPU compute dispatch/u
  );
});

test('Qwen WebGPU dispatch instrumentation is transparent and attributes active graph groups', () => {
  const marker = {};
  const calls: Array<{receiver: unknown; args: [number, number?, number?]}> = [];
  const prototype = {
    dispatchWorkgroups(this: unknown, ...args: [number, number?, number?]) {
      calls.push({receiver: this, args});
      return marker;
    }
  };
  const capture = installQwen35TvmDispatchCapture(prototype, () => 'at runtime (tvm.generated-test.js:1:1)');
  const encoder = Object.create(prototype) as typeof prototype;

  assert.equal(encoder.dispatchWorkgroups(2, 3), marker);
  assert.deepEqual(calls[0], {receiver: encoder, args: [2, 3]});
  assert.deepEqual(capture.dispatchCounts, {});

  capture.activeGroup = 'text';
  assert.equal(encoder.dispatchWorkgroups(4, 5, 6), marker);
  assert.deepEqual(calls[1], {receiver: encoder, args: [4, 5, 6]});
  assert.equal(capture.dispatchCounts.text, 1);
  assert.equal(capture.tvmRuntimeDispatchCounts.text, 1);
  assert.match(capture.tvmRuntimeCallers.text!, /tvm\.generated/u);

  capture.activeGroup = 'vision';
  assert.equal(encoder.dispatchWorkgroups(7), marker);
  assert.equal(capture.dispatchCounts.vision, 1);
  assert.equal(capture.tvmRuntimeDispatchCounts.vision, 1);
});

const expectedResult = {
  model: QWEN35_RELEASE_MODEL.modelId,
  usage: {input_tokens: 31, output_tokens: 3},
  answers: {
    choice: {
      type: 'choice',
      choice: 'gpu',
      probabilities: {cpu: 0.2, gpu: 0.8},
      confidence: 0.6
    },
    score: {
      type: 'score',
      score: 1.5,
      legend: {0: 'low', 1: 'mid', 2: 'high'},
      probabilities: {0: 0.2, 1: 0.4, 2: 0.4},
      confidence: 0.4
    },
    truth: {type: 'noul', noul: 0.9}
  }
};

type OfficialVisionAnswer =
  | {
      type: 'choice';
      choice: string;
      confidence: number;
      probabilities: Record<string, number>;
    }
  | {
      type: 'score';
      score: number;
      confidence: number;
      legend: Record<string, string>;
      probabilities: Record<string, number>;
    }
  | {type: 'noul'; noul: number};

type OfficialVisionResult = {
  model: string;
  usage: {input_tokens: number; output_tokens: number};
  answers: {
    image_choice: Extract<OfficialVisionAnswer, {type: 'choice'}>;
    image_score: Extract<OfficialVisionAnswer, {type: 'score'}>;
    image_noul: Extract<OfficialVisionAnswer, {type: 'noul'}>;
  };
};

type OfficialVisionPrecisionValue = Readonly<{allowedDelta: number}>;
type OfficialOneImageVisionCase = Readonly<{
  name: string;
  imageFiles: readonly unknown[];
  expected: OfficialVisionResult;
  precisionReference: OfficialVisionResult;
  precisionEnvelope: Readonly<{
    answers: Readonly<{
      image_choice: Readonly<{
        confidence: OfficialVisionPrecisionValue;
        probabilities: Readonly<Record<string, OfficialVisionPrecisionValue>>;
      }>;
      image_score: Readonly<{
        score: OfficialVisionPrecisionValue;
        confidence: OfficialVisionPrecisionValue;
        probabilities: Readonly<Record<string, OfficialVisionPrecisionValue>>;
      }>;
      image_noul: Readonly<{noul: OfficialVisionPrecisionValue}>;
    }>;
  }>;
}>;
type OfficialVisionReference = Readonly<{
  format: string;
  status: string;
  cases: readonly OfficialOneImageVisionCase[];
}>;

test('failed upstream source reports cannot enter the exact-package release gate', () => {
  const passed = {format: 'source-reference', status: 'passed-cpu-source-reference'};
  assert.deepEqual(assertQwen35PassedSourceReference(passed, 'pinned source'), passed);
  assert.throws(
    () => assertQwen35PassedSourceReference({...passed, status: 'failed'}, 'pinned source'),
    /not a passing upstream source reference/u
  );
});

test('Qwen parity compares every typed result to pinned upstream outputs', () => {
  const observed = structuredClone(expectedResult);
  observed.answers.choice.probabilities.gpu += 0.004;
  observed.answers.choice.probabilities.cpu -= 0.004;
  compareQwen35Result(observed, expectedResult, 'typed source parity');
  assert.throws(
    () =>
      compareQwen35Result(
        {...observed, answers: {...observed.answers, truth: {type: 'noul', noul: 0.95}}},
        expectedResult,
        'noul source parity'
      ),
    /noul value differs/u
  );
  assert.throws(
    () =>
      compareQwen35Result(
        {...observed, answers: {...observed.answers, choice: {...observed.answers.choice, choice: 'cpu'}}},
        expectedResult,
        'choice source parity'
      ),
    /choice differs/u
  );
  const measuredTextEnvelope = structuredClone(expectedResult);
  measuredTextEnvelope.answers.score.score += 0.0092;
  compareQwen35Result(measuredTextEnvelope, expectedResult, 'measured text source/WebGPU envelope');
  assert.throws(() => {
    const outsideMeasuredEnvelope = structuredClone(expectedResult);
    outsideMeasuredEnvelope.answers.score.score += 0.0093;
    compareQwen35Result(outsideMeasuredEnvelope, expectedResult, 'measured text source/WebGPU envelope');
  }, /source precision envelope/u);
});

test('online parity diagnostics persist candidate/source outputs, confidence, and every choice probability', async () => {
  const candidate = structuredClone(expectedResult);
  candidate.answers.choice.probabilities.cpu = 0.196;
  candidate.answers.choice.probabilities.gpu = 0.804;
  candidate.answers.choice.confidence = 0.62;

  const comparison = buildQwen35ResultParityDiagnostic(candidate, expectedResult, 'text/persisted-parity');
  assert.equal(comparison.status, 'failed');

  const root = await mkdtemp(join(tmpdir(), 'qwen35-online-diagnostic-'));
  const diagnosticPath = join(root, 'outside-reports', 'online-diagnostic.json');
  try {
    await writeQwen35OnlineDiagnostic(diagnosticPath, {
      schemaVersion: 2,
      status: 'online-inference-returned-unverified',
      parity: {status: comparison.status, sourceParity: [comparison]}
    });

    const persisted = JSON.parse(await readFile(diagnosticPath, 'utf8')) as {
      readonly parity: {
        readonly status: string;
        readonly sourceParity: ReadonlyArray<{
          readonly candidateOutput: unknown;
          readonly sourceOutput: unknown;
          readonly answers: ReadonlyArray<{
            readonly name: string;
            readonly numeric: Readonly<
              Record<
                string,
                {
                  readonly absoluteDelta: number | null;
                  readonly allowedDelta: number | null;
                  readonly withinTolerance: boolean | null;
                }
              >
            >;
            readonly probabilities: Readonly<
              Record<
                string,
                {
                  readonly absoluteDelta: number | null;
                  readonly allowedDelta: number | null;
                  readonly withinTolerance: boolean | null;
                }
              >
            >;
          }>;
        }>;
      };
    };
    const [savedComparison] = persisted.parity.sourceParity;
    assert.ok(savedComparison);
    assert.equal(persisted.parity.status, 'failed');
    assert.deepEqual(savedComparison.candidateOutput, candidate);
    assert.deepEqual(savedComparison.sourceOutput, expectedResult);

    const choice = savedComparison.answers.find(({name}) => name === 'choice');
    assert.ok(choice);
    assert.ok(Math.abs((choice.numeric.confidence?.absoluteDelta ?? 0) - 0.02) < 1e-12);
    assert.ok(Math.abs((choice.probabilities.cpu?.absoluteDelta ?? 0) - 0.004) < 1e-12);
    assert.ok(Math.abs((choice.probabilities.gpu?.absoluteDelta ?? 0) - 0.004) < 1e-12);
    assert.equal(choice.probabilities.cpu?.allowedDelta, 0.00915062427520752 + 0.0001);
    assert.equal(choice.probabilities.cpu?.withinTolerance, true);
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('mixed source keeps uncached work and compares package usage to pinned default-cache evidence', async () => {
  const textReference = JSON.parse(
    await readFile(
      new URL(
        '../strands/qwen35/reports/fixtures/qwen35-system-one-source-reference-v1/source-reference.json',
        import.meta.url
      ),
      'utf8'
    )
  ) as {readonly cases: readonly Record<string, unknown>[]};
  const [sourceCase] = textReference.cases;
  assert.ok(sourceCase);
  const cacheReportBytes = await readFile(
    new URL('../strands/qwen35/reports/qwen35-prefix-cache-equivalence-cpu-r1-2026-11-16.json', import.meta.url)
  );
  assert.equal(
    createHash('sha256').update(cacheReportBytes).digest('hex'),
    (sourceCase.candidateUsageExpectation as {evidence: {reportSha256: string}}).evidence.reportSha256
  );
  assert.equal((sourceCase.expected as typeof expectedResult).usage.input_tokens, 395);
  assertQwen35SourceCase(sourceCase, 'mixed source case with cached usage expectation');

  const candidateUsageExpectation = sourceCase.candidateUsageExpectation as {
    readonly usage: {readonly input_tokens: number; readonly output_tokens: number};
  };
  const candidateResult = {
    ...(sourceCase.expected as typeof expectedResult),
    usage: candidateUsageExpectation.usage
  };
  assert.equal(candidateResult.usage.input_tokens, 281);
  assert.throws(
    () => compareQwen35Result(candidateResult, sourceCase.expected, 'mixed uncached usage'),
    /usage differs/u
  );
  assert.doesNotThrow(() =>
    compareQwen35Result(
      candidateResult,
      sourceCase.expected,
      'mixed cached usage',
      undefined,
      sourceCase.candidateUsageExpectation
    )
  );
});

test('batch boundary cache usage preserves uncached work and accounts for one prefix per source group', async () => {
  const boundaryPath =
    '../strands/qwen35/reports/fixtures/qwen35-system-one-boundary-source-reference-v1/boundary-reference.json';
  const boundaryReference = JSON.parse(await readFile(new URL(boundaryPath, import.meta.url), 'utf8')) as {
    readonly cases: Record<string, Record<string, unknown>>;
    readonly sourceRuns: Record<string, unknown>;
  };
  const batchSourceRun = boundaryReference.sourceRuns.batch as {
    readonly reportPath: string;
    readonly reportSha256: string;
  };
  const sourceRunBytes = await readFile(
    new URL(
      `../strands/qwen35/reports/fixtures/qwen35-system-one-boundary-source-reference-v1/${batchSourceRun.reportPath}`,
      import.meta.url
    )
  );
  assert.equal(createHash('sha256').update(sourceRunBytes).digest('hex'), batchSourceRun.reportSha256);

  for (const [name, uncachedTokens, cachedTokens, groups] of [
    ['batchAtLimit', 4081, 2252, [32]],
    ['batchBeyondLimit', 4203, 2374, [32, 1]]
  ] as const) {
    const sourceCase = boundaryReference.cases[name];
    assert.ok(sourceCase);
    assert.equal((sourceCase.expected as {usage: {input_tokens: number}}).usage.input_tokens, uncachedTokens);
    assertQwen35SourceCase(sourceCase, `${name} cache usage source case`);
    const candidateUsageExpectation = sourceCase.candidateUsageExpectation as {
      readonly usage: {readonly input_tokens: number; readonly output_tokens: number};
      readonly evidence: {readonly groupSizes: readonly number[]; readonly prefixTokensPerGroup: readonly number[]};
    };
    assert.deepEqual(candidateUsageExpectation.evidence.groupSizes, groups);
    assert.deepEqual(
      candidateUsageExpectation.evidence.prefixTokensPerGroup,
      groups.map(() => 59)
    );
    assert.equal(candidateUsageExpectation.usage.input_tokens, cachedTokens);
    assert.equal(
      candidateUsageExpectation.usage.input_tokens,
      uncachedTokens - groups.reduce((saved, size) => saved + 59 * (size - 1), 0)
    );

    const candidateResult = {
      ...(sourceCase.expected as typeof expectedResult),
      usage: candidateUsageExpectation.usage
    };
    assert.throws(
      () => compareQwen35Result(candidateResult, sourceCase.expected, `${name} uncached source usage`),
      /usage differs/u
    );
    assert.doesNotThrow(() =>
      compareQwen35Result(
        candidateResult,
        sourceCase.expected,
        `${name} default-cache usage`,
        undefined,
        sourceCase.candidateUsageExpectation
      )
    );
  }
});

test('vision parity uses per-value BF16-to-FP32 source envelopes while keeping labels and usage exact', () => {
  const fp32Reference = structuredClone(expectedResult);
  fp32Reference.answers.score.score += 0.0109;
  const withinEnvelope = structuredClone(expectedResult);
  withinEnvelope.answers.score.score += 0.0103;
  compareQwen35Result(withinEnvelope, expectedResult, 'vision source precision envelope', fp32Reference);
  assert.throws(() => {
    const outsideEnvelope = structuredClone(expectedResult);
    outsideEnvelope.answers.score.score += 0.0115;
    compareQwen35Result(outsideEnvelope, expectedResult, 'vision source precision envelope', fp32Reference);
  }, /source precision envelope/u);
  assert.throws(
    () =>
      compareQwen35Result(
        {...withinEnvelope, usage: {input_tokens: 32, output_tokens: 3}},
        expectedResult,
        'vision source precision envelope',
        fp32Reference
      ),
    /usage differs/u
  );
  assert.throws(
    () =>
      compareQwen35Result(
        {
          ...withinEnvelope,
          answers: {
            ...withinEnvelope.answers,
            score: {...withinEnvelope.answers.score, legend: {0: 'x', 1: 'mid', 2: 'high'}}
          }
        },
        expectedResult,
        'vision source precision envelope',
        fp32Reference
      ),
    /legend differs from source/u
  );
});

test('pinned one-image vision parity accepts numeric envelope boundaries and rejects the next decimal quantum', async () => {
  const fixtureUrl = new URL(
    '../strands/qwen35/reports/fixtures/qwen35-official-vision-source-reference-r2-2026-11-18/vision-reference.json',
    import.meta.url
  );
  const fixtureBytes = await readFile(fixtureUrl);
  assert.equal(createHash('sha256').update(fixtureBytes).digest('hex'), QWEN35_OFFICIAL_VISION_REFERENCE.sha256);
  const reference = JSON.parse(fixtureBytes.toString('utf8')) as OfficialVisionReference;
  assert.equal(reference.format, 'qwen35-official-vision-source-reference-v2');
  assert.equal(reference.status, 'passed-cpu-source-reference');

  const oneImage = reference.cases.find(({name}) => name === 'official-vision-1-image-448');
  assert.ok(oneImage);
  assert.equal(oneImage.imageFiles.length, 1);
  assertQwen35SourceCase(oneImage, 'pinned one-image vision case');

  const source = oneImage.expected;
  const precision = oneImage.precisionReference;
  const envelope = oneImage.precisionEnvelope.answers;
  const quantum = 0.0001;
  const allowedDelta = (value: OfficialVisionPrecisionValue) => Math.round(value.allowedDelta / quantum) * quantum;
  const upperBoundary = (value: number, allowed: OfficialVisionPrecisionValue, extra = 0) =>
    Number((value + allowedDelta(allowed) + extra).toFixed(4));
  const choiceProbabilityDelta = allowedDelta(envelope.image_choice.probabilities['Blue circle']!);
  const scoreProbabilityDelta = allowedDelta(envelope.image_score.probabilities['0']!);
  const scoreLevelOneDelta = allowedDelta(envelope.image_score.probabilities['1']!);
  const metrics: readonly Readonly<{
    apply(candidate: OfficialVisionResult, extra: number): void;
  }>[] = [
    {
      apply: (candidate, extra) => {
        candidate.answers.image_choice.confidence = upperBoundary(
          source.answers.image_choice.confidence,
          envelope.image_choice.confidence,
          extra
        );
      }
    },
    {
      apply: (candidate, extra) => {
        const delta = choiceProbabilityDelta + extra;
        candidate.answers.image_choice.probabilities['Blue circle'] = Number(
          (source.answers.image_choice.probabilities['Blue circle']! + delta).toFixed(4)
        );
        candidate.answers.image_choice.probabilities['Yellow square'] = Number(
          (source.answers.image_choice.probabilities['Yellow square']! - delta).toFixed(4)
        );
      }
    },
    {
      apply: (candidate, extra) => {
        candidate.answers.image_score.score = upperBoundary(
          source.answers.image_score.score,
          envelope.image_score.score,
          extra
        );
      }
    },
    {
      apply: (candidate, extra) => {
        candidate.answers.image_score.confidence = upperBoundary(
          source.answers.image_score.confidence,
          envelope.image_score.confidence,
          extra
        );
      }
    },
    {
      apply: (candidate, extra) => {
        const delta = scoreProbabilityDelta + extra;
        candidate.answers.image_score.probabilities['0'] = Number(
          (source.answers.image_score.probabilities['0']! + delta).toFixed(4)
        );
        candidate.answers.image_score.probabilities['1'] = Number(
          (source.answers.image_score.probabilities['1']! - scoreLevelOneDelta).toFixed(4)
        );
        candidate.answers.image_score.probabilities['2'] = Number(
          (source.answers.image_score.probabilities['2']! - (delta - scoreLevelOneDelta)).toFixed(4)
        );
      }
    },
    {
      apply: (candidate, extra) => {
        candidate.answers.image_noul.noul = upperBoundary(
          source.answers.image_noul.noul,
          envelope.image_noul.noul,
          extra
        );
      }
    }
  ];
  const boundaryCandidate = structuredClone(source);
  for (const {apply} of metrics) apply(boundaryCandidate, 0);
  assert.doesNotThrow(() => compareQwen35Result(boundaryCandidate, source, oneImage.name, precision));

  for (const {apply} of metrics) {
    const outsideCandidate = structuredClone(boundaryCandidate);
    apply(outsideCandidate, quantum);
    assert.throws(
      () => compareQwen35Result(outsideCandidate, source, oneImage.name, precision),
      /source precision envelope/u
    );
  }
});

test('decimal parity accepts the exact four-decimal envelope and rejects one quantum beyond it', () => {
  const source = structuredClone(expectedResult);
  source.answers.choice.probabilities = {cpu: 0.0016, gpu: 0.9984};
  const precisionReference = structuredClone(source);
  const atBoundary = structuredClone(source);
  atBoundary.answers.choice.probabilities = {cpu: 0.0015, gpu: 0.9985};

  compareQwen35Result(atBoundary, source, 'four-decimal parity boundary', precisionReference);
  const diagnostic = buildQwen35ResultParityDiagnostic(
    atBoundary,
    source,
    'four-decimal parity boundary',
    precisionReference
  );
  assert.equal(diagnostic.status, 'passed');
  assert.equal(diagnostic.answers[0]?.probabilities.cpu?.allowedDelta, 0.0001);
  assert.equal(diagnostic.answers[0]?.probabilities.cpu?.withinTolerance, true);

  const beyondBoundary = structuredClone(atBoundary);
  beyondBoundary.answers.choice.probabilities = {cpu: 0.0014, gpu: 0.9986};
  assert.throws(
    () => compareQwen35Result(beyondBoundary, source, 'four-decimal parity boundary', precisionReference),
    /source precision envelope/u
  );
  const failedDiagnostic = buildQwen35ResultParityDiagnostic(
    beyondBoundary,
    source,
    'four-decimal parity boundary',
    precisionReference
  );
  assert.equal(failedDiagnostic.status, 'failed');
  assert.equal(failedDiagnostic.answers[0]?.probabilities.cpu?.withinTolerance, false);

  const spreadSource = structuredClone(expectedResult);
  spreadSource.answers.choice.probabilities = {cpu: 0.9799, gpu: 0.0201};
  const spreadReference = structuredClone(spreadSource);
  spreadReference.answers.choice.probabilities = {cpu: 0.9795, gpu: 0.0205};
  const spreadAtBoundary = structuredClone(spreadSource);
  spreadAtBoundary.answers.choice.probabilities = {cpu: 0.9804, gpu: 0.0196};
  compareQwen35Result(spreadAtBoundary, spreadSource, 'nonzero precision spread boundary', spreadReference);
  const spreadDiagnostic = buildQwen35ResultParityDiagnostic(
    spreadAtBoundary,
    spreadSource,
    'nonzero precision spread boundary',
    spreadReference
  );
  assert.equal(spreadDiagnostic.status, 'passed');
  assert.equal(spreadDiagnostic.answers[0]?.probabilities.cpu?.absoluteDelta, 0.0005);
  assert.equal(spreadDiagnostic.answers[0]?.probabilities.cpu?.allowedDelta, 0.0005);
  assert.equal(spreadDiagnostic.answers[0]?.probabilities.cpu?.withinTolerance, true);

  const spreadBeyondBoundary = structuredClone(spreadAtBoundary);
  spreadBeyondBoundary.answers.choice.probabilities = {cpu: 0.9805, gpu: 0.0195};
  assert.throws(
    () => compareQwen35Result(spreadBeyondBoundary, spreadSource, 'nonzero precision spread boundary', spreadReference),
    /source precision envelope/u
  );
  const spreadFailureDiagnostic = buildQwen35ResultParityDiagnostic(
    spreadBeyondBoundary,
    spreadSource,
    'nonzero precision spread boundary',
    spreadReference
  );
  assert.equal(spreadFailureDiagnostic.status, 'failed');
  assert.equal(spreadFailureDiagnostic.answers[0]?.probabilities.cpu?.withinTolerance, false);
});

test('mixed vision source reference keeps its exact 2/3-image cases and full precision envelope', async () => {
  const sourceReferencePath =
    '../strands/qwen35/reports/fixtures/qwen35-image-counts-2-3-source-oracle-r1-2026-11-17/source-reference.json';
  const sourceReference = JSON.parse(await readFile(new URL(sourceReferencePath, import.meta.url), 'utf8')) as Record<
    string,
    unknown
  >;
  const sourceCases = sourceReference.cases as Array<Record<string, unknown>>;
  const cases = assertQwen35MixedVisionSourceReference(sourceReference);

  assert.deepEqual(
    cases.map(({name, imageFiles}) => [name, imageFiles?.length]),
    [
      ['two-image-mixed-aspect', 2],
      ['three-image-mixed-aspect', 3]
    ]
  );
  for (const [index, item] of cases.entries()) {
    const sourcePrecisionReference = structuredClone(sourceCases[index]!.precisionReference as Record<string, unknown>);
    sourcePrecisionReference.model = QWEN35_RELEASE_MODEL.modelId;
    assert.deepEqual(item.precisionReference, sourcePrecisionReference);
    assert.deepEqual(item.precisionEnvelope, sourceCases[index]!.precisionEnvelope);
  }

  const wrongSource = structuredClone(sourceReference);
  (wrongSource.source as Record<string, unknown>).strandsRevision = '0'.repeat(40);
  assert.throws(() => assertQwen35MixedVisionSourceReference(wrongSource), /Strands revision mismatch/u);
  const wrongProcessor = structuredClone(sourceReference);
  (wrongProcessor.processor as Record<string, unknown>).configSha256 = '0'.repeat(64);
  assert.throws(() => assertQwen35MixedVisionSourceReference(wrongProcessor), /processor config pin mismatch/u);
  const wrongEnvelope = structuredClone(sourceReference);
  const envelope = (wrongEnvelope.cases as Array<Record<string, unknown>>)[0]!.precisionEnvelope as {
    answers: Record<string, Record<string, unknown>>;
  };
  envelope.answers.image_score = {...envelope.answers.image_score, score: {bf16: 1}};
  assert.throws(() => assertQwen35MixedVisionSourceReference(wrongEnvelope), /precision envelope differs/u);
});

test('M4 vision gate fails closed when optional references omit any image count', () => {
  const completeVisionMatrix = [1, 2, 3, 4].map((count) => ({
    imageFiles: Array.from({length: count}, () => ({path: 'pinned.png'}))
  }));
  assert.doesNotThrow(() => assertQwen35VisionImageCountCoverage(completeVisionMatrix));
  for (const missingCount of [1, 2, 3, 4]) {
    assert.throws(
      () =>
        assertQwen35VisionImageCountCoverage(
          completeVisionMatrix.filter(({imageFiles}) => imageFiles.length !== missingCount)
        ),
      new RegExp(`requires a pinned ${missingCount}-image vision case`, 'u')
    );
  }
});

test('mixed vision loader pins the report location and rehashes all source images', async () => {
  const root = await mkdtemp(join(tmpdir(), 'qwen35-mixed-vision-reference-'));
  try {
    const sourceFixtureRoot = new URL(
      '../strands/qwen35/reports/fixtures/qwen35-image-counts-2-3-source-oracle-r1-2026-11-17/',
      import.meta.url
    );
    const sourceReferenceBytes = await readFile(new URL('source-reference.json', sourceFixtureRoot));
    const referencePath = join(root, QWEN35_MIXED_VISION_REFERENCE.path);
    await mkdir(join(root, 'references/qwen35-image-counts-2-3-source-oracle-r1-2026-11-17'), {recursive: true});
    await writeFile(referencePath, sourceReferenceBytes);

    const sourceReference = JSON.parse(sourceReferenceBytes.toString('utf8')) as {
      cases: Array<{imageFiles: Array<{path: string}>}>;
    };
    for (const image of sourceReference.cases.flatMap(({imageFiles}) => imageFiles)) {
      const imagePath = join(root, image.path);
      await mkdir(dirname(imagePath), {recursive: true});
      await writeFile(imagePath, await readFile(new URL(image.path, sourceFixtureRoot)));
    }

    const loaded = await loadQwen35MixedVisionSourceReference(root, QWEN35_MIXED_VISION_REFERENCE);
    assert.equal(loaded.path, QWEN35_MIXED_VISION_REFERENCE.path);
    assert.equal(loaded.sha256, QWEN35_MIXED_VISION_REFERENCE.sha256);
    assert.equal(loaded.cases.length, 2);
    await assert.rejects(
      loadQwen35MixedVisionSourceReference(root, {
        ...QWEN35_MIXED_VISION_REFERENCE,
        path: 'references/renamed-source-reference.json'
      }),
      /reference path is not pinned/u
    );
    await assert.rejects(
      loadQwen35MixedVisionSourceReference(root, {...QWEN35_MIXED_VISION_REFERENCE, sha256: '0'.repeat(64)}),
      /reference digest is not pinned/u
    );

    const changedImagePath = join(root, loaded.cases[0]!.imageFiles![0]!.path);
    await writeFile(changedImagePath, Buffer.from('changed'));
    await assert.rejects(
      loadQwen35MixedVisionSourceReference(root, QWEN35_MIXED_VISION_REFERENCE),
      /image size differs|image digest differs/u
    );
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('official vision v2 pin preserves choice labels in criteria insertion order', async () => {
  const fixtureUrl = new URL(
    '../strands/qwen35/reports/fixtures/qwen35-official-vision-source-reference-r2-2026-11-18/vision-reference.json',
    import.meta.url
  );
  const bytes = await readFile(fixtureUrl);
  const digest = createHash('sha256').update(bytes).digest('hex');
  const reference = JSON.parse(bytes.toString('utf8')) as {
    format: string;
    cases: Array<Record<string, unknown>>;
  };
  assert.equal(
    QWEN35_OFFICIAL_VISION_REFERENCE.path,
    'references/qwen35-official-vision-source-reference-r2-2026-11-18/vision-reference.json'
  );
  assert.equal(digest, QWEN35_OFFICIAL_VISION_REFERENCE.sha256);
  assert.equal(reference.format, 'qwen35-official-vision-source-reference-v2');
  assert.deepEqual(
    reference.cases.map((item) => item.choiceLabels),
    [
      ['Blue circle', 'Orange diamond', 'Red triangle', 'White triangle', 'Yellow square'],
      ['Blue circle', 'Orange diamond', 'Red triangle', 'White triangle', 'Yellow square']
    ]
  );
  for (const item of reference.cases) {
    assert.doesNotThrow(() => assertQwen35SourceCase(item, 'official vision v2 source case'));
    const request = item.request as {questions: Array<[string, {criteria?: Record<string, unknown>}]>};
    const choice = request.questions.find(([name]) => name === 'image_choice')![1];
    const wrongLabels = [...(item.choiceLabels as string[])].reverse();
    assert.notDeepEqual(Object.keys(choice.criteria!), wrongLabels);
    assert.throws(
      () => assertQwen35SourceCase({...item, choiceLabels: wrongLabels}, 'mutated official vision choice order'),
      /choiceLabels differ from request criteria order/u
    );
  }
});

test('offline parity rejects even small output drift', () => {
  assertQwen35OfflineParity(expectedResult, structuredClone(expectedResult));
  assert.throws(
    () => assertQwen35OfflineParity(expectedResult, {...expectedResult, usage: {input_tokens: 32, output_tokens: 3}}),
    /exactly/u
  );
});

test('source cases preserve ordered question-pair names and follow the pinned choice schema', () => {
  const choiceCase = {
    name: 'integer-like-choice-name',
    request: {
      state: 'state',
      questions: [['10', {type: 'choice', instructions: 'choose', criteria: {cpu: null, gpu: null}}]]
    },
    expected: {
      model: QWEN35_RELEASE_MODEL.modelId,
      usage: {input_tokens: 31, output_tokens: 1},
      answers: {
        '10': {type: 'choice', choice: 'gpu', probabilities: {cpu: 0.2, gpu: 0.8}, confidence: 0.6}
      }
    }
  };
  assert.doesNotThrow(() => assertQwen35SourceCase(choiceCase, 'ordered choice case'));
  assert.throws(
    () =>
      assertQwen35SourceCase(
        {
          ...choiceCase,
          request: {
            ...choiceCase.request,
            questions: [['10', {type: 'choice', instructions: 'choose', criteria: ['cpu', 'gpu']}]]
          }
        },
        'array criteria case'
      ),
    /criteria must be an object/u
  );
  assert.throws(
    () =>
      assertQwen35SourceCase(
        {
          ...choiceCase,
          request: {
            ...choiceCase.request,
            questions: [
              ['10', {type: 'choice', instructions: 'choose', criteria: {cpu: null, gpu: null}}],
              ['10', {type: 'choice', instructions: 'choose', criteria: {cpu: null, gpu: null}}]
            ]
          }
        },
        'duplicate question names case'
      ),
    /duplicate question names/u
  );
});

test('source score answers bind probability levels and legend text to the requested rubric', () => {
  const scoreCase = {
    name: 'score-rubric',
    request: {
      state: 'state',
      questions: {rating: {type: 'score', instructions: 'rate', criteria: [' low ', 'middle', 'high']}}
    },
    expected: {
      model: QWEN35_RELEASE_MODEL.modelId,
      usage: {input_tokens: 31, output_tokens: 1},
      answers: {
        rating: {
          type: 'score',
          score: 1.25,
          legend: {0: 'low', 1: 'middle', 2: 'high'},
          probabilities: {0: 0.2, 1: 0.5, 2: 0.3},
          confidence: 0.5
        }
      }
    }
  };
  assert.doesNotThrow(() => assertQwen35SourceCase(scoreCase, 'score rubric case'));
  assert.throws(
    () =>
      assertQwen35SourceCase(
        {
          ...scoreCase,
          expected: {
            ...scoreCase.expected,
            answers: {
              ...scoreCase.expected.answers,
              rating: {...scoreCase.expected.answers.rating, legend: {0: 'wrong', 1: 'middle', 2: 'high'}}
            }
          }
        },
        'mismatched score legend case'
      ),
    /legend differs from request criteria/u
  );
});

test('offline result parity correlates duplicate case names by order, not by a lossy name map', () => {
  const online = [
    {name: 'shared-name', result: {answer: 1}},
    {name: 'shared-name', result: {answer: 2}}
  ];
  assert.doesNotThrow(() => assertQwen35OfflineResults(online, structuredClone(online)));
  assert.throws(() => assertQwen35OfflineResults(online, [...online].reverse()), /exactly/u);
  assert.throws(
    () => assertQwen35OfflineResults(online, [{...online[0]!, name: 'other-name'}, online[1]]),
    /case order/u
  );
});

test('missing generated artifacts fail closed before any browser run', async () => {
  const root = await mkdtemp(join(tmpdir(), 'qwen35-release-evidence-'));
  try {
    await assert.rejects(loadQwen35CandidateInputs(root), /release-harness\.json/u);
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('Qwen package descriptor hash pins the catalog asset manifest separately from the build manifest', async () => {
  const assetManifest = JSON.parse(
    await readFile(new URL('../../packages/web-decision/assets/qwen35/manifest.json', import.meta.url), 'utf8')
  ) as unknown;
  assert.equal(QWEN35_RELEASE_MODEL.assetManifestSha256, canonicalSha256(assetManifest));
});

test('candidate evidence requires the exact package, pinned source, M4 GPU, and all end-to-end checks', () => {
  const b3Result = {
    model: QWEN35_RELEASE_MODEL.modelId,
    usage: {input_tokens: 4_250, output_tokens: 3},
    answers: {
      runtime: {
        type: 'choice',
        choice: 'TVM WebGPU',
        probabilities: {'Node.js': 0.10017982870340347, 'TVM WebGPU': 0.8295549154281616, WebGPU: 0.0702652782201767}
      },
      runtime_copy_2: {
        type: 'choice',
        choice: 'TVM WebGPU',
        probabilities: {'Node.js': 0.10017982870340347, 'TVM WebGPU': 0.8295549154281616, WebGPU: 0.0702652782201767}
      },
      runtime_copy_3: {
        type: 'choice',
        choice: 'TVM WebGPU',
        probabilities: {'Node.js': 0.10017982870340347, 'TVM WebGPU': 0.8295549154281616, WebGPU: 0.0702652782201767}
      }
    }
  };
  const good = {
    schemaVersion: 1,
    status: 'candidate-browser-evidence-passed',
    modelId: QWEN35_RELEASE_MODEL.catalogModelId,
    package: {
      name: 'web-decision',
      version: '0.0.0-qwen35-candidate',
      archiveSha256: 'a'.repeat(64),
      packageTreeSha256: 'b'.repeat(64),
      bundlePath: '/tmp/web-decision/dist/index.js',
      bundleSha256: 'c'.repeat(64),
      candidateBundlePath: '/tmp/web-decision/dist/index.js',
      candidateBundleSha256: 'c'.repeat(64),
      exactInstallMatch: true
    },
    source: {
      baseRepository: QWEN35_RELEASE_MODEL.baseRepository,
      baseRevision: QWEN35_RELEASE_MODEL.baseRevision,
      checkpointRepository: QWEN35_RELEASE_MODEL.repository,
      checkpointRevision: QWEN35_RELEASE_MODEL.revision,
      strandsRepository: QWEN35_RELEASE_MODEL.strandsRepository,
      strandsRevision: QWEN35_RELEASE_MODEL.strandsRevision,
      runner: 'StrandsDeciderModel.load/SystemOneEngine.evaluate',
      weightsLoaded: true,
      tvmUsed: false,
      b3S4096SourceReference: {
        path: 'scripts/strands/qwen35/reports/qwen35-full-pointer-b3-s4096-source-reference-r1-2026-11-16.json',
        sha256: 'd18bf9ef341447d34c519da817d140dc818f9a270df4ca06142fe9991d687900'
      }
    },
    candidateExecution: {
      backend: 'tvm-webgpu',
      runtime: 'TVM.js',
      graphMainCallSites: {r19: true, vision: true, tvmRuntimeDispatch: true},
      graphDispatchCounts: {text: 7, vision: 4, 'batch-3-context-4096': 5},
      tvmRuntimeDispatchCounts: {text: 7, vision: 4, 'batch-3-context-4096': 5},
      graphDispatchCallers: {
        text: 'at dispatch (http://localhost/tvm.generated-abc.js:12:3)',
        vision: 'at dispatch (http://localhost/tvm.generated-abc.js:12:3)',
        'batch-3-context-4096': 'at dispatch (http://localhost/tvm.generated-abc.js:12:3)'
      },
      graphsExecuted: {r19: true, vision: true},
      tvmUsed: true
    },
    artifacts: {
      manifestSha256: 'd'.repeat(64),
      assetManifestSha256: QWEN35_RELEASE_MODEL.assetManifestSha256,
      filesSha256: {'model.wasm': 'e'.repeat(64)}
    },
    browser: {
      name: 'chromium',
      channel: 'chrome',
      version: '141.0.7390.0',
      userAgent: 'Mozilla/5.0 Chrome/141.0.0.0 Safari/537.36'
    },
    hardware: {platform: 'darwin', arch: 'arm64', chipType: 'Apple M4 Pro'},
    gpu: {
      api: 'webgpu',
      adapter: {vendor: 'apple'},
      isFallbackAdapter: false,
      renderer: 'ANGLE Metal Renderer: Apple M4 Pro',
      chromeGpuStatus: 'Hardware accelerated'
    },
    checks: {
      curatedPublicApi: true,
      sourceParity: true,
      typedChoiceScoreNoul: true,
      officialVision: true,
      maximumContextBoundary: true,
      maximumChoiceOptionsBoundary: true,
      maximumScoreLevelsBoundary: true,
      maximumImageCountBoundary: true,
      batchBoundaries: true,
      fullContextBatchBoundary: true,
      onlineInference: true,
      offlineCacheReuse: true
    },
    onlineNetworkRequests: [
      {
        url: 'https://huggingface.co/StrandsAgents/model/resolve/revision/model.safetensors',
        method: 'GET',
        rangeHeader: 'bytes=0-1048575'
      }
    ],
    rangedHuggingFaceRequests: [
      {
        url: 'https://huggingface.co/StrandsAgents/model/resolve/revision/model.safetensors',
        method: 'GET',
        rangeHeader: 'bytes=0-1048575'
      }
    ],
    offlineNetworkAttempts: [],
    fullContextBatch: {
      sourceReference: {
        path: 'scripts/strands/qwen35/reports/qwen35-full-pointer-b3-s4096-source-reference-r1-2026-11-16.json',
        sha256: 'd18bf9ef341447d34c519da817d140dc818f9a270df4ca06142fe9991d687900'
      },
      batchSize: 3,
      sequenceLength: 4096,
      expectedUsage: {input_tokens: 4_250, output_tokens: 3},
      expectedProbabilities: [
        [0.10017982870340347, 0.8295549154281616, 0.0702652782201767],
        [0.10017982870340347, 0.8295549154281616, 0.0702652782201767],
        [0.10017982870340347, 0.8295549154281616, 0.0702652782201767]
      ],
      result: b3Result,
      parityPassed: true
    },
    results: {
      b3S4096: [{name: 'batch-3-context-4096', result: b3Result}],
      offline: [{name: 'batch-3-context-4096', result: b3Result}]
    }
  };
  assert.equal(assertQwen35EvidenceRecord(good).status, 'candidate-browser-evidence-passed');
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, fullContextBatch: {...good.fullContextBatch, batchSize: 2}}),
    /three rows/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, fullContextBatch: {...good.fullContextBatch, sequenceLength: 4095}}),
    /4096-token/u
  );
  const paddedUsage = structuredClone(good);
  paddedUsage.fullContextBatch.result.usage = {input_tokens: 12_288, output_tokens: 3};
  assert.throws(() => assertQwen35EvidenceRecord(paddedUsage), /usage differs from the pinned source boundary/u);
  const wrongChoiceMap = structuredClone(good);
  wrongChoiceMap.fullContextBatch.result.answers.runtime.choice = 'Node.js';
  assert.throws(() => assertQwen35EvidenceRecord(wrongChoiceMap), /choice differs from source/u);
  assert.throws(
    () =>
      assertQwen35EvidenceRecord({
        ...good,
        fullContextBatch: {
          ...good.fullContextBatch,
          result: {
            ...good.fullContextBatch.result,
            answers: {
              ...good.fullContextBatch.result.answers,
              runtime_copy_3: {
                ...good.fullContextBatch.result.answers.runtime_copy_3,
                probabilities: {...good.fullContextBatch.result.answers.runtime_copy_3.probabilities, WebGPU: 0.9}
              }
            }
          }
        }
      }),
    /probability WebGPU/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, source: {...good.source, b3S4096SourceReference: undefined}}),
    /pinned independent source report/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, results: {...good.results, b3S4096: []}}),
    /one B3×S4096 result/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, results: {...good.results, offline: []}}),
    /offline evidence must include the B3×S4096 case/u
  );
  for (const candidateExecution of [
    {...good.candidateExecution, backend: 'onnxruntime-web'},
    {...good.candidateExecution, graphsExecuted: {r19: true, vision: false}},
    {...good.candidateExecution, tvmUsed: false}
  ])
    assert.throws(
      () => assertQwen35EvidenceRecord({...good, candidateExecution}),
      /TVM WebGPU|TVM.js|executed TVM|execution must be recorded/u
    );
  assert.throws(
    () =>
      assertQwen35EvidenceRecord({
        ...good,
        artifacts: {...good.artifacts, assetManifestSha256: 'a'.repeat(64)}
      }),
    /descriptor manifest differs/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, modelId: QWEN35_RELEASE_MODEL.modelId}),
    /model ID mismatch/u
  );
  assert.throws(() => assertQwen35EvidenceRecord({...good, status: 'verified'}), /not a release verification/u);
  assert.throws(
    () =>
      assertQwen35EvidenceRecord({
        ...good,
        candidateExecution: {
          ...good.candidateExecution,
          tvmRuntimeDispatchCounts: {...good.candidateExecution.tvmRuntimeDispatchCounts, vision: 0}
        }
      }),
    /pinned TVM.js dispatches during vision/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, package: {...good.package, name: 'other-package'}}),
    /package name mismatch/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, package: {...good.package, version: '0.4.0-rc.1'}}),
    /stable version or the explicit local candidate/u
  );
  assert.doesNotThrow(() => assertQwen35EvidenceRecord({...good, browser: {...good.browser, channel: 'chromium'}}));
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, browser: {...good.browser, channel: 'webkit'}}),
    /Chromium or Chrome channel/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, hardware: {...good.hardware, arch: 'x64'}}),
    /Apple silicon/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, gpu: {...good.gpu, adapter: {vendor: 'generic'}}}),
    /vendor/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, checks: {...good.checks, officialVision: false}}),
    /officialVision/u
  );
  assert.throws(
    () => assertQwen35EvidenceRecord({...good, offlineNetworkAttempts: ['https://example.invalid/model.bin']}),
    /network requests/u
  );
  assert.throws(
    () =>
      assertQwen35EvidenceRecord({
        ...good,
        onlineNetworkRequests: [{url: 'http://127.0.0.1:4173/model.bin', method: 'GET', rangeHeader: 'bytes=0-'}],
        rangedHuggingFaceRequests: []
      }),
    /ranged GET to Hugging Face/u
  );
  assert.throws(
    () =>
      assertQwen35EvidenceRecord({
        ...good,
        onlineNetworkRequests: [
          {
            url: 'http://huggingface.co/StrandsAgents/model/resolve/revision/model.safetensors',
            method: 'GET',
            rangeHeader: 'bytes=0-1048575'
          }
        ],
        rangedHuggingFaceRequests: []
      }),
    /ranged GET to Hugging Face/u
  );
  assert.throws(
    () =>
      assertQwen35EvidenceRecord({
        ...good,
        onlineNetworkRequests: [
          {
            url: 'https://huggingface.co/StrandsAgents/model/resolve/revision/model.safetensors',
            method: 'GET',
            rangeHeader: 'bytes=10-2'
          }
        ],
        rangedHuggingFaceRequests: []
      }),
    /ranged GET to Hugging Face/u
  );
  assert.throws(
    () =>
      assertQwen35EvidenceRecord({
        ...good,
        package: {...good.package, bundlePath: '/tmp/web-decision/dist/__test/qwen35-candidate/index.js'}
      }),
    /exact public package bundle/u
  );
});

test('Qwen promotion rejects private versions, single-model archives, and browser reports from different bytes', () => {
  const archive: ArchiveInfo = {
    archivePath: '/archive/web-decision-0.5.0.tgz',
    archiveSha256: 'a'.repeat(64),
    installedRoot: '/install/web-decision',
    packageName: 'web-decision',
    packageVersion: '0.5.0',
    packageTreeSha256: 'b'.repeat(64),
    bundlePath: '/install/web-decision/dist/index.js',
    bundleSha256: 'c'.repeat(64),
    installedPackagePathSha256: 'd'.repeat(64),
    entryPathSha256: 'e'.repeat(64),
    archiveFiles: ['assets/kanana/manifest.json', 'assets/qwen35/manifest.json']
  };
  assert.doesNotThrow(() => assertQwen35ProductionArchive(archive));
  assert.throws(
    () => assertQwen35ProductionArchive({...archive, packageVersion: '0.0.0-qwen35-candidate'}),
    /stable npm version/u
  );
  assert.throws(
    () => assertQwen35ProductionArchive({...archive, archiveFiles: ['assets/qwen35/manifest.json']}),
    /also contain Kanana/u
  );
  const {packageVersion: version, archiveSha256, packageTreeSha256, bundleSha256} = archive;
  const evidence = {package: {version, archiveSha256, packageTreeSha256, bundleSha256}};
  assert.doesNotThrow(() => assertQwen35BrowserArchive(evidence, archive));
  for (const field of ['archiveSha256', 'packageTreeSha256', 'bundleSha256'] as const) {
    assert.throws(
      () => assertQwen35BrowserArchive(evidence, {...archive, [field]: 'f'.repeat(64)}),
      /not collected from this exact combined npm archive/u
    );
  }
});

test('Qwen promotion requires unaliased rebuilt WASM and the pinned native build receipt', async (context) => {
  const root = await mkdtemp(join(tmpdir(), 'web-decision-qwen-native-test-'));
  context.after(() => rm(root, {recursive: true, force: true}));
  const packagedPath = fileURLToPath(
    new URL('../../packages/web-decision/assets/qwen35/r19/qwen35-text-pointer.webgpu.wasm', import.meta.url)
  );
  const authoritativeReceipt = fileURLToPath(
    new URL('../strands/qwen35/reports/qwen35-fp32-decoder-build-2026-11-21.json', import.meta.url)
  );
  const wasm = join(root, 'qwen35-gdn-prefix-fp32-candidate.wasm');
  const report = join(root, 'qwen35-gdn-prefix-fp32-candidate.json');
  const pin = {
    bytes: 5_772_332,
    sha256: 'cbad94d32d5839215646e711be9adc8cdf98797beab663e1c2c90e929f3dade3'
  };
  await copyFile(authoritativeReceipt, report);
  await link(packagedPath, wasm);
  await assert.rejects(
    validateQwen35NativeBuild('text', root, pin, [packagedPath], []),
    /must not reuse the packaged artifact file/u
  );
  await rm(wasm);
  await copyFile(packagedPath, wasm);
  await assert.rejects(
    validateQwen35NativeBuild('text', root, pin, [packagedPath], [root]),
    /external Qwen build directory/u
  );
  assert.equal((await validateQwen35NativeBuild('text', root, pin, [packagedPath], [])).sha256, pin.sha256);
  const reportBytes = await readFile(report);
  await writeFile(report, Buffer.concat([reportBytes, Buffer.from('\n')]));
  await assert.rejects(
    validateQwen35NativeBuild('text', root, pin, [packagedPath], []),
    /Pinned native Qwen text build report changed/u
  );
});

test('Qwen prospective pins retain the exact historical Kanana entry', async () => {
  const provenance = await readFile(
    new URL('../../packages/web-decision/src/catalog/model-provenance.json', import.meta.url),
    'utf8'
  );
  const qwen = await readFile(
    new URL(
      '../../packages/web-decision/src/catalog/provenance/strands-decider-2b-qwen3.5-v1-2610.json',
      import.meta.url
    ),
    'utf8'
  );
  const current = JSON.parse(provenance);
  current.models[1].status = 'pending';
  const source = JSON.parse(qwen);
  const pending = {
    sourceLineage: source.sourceLineage,
    upstreamRuntime: source.upstreamRuntime,
    candidate: {artifactLockSha256: '1'.repeat(64)}
  };
  const proposal = createQwen35ProvenancePins(current, pending, {
    rights: [],
    builds: [],
    exactPackage: {
      version: '0.5.0',
      archiveSha256: 'a'.repeat(64),
      packageTreeSha256: 'b'.repeat(64),
      bundleSha256: 'c'.repeat(64),
      reportSha256: 'd'.repeat(64),
      browser: {name: 'chromium', version: '158.0.8127.18', channel: 'chromium'},
      hardware: {platform: 'darwin', arch: 'arm64', chipType: 'Apple M4 Pro'}
    }
  });
  assert.deepEqual(proposal.manifest.models[0], current.models[0]);
  assert.equal(current.models[1].status, 'pending');
  assert.equal(proposal.manifest.models[1]?.status, 'verified');
  assert.deepEqual(proposal.record.sourceLineage, pending.sourceLineage);
  assert.equal(proposal.record.conversion.artifactLockSha256, current.models[1].artifactLockSha256);
  assert.notEqual(proposal.record.conversion.artifactLockSha256, pending.candidate.artifactLockSha256);
  assert.equal(
    proposal.recordSha256,
    createHash('sha256')
      .update(`${JSON.stringify(proposal.record, null, 2)}\n`)
      .digest('hex')
  );
});
