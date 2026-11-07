import {expect, test} from '@playwright/test';
import {mkdir, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';

const enabled = process.env.WEB_DECISION_TAMEV_MICRO_REAL === '1';
const resultPath = process.env.WEB_DECISION_TAMEV_MICRO_OUTPUT;
const expectedCpuWasmVerification = process.env.WEB_DECISION_TAMEV_MICRO_EXPECT_CPU_WASM ?? 'verified';

test.skip(!enabled, 'Set WEB_DECISION_TAMEV_MICRO_REAL=1 to run the pinned TAMEV Micro model.');

test('TAMEV Micro MiniLM accepts current dynamic inputs through CPU WASM', async ({page, browserName}) => {
  test.setTimeout(180_000);
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      get() {
        throw new Error('TAMEV Micro CPU WASM must not inspect or require WebGPU.');
      }
    });
  });
  await page.goto('/');
  const report = await page.evaluate(async () => {
    const {loadDecisionModel} = window.webDecision;
    const loadStarted = performance.now();
    let progress;
    const model = await loadDecisionModel({
      model: 'experimental-tamev-micro-minilm-int8',
      assetsUrl: '/tamev-micro-assets/',
      cache: false,
      onProgress(value) {
        progress = value;
      }
    });
    const loadMilliseconds = performance.now() - loadStarted;
    const state = 'A customer says the same card purchase was charged twice.';
    const instructions = 'Which queue should handle the report?';
    const options = [
      'billing: review a duplicate charge',
      'fraud: investigate unauthorized use',
      'support: fix an app problem'
    ];
    const started = performance.now();
    const generic = await model.decide({state, questions: [{instructions, options}]});
    const firstInferenceMilliseconds = performance.now() - started;
    const reversed = await model.decide({state, questions: [{instructions, options: [...options].reverse()}]});
    const genericProbabilities = Object.fromEntries(
      generic.answers[0].probabilities.map(({option, probability}) => [option, probability])
    );
    const reversedProbabilities = Object.fromEntries(
      reversed.answers[0].probabilities.map(({option, probability}) => [option, probability])
    );
    let maximumPermutationDrift = 0;
    for (const option of options) {
      maximumPermutationDrift = Math.max(
        maximumPermutationDrift,
        Math.abs(genericProbabilities[option] - reversedProbabilities[option])
      );
    }

    const calibratedOptionCounts = {};
    for (const count of [2, 3, 4, 5, 14, 77]) {
      const result = await model.decide({
        state,
        questions: [{instructions, options: Array.from({length: count}, (_, index) => `candidate ${index}`)}]
      });
      calibratedOptionCounts[count] = result.answers[0].probabilities.length;
    }
    const wideOptions = Array.from({length: 78}, (_, index) => 'candidate-' + index);
    const wideGeneric = await model.decide({state, questions: [{instructions, options: wideOptions}]});
    const wideTyped = await model.systemOne(state, {
      wideChoice: {
        type: 'choice',
        instructions,
        criteria: Object.fromEntries(wideOptions.map((option) => [option, null]))
      }
    });
    const wideGenericAnswer = wideGeneric.answers[0];
    const wideTypedAnswer = wideTyped.answers.wideChoice;

    const manyQuestionNames = Array.from(
      {length: 257},
      (_, index) => 'question-' + String(index).padStart(3, '0')
    );
    const expectedQuestionOptionOrder = (index) => {
      const options = [`first-${index}`, `second-${index}`];
      return index % 2 === 0 ? options : options.reverse();
    };
    const manyGeneric = await model.decide({
      state: 'x',
      questions: manyQuestionNames.map((name, index) => ({
        instructions: name,
        options: expectedQuestionOptionOrder(index)
      }))
    });
    const manyTypedQuestions = Object.fromEntries(
      manyQuestionNames.map((name, index) => [
        name,
        {
          type: 'choice',
          instructions: name,
          criteria: Object.fromEntries(expectedQuestionOptionOrder(index).map((option) => [option, null]))
        }
      ])
    );
    const manyTyped = await model.systemOne('x', manyTypedQuestions);
    const genericQuestionOrderPreserved = manyGeneric.answers.every(
      (answer, index) =>
        answer.probabilities.length === 2 &&
        Number.isInteger(answer.selectedIndex) &&
        answer.selectedIndex >= 0 &&
        answer.selectedIndex < 2 &&
        answer.selected === answer.probabilities[answer.selectedIndex].option &&
        answer.probabilities.every(({probability, logit}) => Number.isFinite(probability) && Number.isFinite(logit)) &&
        JSON.stringify(answer.probabilities.map(({option}) => option)) ===
        JSON.stringify(expectedQuestionOptionOrder(index))
    );
    const typedQuestionOrderPreserved =
      JSON.stringify(Object.keys(manyTyped.answers)) === JSON.stringify(manyQuestionNames) &&
      manyQuestionNames.every((name, index) => {
        const answer = manyTyped.answers[name];
        return (
          answer.type === 'choice' &&
          Number.isFinite(answer.confidence) &&
          expectedQuestionOptionOrder(index).includes(answer.choice) &&
          Object.values(answer.probabilities).every(Number.isFinite) &&
          JSON.stringify(Object.keys(answer.probabilities)) === JSON.stringify(expectedQuestionOptionOrder(index))
        );
      });
    const exactContext = await model.decide({
      state: 'x '.repeat(126).trim(),
      questions: [{instructions: '', options: ['yes', 'no']}]
    });
    const overContext = await model.decide({
      state: 'x '.repeat(127).trim(),
      questions: [{instructions: '', options: ['yes', 'no']}]
    });
    const exactOption = await model.decide({
      state,
      questions: [{instructions: '', options: ['x '.repeat(46).trim(), 'short option']}]
    });
    const overOption = await model.decide({
      state,
      questions: [{instructions: '', options: ['x '.repeat(47).trim(), 'short option']}]
    });
    const typed = await model.systemOne(
      {case: 'duplicate card charge', urgent: false},
      {
        route: {
          type: 'choice',
          instructions: 'Select the handling queue',
          criteria: {billing: 'review a duplicate charge', fraud: 'investigate unauthorized use'}
        },
        urgent: {
          type: 'noul',
          instructions: 'Is the customer unable to pay because the duplicate charge drained the account?'
        },
        severity: {type: 'score', instructions: 'Rate the financial impact', criteria: ['low', 'medium', 'high']}
      }
    );
    const warmMilliseconds = [];
    for (let index = 0; index < 8; index++) {
      const warmStarted = performance.now();
      await model.decide({state, questions: [{instructions, options}]});
      warmMilliseconds.push(performance.now() - warmStarted);
    }
    await model.releaseCachedMemory();
    await model.dispose();
    return {
      userAgent: navigator.userAgent,
      description: {
        model: model.description.model,
        backend: model.description.backend,
        verification: model.description.verification,
        assets: model.description.assets.requiredFiles
      },
      loadMilliseconds,
      firstInferenceMilliseconds,
      warmMilliseconds,
      progress,
      genericSelected: generic.answers[0].selected,
      genericProbabilities,
      maximumPermutationDrift,
      calibratedOptionCounts,
      wideChoice: {
        generic: {
          optionOrder: wideGenericAnswer.probabilities.map(({option}) => option),
          selected: wideGenericAnswer.selected,
          selectedIndex: wideGenericAnswer.selectedIndex,
          finiteProbabilities: wideGenericAnswer.probabilities.every(({probability, logit}) =>
            Number.isFinite(probability) && Number.isFinite(logit)
          ),
          truncated: wideGeneric.truncated === true
        },
        typed: {
          type: wideTypedAnswer.type,
          optionOrder: Object.keys(wideTypedAnswer.probabilities),
          selected: wideTypedAnswer.choice,
          finiteConfidence: Number.isFinite(wideTypedAnswer.confidence),
          finiteProbabilities: Object.values(wideTypedAnswer.probabilities).every(Number.isFinite),
          truncated: wideTypedAnswer.truncated === true
        }
      },
      manyQuestions: {
        requested: manyQuestionNames.length,
        generic: manyGeneric.answers.length,
        genericOptionOrderPreserved: genericQuestionOrderPreserved,
        genericTruncated: manyGeneric.truncated === true,
        typed: Object.keys(manyTyped.answers).length,
        typedQuestionOrder: Object.keys(manyTyped.answers),
        typedOptionOrderPreserved: typedQuestionOrderPreserved,
        typedTruncated: manyTyped.truncated === true
      },
      exactContext: {tokenCount: exactContext.tokenCount, truncated: exactContext.truncated === true},
      overContext: {tokenCount: overContext.tokenCount, truncated: overContext.truncated === true},
      exactOption: {tokenCount: exactOption.tokenCount, truncated: exactOption.truncated === true},
      overOption: {tokenCount: overOption.tokenCount, truncated: overOption.truncated === true},
      typed
    };
  });

  expect(report.description).toMatchObject({
    model: 'experimental-tamev-micro-minilm-int8',
    backend: {name: 'onnxruntime-web-wasm', executionProvider: 'wasm', intraOpNumThreads: 1},
    verification: {
      currentBundleGpu: 'not-applicable',
      currentBundleCpuWasm: expectedCpuWasmVerification
    }
  });
  expect(report.progress).toMatchObject({loadedBytes: 24161003, totalBytes: 24161003, fraction: 1});
  expect(report.calibratedOptionCounts).toEqual({2: 2, 3: 3, 4: 4, 5: 5, 14: 14, 77: 77});
  const wideOptionOrder = Array.from({length: 78}, (_, index) => 'candidate-' + index);
  expect(report.wideChoice.generic.optionOrder).toEqual(wideOptionOrder);
  expect(report.wideChoice.generic.selectedIndex).toBeGreaterThanOrEqual(0);
  expect(report.wideChoice.generic.selectedIndex).toBeLessThan(78);
  expect(report.wideChoice.generic.selected).toBe(
    report.wideChoice.generic.optionOrder[report.wideChoice.generic.selectedIndex]
  );
  expect(report.wideChoice.generic.finiteProbabilities).toBe(true);
  expect(report.wideChoice.generic.truncated).toBe(false);
  expect(report.wideChoice.typed.optionOrder).toEqual(wideOptionOrder);
  expect(report.wideChoice.typed.optionOrder).toContain(report.wideChoice.typed.selected);
  expect(report.wideChoice.typed.type).toBe('choice');
  expect(report.wideChoice.typed.finiteConfidence).toBe(true);
  expect(report.wideChoice.typed.finiteProbabilities).toBe(true);
  expect(report.wideChoice.typed.truncated).toBe(false);
  expect(report.manyQuestions.requested).toBe(257);
  expect(report.manyQuestions.generic).toBe(257);
  expect(report.manyQuestions.genericOptionOrderPreserved).toBe(true);
  expect(report.manyQuestions.genericTruncated).toBe(false);
  expect(report.manyQuestions.typed).toBe(257);
  expect(report.manyQuestions.typedQuestionOrder).toEqual(
    Array.from({length: 257}, (_, index) => 'question-' + String(index).padStart(3, '0'))
  );
  expect(report.manyQuestions.typedOptionOrderPreserved).toBe(true);
  expect(report.manyQuestions.typedTruncated).toBe(false);
  expect(report.exactContext.truncated).toBe(false);
  expect(report.exactContext.tokenCount).toBe(134);
  expect(report.overContext.truncated).toBe(true);
  expect(report.exactOption.truncated).toBe(false);
  expect(report.overOption.truncated).toBe(true);
  expect(report.typed.model).toBe('experimental-tamev-micro-minilm-int8');
  expect(Object.keys(report.typed.answers)).toEqual(['route', 'urgent', 'severity']);
  expect(report.typed.answers.route.type).toBe('choice');
  expect(report.typed.answers.urgent.type).toBe('noul');
  expect(report.typed.answers.severity.type).toBe('score');
  expect(report.maximumPermutationDrift).toBeLessThanOrEqual(0.00001);
  expect(Number.isFinite(report.firstInferenceMilliseconds)).toBe(true);
  expect(report.warmMilliseconds).toHaveLength(8);
  expect(report.warmMilliseconds.every(Number.isFinite)).toBe(true);

  if (resultPath) {
    const target = resolve(resultPath);
    await mkdir(dirname(target), {recursive: true});
    await writeFile(target, JSON.stringify({browserName, ...report}, null, 2) + '\n');
  }
});
