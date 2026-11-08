import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {fitSkejjaPlatt} from './calibration.mjs';
import {SKEJJA_MAX_CALIBRATION_EXAMPLES, SKEJJA_MIN_CALIBRATION_EXAMPLES, SKEJJA_INT8_PROFILE} from './contract.mjs';
import {
  finishSkejjaDecision,
  finishSkejjaSystemOne,
  normalizeSkejjaNoulQuestion,
  prepareSkejjaDecision,
  prepareSkejjaSystemOne,
  asSkejjaText
} from './protocol.mjs';

const invalid = (message) => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');

function readSignal(options) {
  if (options == null) return undefined;
  validateDecisionFields(options);
  const descriptor = Object.getOwnPropertyDescriptor(options, 'signal');
  if (descriptor && !Object.hasOwn(descriptor, 'value')) throw invalid('Skejja does not accept signal accessors.');
  const signal = descriptor?.value;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid('Skejja signal must be an AbortSignal.');
  return signal;
}

function snapshotExamples(examples) {
  if (
    !Array.isArray(examples) ||
    examples.length < SKEJJA_MIN_CALIBRATION_EXAMPLES ||
    examples.length > SKEJJA_MAX_CALIBRATION_EXAMPLES
  )
    throw invalid(
      `Skejja calibration needs ${SKEJJA_MIN_CALIBRATION_EXAMPLES} to ${SKEJJA_MAX_CALIBRATION_EXAMPLES} examples.`
    );

  let positives = 0;
  const states = examples.map((example, index) => {
    if (!example || typeof example !== 'object' || Array.isArray(example))
      throw invalid(`Skejja calibration example ${index} must be an object.`);
    const keys = Reflect.ownKeys(example);
    if (keys.some((key) => key !== 'state' && key !== 'answer'))
      throw invalid(`Skejja calibration example ${index} has an unsupported field.`);
    const state = Object.getOwnPropertyDescriptor(example, 'state');
    const answer = Object.getOwnPropertyDescriptor(example, 'answer');
    if (
      !state ||
      !Object.hasOwn(state, 'value') ||
      !answer ||
      !Object.hasOwn(answer, 'value') ||
      ![true, false, 0, 1].includes(answer.value)
    )
      throw invalid(`Skejja calibration example ${index} needs state and a true or false answer.`);
    if (answer.value === true || answer.value === 1) positives++;
    const text = asSkejjaText(state.value);
    return {state: text, answer: answer.value === true || answer.value === 1 ? 1 : 0};
  });
  if (positives === 0 || positives === states.length)
    throw invalid('Skejja calibration needs both true and false answers.');
  return {states, positives};
}

export function createSkejjaSession({runner, tokenizer, profile = SKEJJA_INT8_PROFILE}) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    typeof tokenizer?.encode !== 'function' ||
    typeof tokenizer?.decode !== 'function'
  )
    throw new TypeError(`${profile.name} session components are required.`);

  let queue = Promise.resolve();
  let closed = false;
  let disposing;
  const questionCalibration = new Map();

  const requireOpen = () => {
    if (closed) throw closedError();
  };

  function enqueue(task) {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  }

  async function infer(prepared, signal) {
    requireOpen();
    const schedule = () =>
      enqueue(async () => {
        requireOpen();
        const rows = [];
        for (const question of prepared.questions) {
          signal?.throwIfAborted();
          rows.push(await runner.run(question.inputIds, question.labels.length));
        }
        return rows;
      });
    return signal ? waitForAbort(schedule, signal) : schedule();
  }

  return {
    async decide(request, options = {}) {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const snapshot = validateRequestFields(request, Infinity, Infinity, Infinity, profile.maxChoices);
      const prepared = prepareSkejjaDecision(snapshot, tokenizer, profile);
      const rows = await infer(prepared, signal);
      return finishSkejjaDecision(prepared, rows, profile);
    },
    async systemOne(state, questions, options = {}) {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const prepared = prepareSkejjaSystemOne(state, questions, tokenizer, profile);
      const rows = await infer(prepared, signal);
      return finishSkejjaSystemOne(prepared, rows, profile, questionCalibration);
    },
    async calibrateQuestion(question, examples, options = {}) {
      requireOpen();
      const signal = readSignal(options);
      signal?.throwIfAborted();
      const normalizedQuestion = normalizeSkejjaNoulQuestion(question);
      const snapshots = snapshotExamples(examples);
      const keyQuestion = {calibration: normalizedQuestion};
      const preparedExamples = snapshots.states.map(
        (example) => prepareSkejjaSystemOne(example.state, keyQuestion, tokenizer, profile).questions[0]
      );
      const calibrationKey = preparedExamples[0].calibrationKey;

      const schedule = () =>
        enqueue(async () => {
          requireOpen();
          const logits = [];
          for (const row of preparedExamples) {
            signal?.throwIfAborted();
            logits.push((await runner.run(row.inputIds, row.labels.length))[0]);
          }
          const coefficients = fitSkejjaPlatt(
            logits,
            snapshots.states.map(({answer}) => answer),
            [profile.plattA, profile.plattB]
          );
          questionCalibration.set(calibrationKey, [coefficients.a, coefficients.b]);
          return {...coefficients, examples: snapshots.states.length, positives: snapshots.positives};
        });
      return signal ? waitForAbort(schedule, signal) : schedule();
    },
    async releaseCachedMemory() {
      requireOpen();
      return enqueue(() => runner.releaseCachedMemory());
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await runner.dispose();
        } finally {
          runner = tokenizer = null;
          questionCalibration.clear();
        }
      });
      return disposing;
    }
  };
}
