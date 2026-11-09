import {waitForAbort} from '../assets.js';
import {WebDecisionError} from '../kev.js';
import {validateDecisionFields, validateRequestFields} from '../request-fields.mjs';
import {TAMEV_MICRO_PROFILE} from './contract.mjs';
import type {TAMEV_PROFILES} from './contract.mjs';

type TamevProfile = (typeof TAMEV_PROFILES)[keyof typeof TAMEV_PROFILES];
type TamevSignal = Pick<AbortSignal, 'throwIfAborted' | 'addEventListener' | 'removeEventListener'>;
interface TamevTokenRow {
  readonly ids: readonly number[];
  readonly truncated: boolean;
}
interface TamevTokenizer {
  encodeContext(text: string): TamevTokenRow;
  encodeOption(text: string): TamevTokenRow;
}
interface TamevDecisionRow {
  readonly logits: readonly number[];
  readonly probabilities: readonly number[];
}
interface TamevRunner {
  run(contextIds: readonly number[], optionRows: readonly (readonly number[])[]): PromiseLike<TamevDecisionRow>;
  dispose(): unknown | PromiseLike<unknown>;
  releaseCachedMemory(): unknown | PromiseLike<unknown>;
}
interface TamevConfig {
  readonly calibration_by_k: Readonly<Record<string, number | undefined>>;
}
interface TamevPrepared {
  readonly questions: readonly {
    readonly contextIds: readonly number[];
    readonly optionRows: readonly (readonly number[])[];
  }[];
}
interface TamevSessionOptions {
  runner: TamevRunner | null;
  tokenizer: TamevTokenizer | null;
  config: TamevConfig | null;
  profile?: TamevProfile | undefined;
  model?: string | undefined;
}
import {calibrateTamevProbabilities, finishTamevDecision, prepareTamevDecision} from './protocol.mjs';
import {finishTamevSystemOne, prepareTamevSystemOne} from './typed.mjs';

const invalid = (message: string): WebDecisionError => new WebDecisionError('INVALID_INPUT', message);
const closedError = () => new WebDecisionError('SESSION_CLOSED', 'The model has been disposed.');

function readSignal(options: unknown, profile: TamevProfile): TamevSignal | null | undefined {
  if (options == null) return undefined;
  validateDecisionFields(options);
  const descriptor = Object.getOwnPropertyDescriptor(options as object, 'signal');
  if (descriptor && !Object.hasOwn(descriptor, 'value'))
    throw invalid(`${profile.name} does not accept signal accessors.`);
  const signal = descriptor?.value;
  if (
    signal != null &&
    (typeof signal.throwIfAborted !== 'function' ||
      typeof signal.addEventListener !== 'function' ||
      typeof signal.removeEventListener !== 'function')
  )
    throw invalid(`${profile.name} signal must be an AbortSignal.`);
  return signal;
}

export function createTamevSession({
  runner,
  tokenizer,
  config,
  profile = TAMEV_MICRO_PROFILE,
  model = profile.model
}: TamevSessionOptions) {
  if (
    typeof runner?.run !== 'function' ||
    typeof runner?.dispose !== 'function' ||
    typeof runner?.releaseCachedMemory !== 'function' ||
    typeof tokenizer?.encodeContext !== 'function' ||
    typeof tokenizer?.encodeOption !== 'function' ||
    !config?.calibration_by_k ||
    model !== profile.model
  )
    throw new TypeError(`${profile.name} session components are required.`);

  let queue: Promise<unknown> = Promise.resolve();
  let closed = false;
  let disposing: Promise<void> | undefined;

  const requireOpen = () => {
    if (closed) throw closedError();
  };

  function enqueue<T>(task: () => T | PromiseLike<T>): Promise<T> {
    const current = queue.then(task);
    queue = current.catch(() => {});
    return current;
  }

  async function infer(
    prepared: TamevPrepared,
    signal: TamevSignal | null | undefined,
    applyOptionCalibration: boolean
  ) {
    requireOpen();
    const schedule = () =>
      enqueue(async () => {
        requireOpen();
        const rows = [];
        for (const question of prepared.questions) {
          signal?.throwIfAborted();
          const row = await (runner as TamevRunner).run(question.contextIds, question.optionRows);
          rows.push({
            ...row,
            probabilities: applyOptionCalibration
              ? calibrateTamevProbabilities(row.probabilities, question.optionRows.length, config as TamevConfig)
              : row.probabilities
          });
        }
        return rows;
      });
    return signal ? waitForAbort(schedule, signal as AbortSignal) : schedule();
  }

  return {
    async decide(request: unknown, options: unknown = {}) {
      requireOpen();
      const signal = readSignal(options, profile);
      signal?.throwIfAborted();
      const checked = validateRequestFields(request, Infinity, Infinity, Infinity, Infinity, Infinity, {
        dataPropertiesOnly: true,
        allowInheritedDataProperties: true
      });
      const prepared = prepareTamevDecision(tokenizer as TamevTokenizer, checked, profile);
      const rows = await infer(prepared, signal, true);
      return finishTamevDecision(prepared, rows);
    },
    async systemOne(state: unknown, questions: unknown, options: unknown = {}) {
      requireOpen();
      const signal = readSignal(options, profile);
      signal?.throwIfAborted();
      const prepared = (
        prepareTamevSystemOne as unknown as (
          state: unknown,
          questions: unknown,
          tokenizer: TamevTokenizer | null,
          profile: TamevProfile
        ) => ReturnType<typeof prepareTamevSystemOne>
      )(state, questions, tokenizer, profile);
      const rows = await infer(prepared, signal, false);
      return finishTamevSystemOne(prepared, rows, model);
    },
    async releaseCachedMemory() {
      requireOpen();
      return enqueue(() => (runner as TamevRunner).releaseCachedMemory());
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = queue.then(async () => {
        try {
          await (runner as TamevRunner).dispose();
        } finally {
          runner = tokenizer = config = null;
        }
      });
      return disposing;
    }
  };
}
