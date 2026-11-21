import type {Mapika2bResult} from '../../packages/web-decision/src/mapika2b/readout.mts';

export interface MapikaReferenceSource {
  readonly repository: string;
  readonly revision: string;
  readonly deciderRevision: string;
}

interface ReferenceRequest {
  readonly state: unknown;
  readonly questions: Readonly<Record<string, Record<string, unknown>>>;
  readonly independent?: boolean;
  readonly assets?: Readonly<Record<string, unknown>>;
}

interface ReferenceRow {
  readonly inputIds: readonly number[];
  readonly slots: readonly number[];
  readonly optionCounts: readonly number[];
  readonly temperatures: readonly number[];
}

interface ReferenceQuestion {
  readonly type: string;
  readonly names: readonly (string | number | boolean)[];
}

interface ReferenceCaseBase {
  readonly id: string;
  readonly request: ReferenceRequest;
}

export interface AcceptedMapikaCase extends ReferenceCaseBase {
  readonly status: 'accepted';
  readonly questionIndex: readonly (readonly [string, string, number, number])[];
  readonly renderedQuestions: Readonly<Record<string, ReferenceQuestion>>;
  readonly rows: readonly ReferenceRow[];
  readonly usage: Mapika2bResult['usage'];
}

export interface RejectedMapikaCase extends ReferenceCaseBase {
  readonly status: 'rejected';
  readonly message: string;
}

export type MapikaReferenceCase = AcceptedMapikaCase | RejectedMapikaCase;
export type ValidLogits = readonly (readonly (readonly number[])[])[];

export interface MapikaAnswerReference {
  readonly id: string;
  readonly result: Mapika2bResult;
  readonly rows: readonly {readonly logits: readonly (readonly number[])[]}[];
}

export interface MapikaReference<T> {
  readonly source: MapikaReferenceSource;
  readonly cases: readonly T[];
}

export type MapikaBrowserResult =
  | {
      readonly status: 'accepted';
      readonly result: Mapika2bResult;
      readonly questionNames: readonly string[];
      readonly rows: readonly ReferenceRow[];
      readonly logits: ValidLogits;
    }
  | {readonly status: 'rejected'; readonly message: string; readonly code?: string};

declare global {
  interface Window {
    mapika2bGpuHarness?: MapikaBrowserHarness;
  }
}

export type MapikaCacheMode = 'none' | 'fill' | 'offline';
export const MAPIKA2B_TEST_CACHE = 'web-decision-mapika2b-qa-v1';

export interface MapikaBrowserHarness {
  stage(): Promise<void>;
  initialize(installedArchive: boolean, cacheMode?: MapikaCacheMode): Promise<{readonly wasmSha256: string}>;
  curated(
    item: MapikaReferenceCase,
    cacheMode?: MapikaCacheMode
  ): Promise<{
    readonly result: Mapika2bResult;
    readonly progress: readonly {
      readonly loadedBytes: number;
      readonly totalBytes: number;
      readonly fraction: number;
    }[];
  }>;
  evaluate(item: MapikaReferenceCase): Promise<MapikaBrowserResult>;
  dispose(): Promise<void>;
}

function orderedQuestion(item: AcceptedMapikaCase, name: string): Record<string, unknown> {
  const question = {...item.request.questions[name]!};
  const criteria = question.criteria;
  const rendered = item.renderedQuestions[name]!;
  if (rendered.type !== 'choice' || !criteria || typeof criteria !== 'object' || Array.isArray(criteria))
    return question;
  const values = criteria as Record<string, unknown>;
  return {...question, criteria: new Map(rendered.names.map((option) => [String(option), values[String(option)]]))};
}

export function requestFromMapikaFixture(item: MapikaReferenceCase): unknown {
  if (item.status === 'rejected') return item.request;
  // JSON object enumeration reorders integer-like keys in JS. The oracle records the
  // upstream insertion order for both the questions and their option names.
  const questions = new Map(item.questionIndex.map(([name]) => [name, orderedQuestion(item, name)]));
  return {...item.request, questions};
}

function logitDeltas(actual: ValidLogits, expected: ValidLogits): number[] {
  if (actual.length !== expected.length) throw new Error('Mapika logit row count differs from the oracle.');
  return expected.flatMap((row, rowIndex) => {
    const observed = actual[rowIndex]!;
    if (row.length !== observed.length) throw new Error('Mapika logit slot count differs from the oracle.');
    return row.flatMap((slot, slotIndex) => {
      const observedSlot = observed[slotIndex]!;
      if (slot.length !== observedSlot.length) throw new Error('Mapika option logit count differs from the oracle.');
      return slot.map((value, option) => Math.abs(observedSlot[option]! - value));
    });
  });
}

export function summarizeMapikaLogits(actual: ValidLogits, expected: ValidLogits) {
  const deltas = logitDeltas(actual, expected);
  const maxAbs = deltas.reduce((max, value) => Math.max(max, value), 0);
  const sumSquared = deltas.reduce((sum, value) => sum + value * value, 0);
  return {count: deltas.length, maxAbs, rms: deltas.length ? Math.sqrt(sumSquared / deltas.length) : 0};
}

export function summarizeMapikaAnswerParity(actual: Mapika2bResult, official: Mapika2bResult) {
  const matches: Record<string, boolean> = {};
  for (const [name, truth] of Object.entries(official.answers)) {
    const observed = actual.answers[name];
    const sameType = observed?.type === truth.type;
    const sameChoice = truth.type !== 'choice' || (observed?.type === 'choice' && observed.choice === truth.choice);
    const sameScore = truth.type !== 'score' || (observed?.type === 'score' && observed.score === truth.score);
    const sameNoul = truth.type !== 'noul' || (observed?.type === 'noul' && observed.noul === truth.noul);
    Object.defineProperty(matches, name, {enumerable: true, value: sameType && sameChoice && sameScore && sameNoul});
  }
  return matches;
}

/** Compare exact model structure and decisions, allowing one four-decimal output step. */
export function mapikaReferenceMismatches(actual: unknown, expected: unknown): string[] {
  const mismatches: string[] = [];
  function compare(left: unknown, right: unknown, path: readonly string[]): void {
    if (typeof right === 'number' && typeof left === 'number') {
      const exact = path[0] === 'usage' || (path[0] === 'answers' && path.length === 3 && path[2] === 'score');
      const tolerance = exact ? 0 : 0.0001 + Number.EPSILON;
      if (!Number.isFinite(left) || !Number.isFinite(right) || Math.abs(left - right) > tolerance)
        mismatches.push(path.join('.'));
      return;
    }
    if (right && left && typeof right === 'object' && typeof left === 'object') {
      const actualKeys = Object.keys(left).sort();
      const expectedKeys = Object.keys(right).sort();
      if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
        mismatches.push(path.join('.') + ' keys');
        return;
      }
      for (const key of expectedKeys)
        compare((left as Record<string, unknown>)[key], (right as Record<string, unknown>)[key], [...path, key]);
      return;
    }
    if (left !== right) mismatches.push(path.join('.'));
  }
  compare(actual, expected, []);
  return mismatches;
}
