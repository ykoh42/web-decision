// Corrections: turn stored examples into the `protos` argument of TinyDecide.answer().
// Same recipe as the playground: one prototype per option (the mean of its examples), a centre
// (the mean of every vector seen for this question, or of the examples when none is given), and a
// trust weight `lam` chosen by leave-one-out over the examples themselves.
//
// An example is what answer() returned for a message, plus the option a person said was right:
//   { v: answer.qvec, z: answer.z0 }   stored under lists[k] for the correct option k.
// For a noul question, k = 1 means "true" and k = 0 means "false".

import type {TinyDecidePrototype} from './engine.js';

interface CorrectionExample {
  readonly v: readonly number[];
  readonly z: readonly number[];
}

type CorrectionLists = readonly (readonly CorrectionExample[])[];

const K_BUCKETS = [1, 2, 4, 8];
export const bucketK = (c: number): number => K_BUCKETS.reduce((b, e) => (c > e ? b + 1 : b), 0);

export function meanVec(list: readonly CorrectionExample[], dh: number): number[] {
  // Keep the upstream constructor's length validation and behavior.
  // oxlint-disable-next-line unicorn/no-new-array
  const m = new Array(dh).fill(0);
  for (const e of list) for (let i = 0; i < dh; i++) m[i]! += e.v[i]! / list.length;
  return m;
}

function cosC(a: readonly number[], b: readonly number[], c: readonly number[]): number {
  let d = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]! - c[i]!,
      y = b[i]! - c[i]!;
    d += x * y;
    na += x * x;
    nb += y * y;
  }
  return d / (Math.sqrt(na * nb) || 1e-12);
}

function termFor(
  v: readonly number[],
  lists: CorrectionLists,
  c: readonly number[],
  beta: readonly number[]
): number[] {
  return lists.map((l) => (l.length ? beta[bucketK(l.length)]! * cosC(v, meanVec(l, v.length), c) : 0));
}

export const LAMBDAS = [0, 0.25, 0.5, 1, 2];

// How much to trust this question's corrections: leave-one-out log-likelihood over the examples.
export function lambdaFor(type: string, lists: CorrectionLists, c: readonly number[], beta: readonly number[]): number {
  const all: Array<[number, number, CorrectionExample]> = [];
  lists.forEach((l, k) => l.forEach((e, j) => all.push([k, j, e])));
  if (all.length < 2) return 0.25;
  let best = 0,
    bestLL = -Infinity;
  for (const lam of LAMBDAS) {
    let ll = 0;
    for (const [k, j, e] of all) {
      const rest = lists.map((l, kk) => (kk === k ? l.filter((_, jj) => jj !== j) : l));
      const t = termFor(e.v, rest, c, beta);
      const z = type === 'noul' ? [lam * t[0]!, e.z[1]! + lam * t[1]!] : e.z.map((x, i) => x + lam * t[i]!);
      const m = Math.max(...z),
        lse = m + Math.log(z.reduce((a, x) => a + Math.exp(x - m), 0));
      ll += z[k]! - lse;
    }
    if (ll > bestLL + 1e-9) {
      best = lam;
      bestLL = ll;
    }
  }
  return best;
}

// lists: one array of examples per option (2 for noul). beta: meta.beta. center: optional mean qvec.
// Returns {vec, cnt, center, lam} or null when there are no examples (span questions take none).
export function makeProtos(
  type: string,
  lists: CorrectionLists,
  beta: readonly number[],
  center: readonly number[] | null = null
): TinyDecidePrototype | null {
  if (type === 'span' || !lists.some((l) => l.length)) return null;
  const dh = lists.find((l) => l.length)![0]!.v.length,
    K = lists.length;
  const c = center || meanVec(lists.flat(), dh);
  const vec = new Float32Array(K * dh),
    cnt = new Int32Array(lists.map((l) => l.length));
  lists.forEach((l, k) => {
    if (l.length)
      meanVec(l, dh).forEach((x, i) => {
        vec[k * dh + i] = x;
      });
  });
  return {vec, cnt, center: Float32Array.from(c), lam: lambdaFor(type, lists, c, beta)};
}
