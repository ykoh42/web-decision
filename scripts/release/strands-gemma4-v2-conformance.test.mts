import assert from 'node:assert/strict';
import test from 'node:test';
import {
  gemma4V2Agreement,
  inspectGemma4V2Conformance,
  readGemmaCandidate,
  selectGemmaGoldenRows
} from './strands-gemma4-v2-conformance.mts';

test('upstream agreement chooses the first code on ties and averages each question maximum', () => {
  const reference = new Map([
    ['a', {codes: ['A', 'B'], probabilities: [0.8, 0.2]}],
    ['b', {codes: ['A', 'B'], probabilities: [0.7, 0.3]}]
  ]);
  const actual = new Map([
    ['a', {codes: ['A', 'B'], probabilities: [0.9, 0.1]}],
    ['b', {codes: ['A', 'B'], probabilities: [0.4, 0.6]}]
  ]);
  const result = gemma4V2Agreement(reference, actual);
  assert.equal(result.questions, 2);
  assert.equal(result.top1Agreement, 0.5);
  assert.ok(Math.abs(result.meanAbsDp - 0.2) < 1e-12);
  assert.equal(result.meetsUpstreamMetricThresholds, false);
  const tie = new Map([['q', {codes: ['A', 'B'], probabilities: [0.5, 0.5]}]]);
  const first = new Map([['q', {codes: ['A', 'B'], probabilities: [0.6, 0.4]}]]);
  assert.equal(gemma4V2Agreement(tie, first).top1Agreement, 1);
});

test('upstream thresholds are inclusive but small samples are not conformant', () => {
  const reference = new Map<string, {codes: string[]; probabilities: number[]}>();
  const candidate = new Map<string, {codes: string[]; probabilities: number[]}>();
  for (let index = 0; index < 1000; index++) {
    reference.set(String(index), {codes: ['A', 'B'], probabilities: [0.5, 0.5]});
    candidate.set(String(index), {codes: ['A', 'B'], probabilities: index < 20 ? [0.49, 0.51] : [0.5, 0.5]});
  }
  const boundary = gemma4V2Agreement(reference, candidate);
  assert.equal(boundary.top1Agreement, 0.98);
  assert.equal(boundary.meetsUpstreamMetricThresholds, true);
  candidate.set('20', {codes: ['A', 'B'], probabilities: [0.49, 0.51]});
  assert.equal(gemma4V2Agreement(reference, candidate).meetsUpstreamMetricThresholds, false);
  reference.delete('999');
  candidate.delete('999');
  assert.equal(gemma4V2Agreement(reference, candidate).meetsUpstreamMetricThresholds, false);
});

test('mismatched identity, code order, lengths and synthetic golden fail', () => {
  const ref = new Map([['q', {codes: ['A', 'B'], probabilities: [0.5, 0.5]}]]);
  assert.throws(() => gemma4V2Agreement(ref, new Map()), /same questions/u);
  assert.throws(
    () => gemma4V2Agreement(ref, new Map([['other', {codes: ['A', 'B'], probabilities: [0, 1]}]])),
    /Missing/u
  );
  assert.throws(
    () => gemma4V2Agreement(ref, new Map([['q', {codes: ['B', 'A'], probabilities: [0, 1]}]])),
    /Code order/u
  );
  assert.throws(() => gemma4V2Agreement(ref, new Map([['q', {codes: ['A', 'B'], probabilities: [1]}]])), /width/u);
  assert.throws(() => inspectGemma4V2Conformance(Buffer.from('{}'), Buffer.from('{}')), /golden bytes differ/u);
});

test('declared subset takes whole ordered original rows and rejects duplicate or unknown rows', () => {
  const reference = new Map([
    [JSON.stringify([0, 'a']), {codes: ['A', 'B'], probabilities: [0.9, 0.1]}],
    [JSON.stringify([0, 'b']), {codes: ['A', 'B'], probabilities: [0.8, 0.2]}],
    [JSON.stringify([1, 'a']), {codes: ['A', 'B'], probabilities: [0.7, 0.3]}],
    [JSON.stringify([2, 'a']), {codes: ['A', 'B'], probabilities: [0.6, 0.4]}]
  ]);
  const selected = selectGemmaGoldenRows(reference, [0, 2]);
  assert.equal(selected.size, 3);
  assert.ok(selected.has(JSON.stringify([0, 'b'])));
  assert.throws(() => selectGemmaGoldenRows(reference, [0, 0]), /unique, sorted/u);
  assert.throws(() => selectGemmaGoldenRows(reference, [2, 0]), /unique, sorted/u);
  assert.throws(() => selectGemmaGoldenRows(reference, [0, 3]), /absent/u);
  assert.throws(() => selectGemmaGoldenRows(reference, []), /must contain/u);
});

test('candidate parser rejects duplicate rows, split questions and nonfinite values', () => {
  const line = (number: number, questions: unknown) => JSON.stringify({line: number, questions});
  const a = {a: {codes: ['A', 'B'], probabilities: [0.7, 0.3]}};
  const b = {b: {codes: ['A', 'B'], probabilities: [0.3, 0.7]}};
  assert.throws(() => readGemmaCandidate(Buffer.from(`${line(0, a)}\n${line(0, b)}\n`)), /Duplicate candidate row/u);
  assert.throws(() => readGemmaCandidate(Buffer.from(line(0, {}))), /no questions/u);
  assert.throws(() => readGemmaCandidate(Buffer.from(line(0, {a: {codes: ['A', 'B'], probabilities: [null, 1]}}))));
  const source = new Map([
    [JSON.stringify([0, 'a']), a.a],
    [JSON.stringify([0, 'b']), b.b]
  ]);
  const incomplete = readGemmaCandidate(Buffer.from(line(0, a)));
  assert.throws(() => gemma4V2Agreement(source, incomplete), /same questions/u);
});
