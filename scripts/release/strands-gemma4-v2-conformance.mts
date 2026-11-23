import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

export const GEMMA_V2_OFFICIAL_GOLDEN_SHA256 = '658445826fa307b949d88722f5a312eb39b6354cbdfe62102d4d4982dfc92ded';
export const GEMMA_V2_OFFICIAL_SUBSET_PATH = 'tests/fixtures/strands-gemma4-v2-official-subset.json';
export const GEMMA_V2_OFFICIAL_SUBSET_SHA256 = '9dde7b0e7d4bbc3ccfb40fbe474d5b52eae7b365a321eebe8eff87675b1d8e27';
const sha256 = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
type Row = {codes: string[]; probabilities: number[]};
type GoldenLine = {questions: Record<string, Row>};
type CandidateLine = {line: number; questions: Record<string, Row>};
type Rows = Map<string, Row>;

function addQuestions(rows: Rows, line: number, questions: Record<string, Row>) {
  assert.ok(questions && typeof questions === 'object' && !Array.isArray(questions), 'Invalid questions.');
  for (const [name, row] of Object.entries(questions)) {
    const key = JSON.stringify([line, name]);
    assert.ok(!rows.has(key), `Duplicate question ${key}.`);
    assert.ok(Array.isArray(row.codes) && Array.isArray(row.probabilities), `Missing code probabilities: ${key}.`);
    assert.ok(row.codes.length > 0 && row.codes.length === row.probabilities.length, `Invalid row length: ${key}.`);
    assert.ok(row.codes.every((code) => typeof code === 'string') && new Set(row.codes).size === row.codes.length);
    assert.ok(row.probabilities.every((value) => typeof value === 'number' && Number.isFinite(value)));
    rows.set(key, row);
  }
}

function readGolden(bytes: Uint8Array): Rows {
  const rows: Rows = new Map();
  new TextDecoder()
    .decode(bytes)
    .split(/\r?\n/u)
    .forEach((text, line) => {
      if (text.trim()) addQuestions(rows, line, (JSON.parse(text) as GoldenLine).questions);
    });
  return rows;
}

export function readPinnedGemma4V2Subset(bytes: Uint8Array): number[] {
  assert.equal(sha256(bytes), GEMMA_V2_OFFICIAL_SUBSET_SHA256, 'The predeclared Gemma subset bytes differ.');
  const value = JSON.parse(new TextDecoder().decode(bytes)) as {rows: number[]};
  assert.equal(value.rows.length, 360);
  return value.rows;
}

export function readGemmaCandidate(bytes: Uint8Array): Rows {
  const rows: Rows = new Map();
  const seen = new Set<number>();
  for (const text of new TextDecoder().decode(bytes).split(/\r?\n/u)) {
    if (!text.trim()) continue;
    const row = JSON.parse(text) as CandidateLine;
    assert.ok(Number.isSafeInteger(row.line) && row.line >= 0, 'Candidate line must be a nonnegative integer.');
    assert.ok(!seen.has(row.line), `Duplicate candidate row ${row.line}.`);
    seen.add(row.line);
    assert.ok(row.questions && Object.keys(row.questions).length > 0, 'Candidate row has no questions.');
    addQuestions(rows, row.line, row.questions);
  }
  return rows;
}

/** Restrict to an explicitly declared set of complete original requests. */
export function selectGemmaGoldenRows(golden: Rows, rows: readonly number[]): Rows {
  assert.ok(Array.isArray(rows) && rows.length > 0, 'Selection must contain original row numbers.');
  let previous = -1;
  for (const row of rows) {
    assert.ok(
      Number.isSafeInteger(row) && row > previous,
      'Selected row numbers must be unique, sorted, nonnegative integers.'
    );
    previous = row;
  }
  const selection = new Set(rows);
  const found = new Set<number>();
  const selected: Rows = new Map();
  for (const [key, value] of golden) {
    const [row] = JSON.parse(key) as [number, string];
    if (selection.has(row)) {
      found.add(row);
      selected.set(key, value);
    }
  }
  assert.equal(found.size, selection.size, 'Selection contains an absent golden row.');
  return selected;
}

function originalRows(rows: Rows) {
  return new Set(Array.from(rows.keys(), (key) => (JSON.parse(key) as [number, string])[0]));
}

function top1(values: number[]) {
  let best = 0;
  for (let index = 1; index < values.length; index++) if (values[index]! > values[best]!) best = index;
  return best;
}

/** Exact calculation from the pinned upstream tests/core_conformance.py, without release approval. */
export function gemma4V2Agreement(reference: Rows, candidate: Rows) {
  assert.ok(reference.size > 0 && reference.size === candidate.size, 'Both runtimes must answer the same questions.');
  let matched = 0;
  let sumMaxDifference = 0;
  for (const [key, expected] of reference) {
    const actual = candidate.get(key);
    assert.ok(actual, `Missing candidate question ${key}.`);
    assert.deepEqual(actual.codes, expected.codes, `Code order differs for ${key}.`);
    assert.equal(actual.probabilities.length, expected.probabilities.length, `Probability width differs for ${key}.`);
    if (top1(actual.probabilities) === top1(expected.probabilities)) matched++;
    sumMaxDifference += Math.max(...expected.probabilities.map((p, i) => Math.abs(actual.probabilities[i]! - p)));
  }
  const questions = reference.size;
  const top1Agreement = matched / questions;
  const meanAbsDp = sumMaxDifference / questions;
  return {
    questions,
    top1Agreement,
    meanAbsDp,
    meetsUpstreamMetricThresholds: questions >= 1000 && top1Agreement >= 0.98 && meanAbsDp <= 0.02
  };
}

/** Calculate metrics only; source provenance and exact installed archive must be verified separately. */
export function inspectGemma4V2Conformance(
  goldenBytes: Uint8Array,
  candidateBytes: Uint8Array,
  selection?: {bytes: Uint8Array; sha256: string}
) {
  assert.equal(sha256(goldenBytes), GEMMA_V2_OFFICIAL_GOLDEN_SHA256, 'Official checkpoint golden bytes differ.');
  const golden = readGolden(goldenBytes);
  assert.equal(golden.size, 1426, 'Unexpected official golden question count.');
  let selected = golden;
  if (selection) {
    assert.match(selection.sha256, /^[a-f0-9]{64}$/u, 'A selection SHA-256 is required.');
    assert.equal(sha256(selection.bytes), selection.sha256, 'Selection bytes differ from declared SHA-256.');
    const definition = JSON.parse(new TextDecoder().decode(selection.bytes)) as {rows: number[]};
    assert.deepEqual(Object.keys(definition), ['rows'], 'Selection must contain only row numbers.');
    selected = selectGemmaGoldenRows(golden, definition.rows);
  }
  const metrics = gemma4V2Agreement(selected, readGemmaCandidate(candidateBytes));
  const allRows = originalRows(golden);
  const selectedOriginalRows = originalRows(selected);
  const omittedRows = [...allRows].filter((row) => !selectedOriginalRows.has(row));
  const totalRows = allRows.size;
  const selectedRows = selectedOriginalRows.size;
  return {
    status: 'metrics-only',
    releaseApproved: false,
    candidateProvenanceVerified: false,
    checkpointRevision: 'a5f32ea1bffe2e91c25c0c641d444e8115e28212',
    upstreamRevision: 'c3c845782095b31d438963ac2afcdfe41b49f757',
    goldenSha256: GEMMA_V2_OFFICIAL_GOLDEN_SHA256,
    candidateSha256: sha256(candidateBytes),
    selectionSha256: selection?.sha256 ?? null,
    scope: selectedRows === totalRows ? 'full-corpus' : 'declared-subset',
    selectedRows,
    missingRows: totalRows - selectedRows,
    omittedRows,
    totalRows,
    missingQuestions: golden.size - selected.size,
    totalQuestions: golden.size,
    ...metrics
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [golden, candidate, ...extra] = process.argv.slice(2);
  assert.ok(golden && candidate, 'Expected official golden JSONL and candidate code-probability JSONL.');
  let selection: {bytes: Uint8Array; sha256: string} | undefined;
  if (extra.length) {
    assert.ok(
      extra.length === 4 && extra[0] === '--subset' && extra[2] === '--subset-sha256',
      'Expected --subset <file> --subset-sha256 <sha256>.'
    );
    selection = {bytes: await readFile(extra[1]!), sha256: extra[3]!};
  }
  console.log(
    JSON.stringify(inspectGemma4V2Conformance(await readFile(golden), await readFile(candidate), selection), null, 2)
  );
}
