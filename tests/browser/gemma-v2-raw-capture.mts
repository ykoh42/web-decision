import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import type {Page} from '@playwright/test';
import {sha256} from '../../scripts/release/curated-gpu-evidence.mjs';

export const gemmaV2RawProbabilityEncoding = 'chromium-cdp-unrounded-fp32';
const scorerSignature = 'function finishGemma4V2Answer(question, logits, config) {';
const probabilityAssignment = 'const probabilities = gemma4V2Probabilities(logits, count, config);';
const followingStatement = 'if (question.kind === "noul") return {';

type ScorerLocation = Readonly<{lineNumber: number; columnNumber: number; conditionEndColumn: number}>;
type ScorerPin = ScorerLocation & Readonly<{path: string; sha256: string}>;
export type RawGemmaFrame = Readonly<{name: string; options: string[]; codes: string[]; probabilities: number[]}>;

function scorerLocation(source: string) {
  const lines = source.split('\n');
  const start = lines.reduce<number[]>((matches, line, index) => {
    if (line === scorerSignature) matches.push(index);
    return matches;
  }, []);
  if (start.length === 0) return null;
  assert.equal(start.length, 1, 'Expected one installed Gemma scorer function.');
  const end = lines.indexOf('}', start[0]! + 1);
  assert.ok(end > start[0]!, 'The installed Gemma scorer scope could not be pinned.');
  const statements = lines.slice(start[0]! + 1, end).map((line) => line.trimStart());
  const assignment = statements.indexOf(probabilityAssignment);
  assert.ok(assignment >= 0, 'The installed Gemma scorer probability assignment changed.');
  assert.equal(
    statements.lastIndexOf(probabilityAssignment),
    assignment,
    'Ambiguous installed Gemma scorer assignment.'
  );
  assert.equal(
    statements[assignment + 1],
    followingStatement,
    'The installed Gemma scorer breakpoint statement changed.'
  );
  // Break at the start of the conditional so both outcomes execute the site.
  const lineNumber = start[0]! + assignment + 2;
  const line = lines[lineNumber]!;
  return {lineNumber, columnNumber: line.indexOf('if'), conditionEndColumn: line.indexOf('return')};
}

export async function findGemmaV2ScorerPin(installedRoot: string, archiveFiles: readonly string[]): Promise<ScorerPin> {
  const matches: ScorerPin[] = [];
  for (const path of archiveFiles.filter((file) => file.startsWith('dist/') && file.endsWith('.js'))) {
    const bytes = await readFile(join(installedRoot, path));
    const statement = scorerLocation(new TextDecoder('utf-8', {fatal: true, ignoreBOM: true}).decode(bytes));
    if (statement) matches.push({path, sha256: sha256(bytes), ...statement});
  }
  assert.equal(matches.length, 1, 'The installed archive must contain exactly one unminified Gemma scorer.');
  return matches[0]!;
}

export async function openGemmaV2RawCapture(page: Page, scriptUrl: string, pin: ScorerPin) {
  const client = await page.context().newCDPSession(page);
  const scripts = new Set<string>();
  client.on('Debugger.scriptParsed', (script) => {
    if (script.url === scriptUrl) scripts.add(script.scriptId);
  });
  try {
    await client.send('Debugger.enable');
    assert.equal(scripts.size, 1, 'Chromium must parse exactly one copy of the installed Gemma scorer script.');
    const scriptId = [...scripts][0]!;
    const {scriptSource} = await client.send('Debugger.getScriptSource', {scriptId});
    assert.equal(
      sha256(Buffer.from(scriptSource, 'utf8')),
      pin.sha256,
      'Chromium Gemma scorer bytes differ from the installed archive.'
    );
    const {lineNumber, columnNumber, conditionEndColumn} = pin;
    assert.deepEqual(scorerLocation(scriptSource), {lineNumber, columnNumber, conditionEndColumn});
    // V8 may expose the conditional's start or its property load. Both execute for every kind.
    const {locations} = await client.send('Debugger.getPossibleBreakpoints', {
      start: {scriptId, lineNumber, columnNumber},
      end: {scriptId, lineNumber, columnNumber: conditionEndColumn},
      restrictToFunction: true
    });
    assert.equal(locations.length, 1, 'Chromium must provide exactly one breakpoint in the pinned Gemma condition.');
    const breakpointColumn = locations[0]!.columnNumber;
    assert.ok(
      breakpointColumn !== undefined && breakpointColumn >= columnNumber && breakpointColumn < conditionEndColumn
    );
    const expectedLocation = {scriptId, lineNumber, columnNumber: breakpointColumn};
    assert.equal(locations[0]!.scriptId, scriptId);
    assert.equal(locations[0]!.lineNumber, lineNumber);
    const {breakpointId, actualLocation} = await client.send('Debugger.setBreakpoint', {location: expectedLocation});
    assert.deepEqual(actualLocation, expectedLocation, 'Chromium moved the Gemma scorer breakpoint.');
    let active: RawGemmaFrame[] | undefined;
    let failure: unknown;
    let totalFrames = 0;

    client.on('Debugger.paused', async (event) => {
      try {
        assert.deepEqual(event.hitBreakpoints, [breakpointId], 'Unexpected Chromium breakpoint.');
        const frame = event.callFrames[0];
        assert.ok(frame && frame.functionName === 'finishGemma4V2Answer', 'Unexpected Chromium scorer frame.');
        assert.deepEqual(frame.location, expectedLocation, 'Chromium stopped outside the exact scorer statement.');
        assert.ok(active, 'Chromium reached the scorer outside a recorded request.');
        const {exceptionDetails, result} = await client.send('Debugger.evaluateOnCallFrame', {
          callFrameId: frame.callFrameId,
          expression: '({name: question.name, options: question.options, codes: question.codes, probabilities})',
          throwOnSideEffect: true,
          returnByValue: true
        });
        assert.equal(exceptionDetails, undefined, 'Chromium could not read scorer state without side effects.');
        const values: unknown = result.value;
        assert.ok(values && typeof values === 'object', 'Missing Chromium scorer state.');
        const readout = values as RawGemmaFrame;
        assert.ok(typeof readout.name === 'string' && Array.isArray(readout.codes) && Array.isArray(readout.options));
        assert.ok(
          Array.isArray(readout.probabilities) &&
            readout.probabilities.every(
              (value) =>
                typeof value === 'number' &&
                Number.isFinite(value) &&
                value >= 0 &&
                value <= 1 &&
                Math.fround(value) === value
            ),
          'Chromium scorer values are not unrounded finite probabilities.'
        );
        active.push(readout);
      } catch (error) {
        failure ??= error;
      } finally {
        try {
          await client.send('Debugger.resume');
        } catch (error) {
          failure ??= error;
        }
      }
    });

    return {
      async capture<T>(expectedQuestions: readonly string[], invoke: () => Promise<T>) {
        assert.equal(active, undefined, 'Chromium scorer capture is already recording a request.');
        assert.equal(failure, undefined);
        active = [];
        try {
          const result = await invoke();
          if (failure !== undefined) throw failure;
          assert.deepEqual(
            active.map(({name}) => name),
            expectedQuestions,
            'Chromium scorer question identities differ.'
          );
          totalFrames += active.length;
          return {result, frames: active};
        } finally {
          active = undefined;
        }
      },
      evidence() {
        return {...pin, breakpointColumn, frames: totalFrames};
      },
      async close() {
        await client.detach();
        if (failure !== undefined) throw failure;
      }
    };
  } catch (error) {
    await client.detach();
    throw error;
  }
}
