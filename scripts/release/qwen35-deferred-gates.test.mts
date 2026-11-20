import assert from 'node:assert/strict';
import test from 'node:test';
import {recordQwen35GateFailure, writeQwen35ReportThenThrow, type Qwen35GateFailure} from './qwen35-deferred-gates.mts';

test('a failed source parity check does not stop offline or adapter checks', async () => {
  const failures: Qwen35GateFailure[] = [];
  const completed: string[] = [];

  await recordQwen35GateFailure(failures, 'online', 'source parity', () => {
    completed.push('parity');
    throw new Error('vision confidence differs');
  });
  await recordQwen35GateFailure(failures, 'online', 'diagnostic update', () => {
    completed.push('diagnostic update');
    throw new Error('unable to rewrite diagnostic');
  });
  await recordQwen35GateFailure(failures, 'offline', 'cache reuse', () => {
    completed.push('offline');
  });
  await recordQwen35GateFailure(failures, 'adapter', 'M4 WebGPU', () => {
    completed.push('adapter');
  });

  assert.deepEqual(completed, ['parity', 'diagnostic update', 'offline', 'adapter']);
  assert.deepEqual(failures, [
    {stage: 'online', label: 'source parity', message: 'vision confidence differs'},
    {stage: 'online', label: 'diagnostic update', message: 'unable to rewrite diagnostic'}
  ]);
});

test('the final report is written with failures before the gate throws', async () => {
  const failures: Qwen35GateFailure[] = [];
  const events: string[] = [];
  await recordQwen35GateFailure(failures, 'online', 'source parity', () => {
    throw new Error('output mismatch');
  });

  await assert.rejects(
    writeQwen35ReportThenThrow(failures, (reportFailures) => {
      events.push('report-written');
      assert.equal(reportFailures, failures);
      assert.equal(reportFailures[0]?.stage, 'online');
    }),
    /Qwen browser evidence gate failed after all checks completed/u
  );

  assert.deepEqual(events, ['report-written']);
});
