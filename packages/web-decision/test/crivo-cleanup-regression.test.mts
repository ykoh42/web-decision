import assert from 'node:assert/strict';
import test from 'node:test';
import {createCrivoRunner} from '../src/crivo/runner.mjs';

class Tensor {
  constructor(
    readonly type: string,
    readonly data: ArrayBufferView,
    readonly dims: number[]
  ) {}
  dispose(): void {}
}

test('Crivo preserves cleanup failure when a runtime output tensor is null', async () => {
  const runner = createCrivoRunner(
    {Tensor},
    {
      async run() {
        return {logits: new Tensor('float32', new Float32Array([1, 2]), [1, 2]), unexpected: null};
      },
      async release() {}
    }
  );
  await assert.rejects(runner.run([{inputIds: [1], optionPositions: [0, 0], qtype: 0}]), TypeError);
});
