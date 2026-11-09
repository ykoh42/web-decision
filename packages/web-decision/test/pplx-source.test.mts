import assert from 'node:assert/strict';
import test from 'node:test';
import {indexFlashFiles} from '../src/pplx/source.mjs';

interface InventoryEntry {
  bindingKind: string;
  outputPath?: string | undefined;
  conversion?: string | undefined;
  sourceOffset?: number | undefined;
  outputBytes?: number | undefined;
}

function inventory() {
  const entries: InventoryEntry[] = Array.from({length: 680}, (_, index) => ({
    bindingKind: 'q4-candidate',
    outputPath: `candidate/weights-${Math.min(index, 675)}`,
    conversion: 'none',
    sourceOffset: index * 4,
    outputBytes: 4
  }));
  entries.push({bindingKind: 'derived-rotary'}, {bindingKind: 'derived-rotary'});
  return {state: 'pplx-q4-joint-assets-pending', entries};
}

test('indexes shared files by their largest end offset and converted files from zero', () => {
  const manifest = inventory();
  manifest.entries[0]!.conversion = 'fp32-to-fp16';
  manifest.entries[0]!.sourceOffset = 100;
  const files = indexFlashFiles(manifest);
  assert.equal(files.size, 676);
  assert.equal(files.get('candidate/weights-0'), 4);
  assert.equal(files.get('candidate/weights-675'), 680 * 4);
});

test('rejects incomplete inventories, unsafe paths and invalid byte ranges', () => {
  for (const change of [
    (value: ReturnType<typeof inventory>) => value.entries.pop(),
    (value: ReturnType<typeof inventory>) => (value.entries[0]!.outputPath = 'candidate/../weights'),
    (value: ReturnType<typeof inventory>) => (value.entries[0]!.conversion = 'unknown'),
    (value: ReturnType<typeof inventory>) => (value.entries[0]!.sourceOffset = -1),
    (value: ReturnType<typeof inventory>) => (value.entries[0]!.outputBytes = 0),
    (value: ReturnType<typeof inventory>) => (value.entries[0]!.outputPath = value.entries[1]!.outputPath)
  ]) {
    const manifest = inventory();
    change(manifest);
    assert.throws(() => indexFlashFiles(manifest), {code: 'INVALID_ASSET'});
  }
});
