import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';

test('package includes the pinned Emscripten license text', async () => {
  const metadata = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  assert.ok(metadata.files.includes('EMSCRIPTEN-LICENSE'));
  const license = await readFile(new URL('../EMSCRIPTEN-LICENSE', import.meta.url));
  assert.equal(
    createHash('sha256').update(license).digest('hex'),
    '620a78084fc7ca97c0b5dea9abf891f3ffcadfdbf305276f099c9c4e12fc1d86'
  );
});
