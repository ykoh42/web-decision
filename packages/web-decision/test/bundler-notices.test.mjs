import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';

test('package includes the pinned Vite and Rolldown notice texts', async () => {
  const metadata = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  const notices = {
    'VITE-LICENSE.md': 'a22152baa1aef74402443008e767e7c689a524c0909d19b0431d97f5ab066a13',
    'ROLLDOWN-LICENSE': '23ecfff35a5a2e80d92142f75228912c3b1abc4b5a8337a821ff4397e2f9f734',
    'ROLLDOWN-THIRD-PARTY-LICENSE': 'a877291d800ed43692f3f9ae09d8e01cc6f7293ad39d43896059c188ffbb8b7c'
  };
  for (const [name, expected] of Object.entries(notices)) {
    assert.ok(metadata.files.includes(name), name);
    const text = await readFile(new URL(`../${name}`, import.meta.url));
    assert.equal(createHash('sha256').update(text).digest('hex'), expected, name);
  }
});
