import assert from 'node:assert/strict';

export const KEV_CURRENT_PREVIEW_NOTICE_FILES = Object.freeze([
  'LICENSE',
  'KEV06-NOTICE',
  'KEV-NOTICE',
  'TVM-LICENSE',
  'TVM-NOTICE',
  'EMSCRIPTEN-LICENSE',
  'ROLLDOWN-LICENSE',
  'ROLLDOWN-THIRD-PARTY-LICENSE',
  'VITE-LICENSE.md'
]);

export function assertKevQwen3PackagePolicy({noticeFiles, packageFiles, routeNotice, packageReadme}) {
  assert.ok(noticeFiles.includes('KEV06-NOTICE'), 'The Kev 0.6B route notice is required.');
  assert.ok(noticeFiles.includes('KEV-NOTICE'), 'The adapted Kev code notice is required.');
  assert.ok(!noticeFiles.includes('QWEN-NOTICE'), 'The generic Qwen3.5 notice does not apply to this route.');
  assert.ok(packageFiles.includes('KEV06-NOTICE'), 'The package must ship the Kev 0.6B route notice.');
  assert.ok(packageFiles.includes('KEV-NOTICE'), 'The package must ship the adapted Kev code notice.');
  assert.ok(!packageFiles.includes('QWEN-NOTICE'), 'The package must not ship the generic Qwen3.5 notice.');
  assert.match(routeNotice, /Qwen3-0\.6B-Base/);
  assert.match(routeNotice, /da87bfb608c14b7cf20ba1ce41287e8de496c0cd/);
  assert.match(routeNotice, /Copyright 2024 Alibaba Cloud/);
  assert.match(routeNotice, /Apache-2\.0/);
  assert.match(routeNotice, /MODEL-APACHE-2\.0\.txt/);
  assert.doesNotMatch(routeNotice, /Qwen3\.5/);
  assert.match(packageReadme, /`KEV06-NOTICE`/);
  assert.match(packageReadme, /`KEV-NOTICE`/);
  assert.doesNotMatch(packageReadme, /`QWEN-NOTICE`/);
}
