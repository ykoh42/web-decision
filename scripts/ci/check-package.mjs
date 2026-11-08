import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {checkConsumerTypes} from './check-types.mjs';
import {findPackedContentIssues} from './packed-content.mjs';
import {withTemporaryDirectories} from './temporary-directories.mjs';

const stableKevManifestSha256 = '8577804e638051cb056af51fee89eb18ea5de85ccec4f196594d830c16fdc767';

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical(value[key])])
    );
  return value;
}

function localChunkImports(text) {
  const imports = new Set();
  const pattern = /\b(?:from\s*|import\s*(?:\(\s*)?)["']\.\/([^"']+\.js)["']/gu;
  for (const match of text.matchAll(pattern)) imports.add(match[1]);
  return imports;
}

function collectChunkGraph(directory, entryFiles) {
  const pending = [...entryFiles];
  const visited = new Set();
  const chunks = [];
  while (pending.length) {
    const name = pending.pop();
    if (visited.has(name)) continue;
    visited.add(name);
    const text = readFileSync(join(directory, name), 'utf8');
    chunks.push({name, text});
    for (const imported of localChunkImports(text)) {
      if (!visited.has(imported)) pending.push(imported);
    }
  }
  return chunks;
}

function verifyInlineAssets(chunks, assets, description) {
  const uniqueChunks = [...new Map(chunks.map((chunk) => [chunk.name, chunk])).values()];
  const dataHashes = new Set();
  for (const {text} of uniqueChunks) {
    for (const match of text.matchAll(/["'](data:[^"']+)["']/gu)) {
      const dataUrl = match[1];
      const comma = dataUrl.indexOf(',');
      if (comma <= 5) continue;
      const metadata = dataUrl.slice(5, comma);
      const payload = dataUrl.slice(comma + 1);
      let bytes;
      try {
        bytes = metadata.includes(';base64')
          ? Buffer.from(payload, 'base64')
          : Buffer.from(decodeURIComponent(payload));
      } catch {
        continue;
      }
      dataHashes.add(createHash('sha256').update(bytes).digest('hex'));
    }
  }

  for (const [name, sourcePath] of assets) {
    const key = name.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
    const keyPattern = new RegExp(`["']${key}["']\\s*:`, 'u');
    assert.ok(
      uniqueChunks.some(({text}) => keyPattern.test(text)),
      `The ${description} lazy chunk is missing the asset key ${name}.`
    );
    const source = readFileSync(join(pkg, sourcePath));
    const sourceHash = createHash('sha256').update(source).digest('hex');
    assert.ok(dataHashes.has(sourceHash), `The ${description} lazy chunk changed the pinned asset ${name}.`);
  }
}

const root = fileURLToPath(new URL('../../', import.meta.url));
const pkg = join(root, 'packages/web-decision');
withTemporaryDirectories((makeTempDirectory) => {
  const consumer = makeTempDirectory('web-decision-ci-consumer-');
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const metadata = JSON.parse(readFileSync(join(pkg, 'package.json'), 'utf8'));
  const smokeTest = `
  import assert from 'node:assert/strict';
  import * as library from 'web-decision';
  assert.equal(typeof library.loadDecisionModel, 'function');
  assert.equal(typeof library.clearModelCache, 'function');
  assert.equal(typeof library.WebDecisionError, 'function');
  assert.deepEqual(library.listStableModelIds(), [
    'tamev-micro-minilm-int8',
    'tamev-nano-tinybert-int8',
    'tamev-nano-flash-tinybert-int8',
    'kev-0.6b-q4f16'
  ]);
  const stableKev = library.describeStableModel('kev-0.6b-q4f16');
  assert.equal(stableKev.assets.integrity, 'pinned-manifest');
  assert.equal(stableKev.assets.manifestSha256, '${stableKevManifestSha256}');
  assert.equal(stableKev.assets.location, 'package-and-pinned-upstream');
  assert.equal(library.isStableModelId('tamev-micro-minilm-int8'), true);
  assert.equal(library.isStableModelId('tamev-nano-flash-tinybert-int8'), true);
  assert.equal(library.isStableModelId('experimental-tamev-micro-minilm-int8'), false);
  assert.equal(library.describeStableModel('tamev-micro-minilm-int8').model, 'tamev-micro-minilm-int8');
  assert.throws(() => library.describeStableModel('experimental-tamev-micro-minilm-int8'), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('mapika-0.8b-q4f16').features.textChoice.limits[0].max, 128);
  assert.equal(library.isDecisionModel('imajev-4b'), false);
  assert.ok(library.listDecisionModels().includes('experimental-elephant-9b-q4f16-dlight'));
  assert.ok(library.listDecisionModels().includes('experimental-clef-9b-q4f16-selected'));
  assert.ok(library.listDecisionModels().includes('experimental-opendecider-nano-fp16'));
  assert.ok(library.listDecisionModels().includes('experimental-crivo-32m-v2-fp32'));
  assert.ok(library.listDecisionModels().includes('experimental-crivo-32m-v2-int8'));
  assert.equal(library.describeModel('experimental-crivo-32m-v2-fp32').backend.name, 'onnxruntime-web-webgpu');
  assert.equal(library.describeModel('experimental-crivo-32m-v2-int8').backend.name, 'onnxruntime-web-wasm');
  assert.equal(library.describeModel('experimental-crivo-32m-v2-int8').precision, 'int8');
  assert.equal(library.describeModel('experimental-crivo-32m-v2-fp32').verification.currentBundleGpu, 'not-revalidated');
  assert.equal(library.describeModel('experimental-crivo-32m-v2-int8').verification.currentBundleCpuWasm, 'verified');
  await assert.rejects(
    library.loadDecisionModel({model: 'experimental-crivo-32m-v2-int8', assetsUrl: 'file:///model/'}),
    {code: 'INVALID_INPUT'}
  );
  assert.ok(library.listDecisionModels().includes('experimental-decima-small-int8'));
  assert.ok(library.listDecisionModels().includes('experimental-tamev-micro-minilm-int8'));
  assert.ok(library.listDecisionModels().includes('experimental-tamev-nano-tinybert-int8'));
  assert.ok(library.listDecisionModels().includes('experimental-tamev-nano-flash-tinybert-int8'));
  assert.ok(library.listDecisionModels().includes('experimental-skejja-edge32k-ft6-int8'));
  assert.ok(library.listDecisionModels().includes('experimental-skejja-edge32k-ft6-int8dyn'));
  assert.ok(library.listDecisionModels().includes('experimental-ettin-68m-int8'));
  assert.ok(library.listDecisionModels().includes('experimental-laya-multilingual-fp16'));
  assert.ok(library.listDecisionModels().includes('experimental-gliner2.5-decide-mobile-q4f16'));
  assert.equal(library.describeModel('experimental-gliner2.5-decide-mobile-q4f16').precision, 'q4f16');
  assert.equal(library.describeModel('experimental-gliner2.5-decide-mobile-q4f16').backend.name, 'onnxruntime-web-webgpu');
  await assert.rejects(
    library.loadDecisionModel({model: 'experimental-gliner2.5-decide-mobile-q4f16', assetsUrl: 'file:///model/'}),
    {code: 'INVALID_INPUT'}
  );
  assert.equal(library.describeModel('experimental-laya-multilingual-fp16').backend.name, 'onnxruntime-web-webgpu');
  assert.equal(
    library.describeModel('experimental-laya-multilingual-fp16').checkpoint.graphSha256,
    '81f641e21289ddb2cd8ac774c1cddebfdb0fa7e4b183e49ad476d2faee46d98a'
  );
  await assert.rejects(
    library.loadDecisionModel({model: 'experimental-laya-multilingual-fp16', assetsUrl: 'file:///model/'}),
    {code: 'INVALID_INPUT'}
  );
  assert.ok(library.listDecisionModels().includes('experimental-d1-omni-600m-q8'));
  assert.equal(library.describeModel('experimental-d1-omni-600m-q8').precision, 'q8');
  assert.equal(library.describeModel('experimental-d1-omni-600m-q8').backend.name, 'onnxruntime-web-webgpu');
  assert.equal(library.describeModel('experimental-d1-omni-600m-q8').verification.currentBundleGpu, 'not-revalidated');
  assert.equal(
    library.describeModel('experimental-d1-omni-600m-q8').features.textChoice.limits.find((limit) => limit.resource === 'tokens').max,
    4096
  );
  await assert.rejects(
    library.loadDecisionModel({model: 'experimental-d1-omni-600m-q8', assetsUrl: 'file:///model/'}),
    {code: 'INVALID_INPUT'}
  );
  assert.ok(library.listDecisionModels().includes('experimental-openjev-verdict-151m-fp16'));
  assert.equal(library.describeModel('experimental-openjev-verdict-151m-fp16').precision, 'f16');
  assert.equal(library.describeModel('experimental-openjev-verdict-151m-fp16').verification.currentBundleGpu, 'not-revalidated');
  assert.equal(library.describeModel('experimental-openjev-verdict-151m-fp16').features.textChoice.abstention, 'implemented');
  await assert.rejects(
    library.loadDecisionModel({model: 'experimental-openjev-verdict-151m-fp16', assetsUrl: 'file:///model/'}),
    {code: 'INVALID_INPUT'}
  );
  assert.ok(library.listDecisionModels().includes('experimental-tinydecide-10m-q4'));
  assert.equal(library.describeModel('experimental-tinydecide-10m-q4').precision, 'q4');
  assert.equal(library.describeModel('experimental-tinydecide-10m-q4').backend.name, 'javascript');
  assert.equal(library.describeModel('experimental-decima-small-int8').precision, 'int8');
  assert.equal(library.describeModel('experimental-decima-small-int8').backend.name, 'onnxruntime-web-wasm');
  assert.equal(library.describeModel('experimental-tamev-micro-minilm-int8').backend.name, 'onnxruntime-web-wasm');
  await assert.rejects(library.loadDecisionModel({model: 'experimental-tamev-micro-minilm-int8', assetsUrl: 'file:///model/'}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-tamev-nano-tinybert-int8').backend.name, 'onnxruntime-web-wasm');
  await assert.rejects(library.loadDecisionModel({model: 'experimental-tamev-nano-tinybert-int8', assetsUrl: 'file:///model/'}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-tamev-nano-flash-tinybert-int8').backend.name, 'onnxruntime-web-wasm');
  await assert.rejects(library.loadDecisionModel({model: 'experimental-tamev-nano-flash-tinybert-int8', assetsUrl: 'file:///model/'}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-skejja-edge32k-ft6-int8').backend.name, 'onnxruntime-web-wasm');
  await assert.rejects(library.loadDecisionModel({model: 'experimental-skejja-edge32k-ft6-int8', assetsUrl: 'file:///model/'}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-skejja-edge32k-ft6-int8dyn').backend.name, 'onnxruntime-web-wasm');
  await assert.rejects(library.loadDecisionModel({model: 'experimental-skejja-edge32k-ft6-int8dyn', assetsUrl: 'file:///model/'}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-ettin-68m-int8').backend.name, 'onnxruntime-web-wasm');
  assert.equal(library.describeModel('experimental-ettin-68m-int8').features.textChoice.limits[0].max, 2048);
  await assert.rejects(library.loadDecisionModel({model: 'experimental-ettin-68m-int8', assetsUrl: 'file:///model/'}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-opendecider-nano-fp16').checkpoint.graphSha256, '3c92c5e4dd01caae3afe4d3d9ffce7295343fe6fecceb39b030d895dd8fa4488');
  assert.equal(library.describeModel('experimental-clef-9b-q4f16-selected').features.textChoice.limits.find((limit) => limit.resource === 'tokens').max, 512);
  await assert.rejects(library.loadDecisionModel({model: 'experimental-clef-9b-q4f16-selected', assetsUrl: 'https://example.invalid/', cache: true}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-mapika-4b-q4f16').features.textChoice.handling, 'packed-request');
  assert.equal(library.describeModel('experimental-mapika-4b-q4f16-dlight').checkpoint.graphSha256, '0fe97fa7314e66c15540ee870e0bcc30fe852ee8c0a231a94a7cd1eaad1b8375');
  assert.equal(library.describeModel('experimental-pplx-decider-flash-v1.2-q4f16-dlight').features.textChoice.handling, 'packed-request');
  assert.equal(library.describeModel('experimental-imajev-4b-candidate').features.textChoice.abstention, 'implemented');
  assert.equal(library.describeModel('experimental-nari-decision-4b-mixed16-q4f16').precision, 'mixed-f16-q4f16');
  assert.equal(library.describeModel('experimental-nari-decision-4b-q4f16').features.textChoice.limits.find((limit) => limit.resource === 'options' && limit.scope === 'request').max, 64);
  await assert.rejects(library.loadDecisionModel({model: 'experimental-nari-decision-4b-q4f16', assetsUrl: 'https://example.invalid/', cache: true}), {code: 'INVALID_INPUT'});
  assert.equal(library.describeModel('experimental-h2o-lightning-4b-q4f16').features.textChoice.score.temperature, 0.75);
  assert.equal(library.describeModel('experimental-h2o-lightning-4b-q4f16-dlight').checkpoint.graphSha256, '32da4bd915dec178f29828acd2a57a202df069f4c6c0f13b0d9e7c55ff782336');
  assert.equal(library.describeModel('experimental-h2o-lightning-4b-q4f16-dual').checkpoint.graphSha256, 'd4531ab4a4690b192ba205c42a4e9ca0117d09e9454ff90ee9958306f35ce9b8');
  const internal = await import(new URL('../src/h2o/source-entry.mjs', import.meta.resolve('web-decision')));
  assert.equal(typeof internal.openH2oAssetSource, 'function');
  await assert.rejects(library.loadDecisionModel({model: 'experimental-h2o-lightning-4b-q4f16', assetsUrl: 'https://example.invalid/', cache: true}), {code: 'INVALID_INPUT'});
  await assert.rejects(library.loadDecisionModel({model: 'experimental-h2o-lightning-4b-q4f16', assetsUrl: 'https://example.invalid/', onProgress() {}}), {code: 'INVALID_INPUT'});
  await assert.rejects(library.loadDecisionModel({model: 'experimental-h2o-lightning-4b-q4f16-dlight', assetsUrl: 'https://example.invalid/', cache: true}), {code: 'INVALID_INPUT'});
  await assert.rejects(library.loadDecisionModel({model: 'experimental-opendecider-nano-fp16', assetsUrl: 'https://example.invalid/', cache: true}), {code: 'INVALID_INPUT'});
  await assert.rejects(import('web-decision/h2o-source'), {code: 'ERR_PACKAGE_PATH_NOT_EXPORTED'});
`;
  assert.notEqual(metadata.private, true);
  assert.equal(metadata.license, 'Apache-2.0');

  const output = execFileSync(npm, ['pack', '--json', '--pack-destination', consumer], {
    cwd: pkg,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit']
  });
  const [archive] = JSON.parse(output);
  const files = archive.files.map((file) => file.path).sort();
  const distDirectory = join(pkg, 'dist');
  const distFiles = readdirSync(distDirectory).sort();
  const bundle = readFileSync(join(distDirectory, 'index.js'));
  const bundleText = bundle.toString('utf8');
  const dynamicKevChunkFiles = [...bundleText.matchAll(/\bimport\s*\(\s*["']\.\/([^"']+\.js)["']\s*\)/gu)]
    .map((match) => match[1])
    .filter((name) => name.includes('bundled-assets'));
  const kevChunkFiles = [...new Set(dynamicKevChunkFiles)].sort();
  assert.ok(kevChunkFiles.length > 0, 'The build must emit lazy Kev asset chunks.');
  assert.ok(
    kevChunkFiles.every((name) => distFiles.includes(name)),
    'Every lazy Kev asset chunk must be in dist.'
  );
  const kevChunkGroups = new Map(kevChunkFiles.map((name) => [name, collectChunkGraph(distDirectory, [name])]));
  const experimentalKevChunks = [...kevChunkGroups.values()]
    .filter((chunks) => chunks.some(({text}) => text.includes('experimentalKevBundledUrls')))
    .flat();
  const stableKevChunks = [...kevChunkGroups.values()]
    .filter((chunks) => chunks.some(({text}) => text.includes('stableKevBundledUrls')))
    .flat();
  assert.ok(experimentalKevChunks.length, 'The candidate Kev lazy chunk must remain packaged.');
  assert.ok(stableKevChunks.length, 'The stable Kev lazy chunk must remain packaged.');

  const stableManifest = JSON.parse(readFileSync(join(pkg, 'assets/kev06-stable/manifest.json'), 'utf8'));
  const stableManifestHash = createHash('sha256')
    .update(JSON.stringify(canonical(stableManifest)) + '\n')
    .digest('hex');
  assert.equal(stableManifestHash, stableKevManifestSha256, 'The bundled stable manifest must match its released pin.');
  assert.ok(bundleText.includes(stableManifestHash), 'The package entry must retain the stable manifest pin.');
  for (const [name, metadata] of Object.entries(stableManifest.files)) {
    assert.ok(bundleText.includes(metadata.sha256), `The package entry must embed the stable pin for ${name}.`);
  }

  const experimentalAssetInputs = [
    ['model.wasm', 'assets/experimental-kev-upstream/model.wasm'],
    ['tokenizer.json', 'assets/experimental-kev-upstream/tokenizer.json'],
    ['tokenizer_config.json', 'assets/experimental-kev-upstream/tokenizer_config.json'],
    ['model_config.json', 'assets/experimental-kev-upstream/model_config.json'],
    ['weights/tensor-cache.json', 'assets/experimental-kev-upstream/weights/tensor-cache.json']
  ];
  const stableAssetInputs = [
    ['model.wasm', 'assets/kev06-stable/model.wasm'],
    ['tokenizer.json', 'assets/experimental-kev-upstream/tokenizer.json'],
    ['tokenizer_config.json', 'assets/experimental-kev-upstream/tokenizer_config.json'],
    ['model_config.json', 'assets/experimental-kev-upstream/model_config.json'],
    ['weights/tensor-cache.json', 'assets/experimental-kev-upstream/weights/tensor-cache.json']
  ];
  verifyInlineAssets(experimentalKevChunks, experimentalAssetInputs, 'experimental Kev');
  for (const [name, sourcePath] of stableAssetInputs) {
    const source = readFileSync(join(pkg, sourcePath));
    const metadata = stableManifest.files[name];
    assert.equal(source.byteLength, metadata.bytes, `The stable source asset size differs from its manifest: ${name}.`);
    assert.equal(
      createHash('sha256').update(source).digest('hex'),
      metadata.sha256,
      `The stable source asset differs from its manifest: ${name}.`
    );
  }
  verifyInlineAssets(stableKevChunks, stableAssetInputs, 'stable Kev');

  const expectedFiles = [
    'CLEF-NOTICE',
    'CRIVO-NOTICE',
    'CRIVO_SUPPORT.md',
    'D1-OMNI-NOTICE',
    'DEBERTA-NOTICE',
    'DECIMA-AGENT-NOTICE',
    'DECIMA-BASE-CLEAN-NOTICE',
    'DECIMA-NOTICE',
    'EMSCRIPTEN-LICENSE',
    'H2O-NOTICE',
    'H2O-SHA256-LICENSE',
    'IMAJEV-NOTICE',
    'ELEPHANT-NOTICE',
    'GLINER-MOBILE-NOTICE',
    'LICENSE',
    'KEV06-NOTICE',
    'KEV-NOTICE',
    'KEV-REFLEXES-NOTICE',
    'LAYA-NOTICE',
    'LAYA-TYPED-NOTICE',
    'MAGICK-WASM-LICENSE',
    'MAGICK-WASM-NOTICE',
    'MAPIKA-NOTICE',
    'NARI-NOTICE',
    'OPENDECIDER-NANO-LICENSE',
    'OPENDECIDER-NANO-NOTICE',
    'QWEN-NOTICE',
    'PPLX-NOTICE',
    'STRANDS-NOTICE',
    'TAMEV-NOTICE',
    'TINYDECIDE-NOTICE',
    'ETTIN-NOTICE',
    'README.md',
    'ROLLDOWN-LICENSE',
    'ROLLDOWN-THIRD-PARTY-LICENSE',
    'SAFEDECIDE-NOTICE',
    'SKEJJA-LICENSE',
    'SKEJJA-NOTICE',
    'TVM-LICENSE',
    'TVM-NOTICE',
    'VERDICT-NOTICE',
    'VITE-LICENSE.md',
    'XDECISION-NOTICE',
    'assets/experimental-kev-upstream/PROVENANCE.md',
    'assets/experimental-kev-upstream/notices/COMPILER-RT-HEADER-NOTICE.txt',
    'assets/experimental-kev-upstream/notices/COMPILER-RT-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/DLPACK-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/EMSCRIPTEN-COMPILER-RT-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/EMSCRIPTEN-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/LIBCXX-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/LIBCXXABI-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/MODEL-APACHE-2.0.txt',
    'assets/experimental-kev-upstream/notices/MUSL-COPYRIGHT.txt',
    'assets/experimental-kev-upstream/notices/NOTICE-SOURCES.md',
    'assets/experimental-kev-upstream/notices/PROVENANCE.md',
    'assets/experimental-kev-upstream/notices/TVM-FFI-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/TVM-FFI-NOTICE.txt',
    'assets/experimental-kev-upstream/notices/TVM-LICENSE.txt',
    'assets/experimental-kev-upstream/notices/TVM-NOTICE.txt',
    'assets/kev06-stable/PROVENANCE.md',
    'assets/kev06-stable/notices/MODEL-APACHE-2.0.txt',
    ...distFiles.map((name) => `dist/${name}`),
    'package.json',
    'src/STRANDS-GEMMA4-NOTICE',
    'src/abort.js',
    'src/assets.js',
    'src/bounded-response.js',
    'src/crivo/contract.mjs',
    'src/crivo/loader.mjs',
    'src/crivo/protocol.mjs',
    'src/crivo/runner.mjs',
    'src/crivo/session.mjs',
    'src/crivo/source.mjs',
    'src/crivo/tokenizer.mjs',
    'src/d1-omni/contract.mjs',
    'src/d1-omni/loader.mjs',
    'src/d1-omni/protocol.mjs',
    'src/d1-omni/runner.mjs',
    'src/d1-omni/session.mjs',
    'src/d1-omni/source.mjs',
    'src/d1-omni/tokenizer.mjs',
    'src/gpu-session-utils.js',
    'src/h2o/README.md',
    'src/h2o/binding.mjs',
    'src/h2o/hasher.mjs',
    'src/h2o/loader.mjs',
    'src/h2o/owner.mjs',
    'src/h2o/plan.mjs',
    'src/h2o/protocol.mjs',
    'src/h2o/runner.mjs',
    'src/h2o/session.mjs',
    'src/h2o/sha256.mjs',
    'src/h2o/signal.mjs',
    'src/h2o/source-entry.mjs',
    'src/h2o/source-evidence.mjs',
    'src/h2o/source.mjs',
    'src/h2o/strict-json.mjs',
    'src/h2o/teardown.mjs',
    'src/h2o/tokenizer-adapter.mjs',
    'src/h2o/tokenizer.mjs',
    'src/h2o/upload.mjs',
    'src/kev.js',
    'src/laya-typed/contract.mjs',
    'src/laya-typed/loader.mjs',
    'src/laya/contract.mjs',
    'src/laya/loader.mjs',
    'src/laya/protocol.mjs',
    'src/laya/runner.mjs',
    'src/laya/session.mjs',
    'src/laya/source.mjs',
    'src/laya/tokenizer.mjs',
    'src/load-device-loss.js',
    'src/opendecider-nano/contract.mjs',
    'src/opendecider-nano/gpu-compat.mjs',
    'src/opendecider-nano/loader.mjs',
    'src/opendecider-nano/protocol.mjs',
    'src/opendecider-nano/runner.mjs',
    'src/opendecider-nano/session.mjs',
    'src/opendecider-nano/source.mjs',
    'src/opendecider-nano/tokenizer.mjs',
    'src/opendecider-nano/typed.mjs',
    'src/decima/contract.mjs',
    'src/decima/loader.mjs',
    'src/decima/normalize.mjs',
    'src/decima/protocol.mjs',
    'src/decima/runner.mjs',
    'src/decima/session.mjs',
    'src/decima/source.mjs',
    'src/decima/tokenizer.mjs',
    'src/decima/typed.mjs',
    'src/ettin/contract.mjs',
    'src/ettin/loader.mjs',
    'src/ettin/protocol.mjs',
    'src/ettin/runner.mjs',
    'src/ettin/session.mjs',
    'src/ettin/source.mjs',
    'src/ettin/tokenizer.mjs',
    'src/gliner-mobile/contract.mjs',
    'src/gliner-mobile/loader.mjs',
    'src/gliner-mobile/protocol.mjs',
    'src/gliner-mobile/runner.mjs',
    'src/gliner-mobile/session.mjs',
    'src/gliner-mobile/source.mjs',
    'src/gliner-mobile/tokenizer.mjs',
    'src/request-fields.mjs',
    'src/runtime-lifecycle.js',
    'src/safedecide/contract.mjs',
    'src/safedecide/loader.mjs',
    'src/safedecide/protocol.mjs',
    'src/safedecide/runner.mjs',
    'src/safedecide/session.mjs',
    'src/safedecide/source.mjs',
    'src/safedecide/tokenizer.mjs',
    'src/skejja/calibration.mjs',
    'src/skejja/contract.mjs',
    'src/skejja/loader.mjs',
    'src/skejja/protocol.mjs',
    'src/skejja/runner.mjs',
    'src/skejja/session.mjs',
    'src/skejja/source.mjs',
    'src/skejja/tokenizer.mjs',
    'src/strands/disposal-guard.mjs',
    'src/strands/gemma4-e2b-contract.mjs',
    'src/strands/gemma4-e2b-loader.mjs',
    'src/strands/gemma4-e2b-runner.mjs',
    'src/strands/gemma4-e2b-session.mjs',
    'src/strands/gemma4-manifest.mjs',
    'src/strands/gemma4-progress.mjs',
    'src/strands/gemma4-protocol.mjs',
    'src/strands/gemma4-upload.mjs',
    'src/strands/range-reader.mjs',
    'src/strands/scope.mjs',
    'src/tinydecide/contract.mjs',
    'src/tinydecide/corrections.js',
    'src/tinydecide/engine.js',
    'src/tinydecide/inputs.mjs',
    'src/tinydecide/loader.mjs',
    'src/tinydecide/session.mjs',
    'src/tinydecide/source.mjs',
    'src/verdict/contract.mjs',
    'src/verdict/loader.mjs',
    'src/verdict/protocol.mjs',
    'src/verdict/runner.mjs',
    'src/verdict/session.mjs',
    'src/verdict/source.mjs',
    'src/verdict/tokenizer.mjs',
    'src/tamev/contract.mjs',
    'src/tamev/loader.mjs',
    'src/tamev/protocol.mjs',
    'src/tamev/runner.mjs',
    'src/tamev/session.mjs',
    'src/tamev/source.mjs',
    'src/tamev/tokenizer.mjs',
    'src/tamev/typed.mjs',
    'src/index.d.ts'
  ].sort();
  assert.deepEqual(files, expectedFiles);
  const packagePath = resolve(consumer, archive.filename);
  writeFileSync(join(consumer, 'package.json'), '{"private":true,"type":"module"}\n');
  execFileSync(npm, ['install', '--ignore-scripts', '--no-audit', '--no-fund', packagePath], {
    cwd: consumer,
    stdio: 'inherit'
  });
  execFileSync(npm, ['audit', '--omit=dev'], {cwd: consumer, stdio: 'inherit'});
  const installed = JSON.parse(readFileSync(join(consumer, 'node_modules/web-decision/package.json')));
  assert.equal(installed.version, metadata.version);
  assert.equal(installed.license, metadata.license);
  for (const file of files) {
    const issues = findPackedContentIssues(readFileSync(join(consumer, 'node_modules/web-decision', file)));
    assert.deepEqual(issues, [], `Potential private content in packed file: ${file}`);
  }
  for (const notice of [
    'CLEF-NOTICE',
    'DECIMA-AGENT-NOTICE',
    'DECIMA-BASE-CLEAN-NOTICE',
    'DECIMA-NOTICE',
    'EMSCRIPTEN-LICENSE',
    'H2O-NOTICE',
    'H2O-SHA256-LICENSE',
    'IMAJEV-NOTICE',
    'ELEPHANT-NOTICE',
    'GLINER-MOBILE-NOTICE',
    'KEV06-NOTICE',
    'KEV-NOTICE',
    'KEV-REFLEXES-NOTICE',
    'DEBERTA-NOTICE',
    'XDECISION-NOTICE',
    'LAYA-NOTICE',
    'LAYA-TYPED-NOTICE',
    'MAPIKA-NOTICE',
    'NARI-NOTICE',
    'OPENDECIDER-NANO-LICENSE',
    'OPENDECIDER-NANO-NOTICE',
    'QWEN-NOTICE',
    'PPLX-NOTICE',
    'SKEJJA-LICENSE',
    'SKEJJA-NOTICE',
    'STRANDS-NOTICE',
    'src/STRANDS-GEMMA4-NOTICE',
    'TAMEV-NOTICE',
    'TINYDECIDE-NOTICE',
    'ETTIN-NOTICE',
    'VERDICT-NOTICE',
    'ROLLDOWN-LICENSE',
    'ROLLDOWN-THIRD-PARTY-LICENSE',
    'VITE-LICENSE.md'
  ]) {
    assert.deepEqual(
      readFileSync(join(consumer, 'node_modules/web-decision', notice)),
      readFileSync(join(pkg, notice))
    );
  }
  execFileSync(process.execPath, ['--input-type=module', '-e', smokeTest], {cwd: consumer, stdio: 'inherit'});
  const lazyChunkSmokeTest = `
    import assert from 'node:assert/strict';
    import {createHash} from 'node:crypto';
    const entry = import.meta.resolve('web-decision');
    const modules = await Promise.all(
      ${JSON.stringify(kevChunkFiles)}.map((name) => import(new URL('./' + name, entry)))
    );
    const bundles = modules.flatMap((module) => Object.values(module));
    const stableExports = modules.flatMap((module) => Object.entries(module));
    assert.ok(stableExports.some(([name]) => name === 'stableKevBundledUrls'));
    assert.ok(stableExports.some(([name]) => name === 'experimentalKevBundledUrls'));
    const assetMaps = bundles.filter((value) => value && typeof value === 'object' && value['model.wasm']);
    const hashDataUrl = (dataUrl) => {
      const comma = dataUrl.indexOf(',');
      const metadata = dataUrl.slice(5, comma);
      const payload = dataUrl.slice(comma + 1);
      const bytes = metadata.includes(';base64') ? Buffer.from(payload, 'base64') : Buffer.from(decodeURIComponent(payload));
      return createHash('sha256').update(bytes).digest('hex');
    };
    const wasmHashes = assetMaps.map((assets) => hashDataUrl(assets['model.wasm']));
    assert.ok(wasmHashes.includes('68fe0331180dc1adfb3d1e4295c06133815b3f5bfc5ad8dbd73ae66f3a02b8fb'));
    assert.ok(wasmHashes.includes('78026118d5c1f85941f75e0929d88d72edfc1a0418895d3f26bf6738a506882d'));
  `;
  execFileSync(process.execPath, ['--input-type=module', '-e', lazyChunkSmokeTest], {cwd: consumer, stdio: 'inherit'});
  checkConsumerTypes(consumer);
  const installedDist = join(consumer, 'node_modules/web-decision/dist');
  const installedBundle = readFileSync(join(installedDist, 'index.js'));
  assert.deepEqual(installedBundle, bundle);
  for (const name of kevChunkFiles) {
    const kevChunkText = readFileSync(join(installedDist, name), 'utf8');
    assert.doesNotMatch(
      kevChunkText,
      /\?url(?:&|["'])/u,
      'The published lazy chunk must not depend on consumer asset plugins.'
    );
  }
  for (const name of distFiles.filter((file) => file !== 'index.js')) {
    assert.deepEqual(
      readFileSync(join(installedDist, name)),
      readFileSync(join(distDirectory, name)),
      `The tarball must preserve dist/${name}.`
    );
  }
  console.log(
    `Installed ${metadata.name}@${metadata.version}; bundle SHA-256: ${createHash('sha256').update(bundle).digest('hex')}; lazy Kev chunks: ${kevChunkFiles.join(', ')}`
  );
  const pnpmConsumer = makeTempDirectory('web-decision-ci-pnpm-consumer-');
  const pnpm = process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm';
  const packageManager = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).packageManager;
  writeFileSync(
    join(pnpmConsumer, 'package.json'),
    JSON.stringify({private: true, type: 'module', packageManager}) + '\n'
  );
  execFileSync(pnpm, ['add', '--ignore-scripts', packagePath], {cwd: pnpmConsumer, stdio: 'inherit'});
  const pnpmInstalled = JSON.parse(readFileSync(join(pnpmConsumer, 'node_modules/web-decision/package.json')));
  assert.equal(pnpmInstalled.version, metadata.version);
  assert.equal(pnpmInstalled.license, metadata.license);
  assert.deepEqual(readFileSync(join(pnpmConsumer, 'node_modules/web-decision/dist/index.js')), bundle);
  for (const name of distFiles.filter((file) => file !== 'index.js')) {
    assert.deepEqual(
      readFileSync(join(pnpmConsumer, 'node_modules/web-decision/dist', name)),
      readFileSync(join(distDirectory, name)),
      `The pnpm consumer must receive dist/${name}.`
    );
  }
  execFileSync(process.execPath, ['--input-type=module', '-e', smokeTest], {cwd: pnpmConsumer, stdio: 'inherit'});
  execFileSync(process.execPath, ['--input-type=module', '-e', lazyChunkSmokeTest], {
    cwd: pnpmConsumer,
    stdio: 'inherit'
  });
  checkConsumerTypes(pnpmConsumer);
  console.log('The archive installs, imports, and type-checks in clean npm and pnpm consumers.');
  console.log('This check imports the package in Node; it does not run browser inference.');
});
