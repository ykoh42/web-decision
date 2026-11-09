import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {convertBf16ToFp16, inspectBf16ToFp16, parseSafetensorsHeader} from '../src/kanana/safetensors.mjs';
import {
  KANANA_SAFETENSORS_SPEC,
  openKananaWeightSource,
  openSafetensorsWeightSource
} from '../src/kanana/weight-source.mjs';
import {hashSafetensorsFile} from '../../../scripts/kanana/hash_safetensors_ranges.mjs';

function safetensorsFixture({header, data}) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const paddedHeaderLength = Math.ceil(json.byteLength / 8) * 8;
  const headerBytes = new Uint8Array(paddedHeaderLength).fill(0x20);
  headerBytes.set(json);
  const prefix = new Uint8Array(8 + paddedHeaderLength);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(paddedHeaderLength), true);
  prefix.set(headerBytes, 8);
  const file = new Uint8Array(prefix.byteLength + data.byteLength);
  file.set(prefix);
  file.set(data, prefix.byteLength);
  return {file, headerRange: prefix, dataOffset: prefix.byteLength};
}

function tinyModel() {
  return safetensorsFixture({
    header: {
      __metadata__: {source: 'small fixture'},
      'weights.bf16': {dtype: 'BF16', shape: [4], data_offsets: [0, 8]},
      'scale.f32': {dtype: 'F32', shape: [], data_offsets: [8, 12]},
      'model.visual.patch_embed.proj.weight': {dtype: 'F16', shape: [2], data_offsets: [12, 16]},
      'model.language_model.layers.0.self_attn.q_norm.weight': {dtype: 'BF16', shape: [2], data_offsets: [16, 20]},
      'empty.f16': {dtype: 'F16', shape: [0], data_offsets: [20, 20]}
    },
    data: Uint8Array.of(
      0x80,
      0x3f,
      0x20,
      0xc0,
      0x00,
      0x3e,
      0x01,
      0x33,
      0x00,
      0x00,
      0x80,
      0x3f,
      0x00,
      0x3c,
      0x00,
      0xc0,
      0x80,
      0x3f,
      0x80,
      0xbf
    )
  });
}

test('parses Safetensors header length, tensor offsets, shapes, dtypes and empty tensors', () => {
  const fixture = tinyModel();
  const parsed = parseSafetensorsHeader(fixture.headerRange, fixture.file.byteLength);

  assert.equal(parsed.headerLength, fixture.headerRange.byteLength - 8);
  assert.equal(parsed.dataOffset, fixture.dataOffset);
  assert.equal(parsed.dataBytes, 20);
  assert.equal(parsed.tensorCount, 5);
  assert.deepEqual(parsed.tensors, [
    {
      name: 'weights.bf16',
      dtype: 'BF16',
      shape: [4],
      offset: fixture.dataOffset,
      length: 8
    },
    {name: 'scale.f32', dtype: 'F32', shape: [], offset: fixture.dataOffset + 8, length: 4},
    {
      name: 'model.visual.patch_embed.proj.weight',
      dtype: 'F16',
      shape: [2],
      offset: fixture.dataOffset + 12,
      length: 4
    },
    {
      name: 'model.language_model.layers.0.self_attn.q_norm.weight',
      dtype: 'BF16',
      shape: [2],
      offset: fixture.dataOffset + 16,
      length: 4
    },
    {name: 'empty.f16', dtype: 'F16', shape: [0], offset: fixture.dataOffset + 20, length: 0}
  ]);
});

test('rejects malformed Safetensors headers, unknown dtypes, inconsistent sizes and invalid ranges', () => {
  const fixture = tinyModel();
  const shortHeader = fixture.headerRange.slice(0, fixture.headerRange.byteLength - 1);
  assert.throws(() => parseSafetensorsHeader(shortHeader, fixture.file.byteLength), {code: 'INVALID_ASSET'});
  assert.throws(() => parseSafetensorsHeader(Uint8Array.of(1, 2, 3), fixture.file.byteLength), {code: 'INVALID_ASSET'});

  const unknownType = safetensorsFixture({
    header: {bad: {dtype: 'F128', shape: [1], data_offsets: [0, 16]}},
    data: new Uint8Array(16)
  });
  assert.throws(() => parseSafetensorsHeader(unknownType.headerRange, unknownType.file.byteLength), {
    code: 'INVALID_ASSET'
  });

  const overlap = safetensorsFixture({
    header: {
      first: {dtype: 'U8', shape: [2], data_offsets: [0, 2]},
      second: {dtype: 'U8', shape: [2], data_offsets: [1, 3]}
    },
    data: Uint8Array.of(1, 2, 3)
  });
  assert.throws(() => parseSafetensorsHeader(overlap.headerRange, overlap.file.byteLength), {
    code: 'INVALID_ASSET'
  });

  const gap = safetensorsFixture({
    header: {only: {dtype: 'U8', shape: [1], data_offsets: [1, 2]}},
    data: Uint8Array.of(0, 1)
  });
  assert.throws(() => parseSafetensorsHeader(gap.headerRange, gap.file.byteLength), {code: 'INVALID_ASSET'});
});

test('rounds BF16 values to finite FP16 using IEEE nearest-even semantics', () => {
  const bf16 = Uint8Array.of(0x80, 0x3f, 0x20, 0xc0, 0x00, 0x3e, 0x00, 0x00);
  const fp16 = convertBf16ToFp16(bf16);

  assert.deepEqual([...fp16], [0x00, 0x3c, 0x00, 0xc1, 0x00, 0x30, 0x00, 0x00]);
  assert.deepEqual(
    [...convertBf16ToFp16(Uint8Array.of(0x01, 0x00, 0x00, 0x33, 0x01, 0x33, 0x00, 0xb3, 0x80, 0x33))],
    [0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x80, 0x01, 0x00]
  );
  assert.deepEqual([...convertBf16ToFp16(Uint8Array.of(0x81, 0x36, 0x83, 0x36))], [0x40, 0x00, 0x42, 0x00]);
  assert.throws(() => convertBf16ToFp16(Uint8Array.of(0x80, 0x47)), {code: 'INVALID_ASSET'});
  assert.throws(() => convertBf16ToFp16(Uint8Array.of(0x80, 0x7f)), {code: 'INVALID_ASSET'});
  assert.throws(() => convertBf16ToFp16(Uint8Array.of(0xc1, 0x7f)), {code: 'INVALID_ASSET'});
  assert.throws(() => convertBf16ToFp16(Uint8Array.of(0x00)), {code: 'INVALID_ASSET'});
  assert.deepEqual(inspectBf16ToFp16(Uint8Array.of(0x80, 0x47, 0x80, 0x7f, 0xc1, 0x7f, 0x01, 0x33)), {
    invalidValueCount: 3,
    overflowValueCount: 1,
    nonFiniteValueCount: 2,
    firstFailure: {
      byteOffset: 0,
      reason: 'overflow',
      message: 'BF16 value overflows finite FP16. At tensor byte offset 0.'
    }
  });
});

test('reports the first BF16 overflow and absolute byte offset while hashing', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'kanana-safetensors-'));
  const fixture = safetensorsFixture({
    header: {bad: {dtype: 'BF16', shape: [1], data_offsets: [0, 2]}},
    data: Uint8Array.of(0x80, 0x47)
  });
  const filePath = join(directory, 'bad.safetensors');
  await writeFile(filePath, fixture.file);

  try {
    const manifest = await hashSafetensorsFile(filePath, {
      path: 'bad.safetensors',
      chunkBytes: 8,
      verifyBf16ToFp16: true
    });
    assert.deepEqual(manifest.conversionVerification, {
      enabled: true,
      sourceDtype: 'BF16',
      targetDtype: 'float16',
      rounding: 'nearest-even',
      valid: false,
      invalidValueCount: 1,
      overflowValueCount: 1,
      nonFiniteValueCount: 0,
      failures: [
        {
          tensor: 'bad',
          sourceDtype: 'BF16',
          targetDtype: 'float16',
          fileOffset: fixture.dataOffset,
          tensorOffset: 0,
          reason: 'overflow',
          message: 'BF16 value overflows finite FP16. At tensor byte offset 0.',
          invalidValueCount: 1,
          overflowValueCount: 1,
          nonFiniteValueCount: 0
        }
      ]
    });
    assert.deepEqual(manifest.tensors[0].conversion, {
      sourceDtype: 'BF16',
      targetDtype: 'float16',
      verified: true,
      rounding: 'nearest-even',
      valid: false,
      invalidValueCount: 1,
      overflowValueCount: 1,
      nonFiniteValueCount: 0,
      failure: manifest.conversionVerification.failures[0]
    });
    assert.equal(manifest.tensors[0].conversion.failure.fileOffset, fixture.dataOffset);
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});

test('hashes file ranges incrementally and opens verified, cached source tensors', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'kanana-safetensors-'));
  const fixture = tinyModel();
  const filePath = join(directory, 'model.safetensors');
  await writeFile(filePath, fixture.file);

  const sourceManifest = await hashSafetensorsFile(filePath, {
    path: 'model.safetensors',
    chunkBytes: 8,
    verifyBf16ToFp16: true
  });
  const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
  assert.equal(sourceManifest.headerSha256, digest(fixture.headerRange));
  assert.deepEqual(sourceManifest.tensors[0].chunks, [
    {
      offset: fixture.dataOffset,
      length: 8,
      sha256: digest(fixture.file.subarray(fixture.dataOffset, fixture.dataOffset + 8))
    }
  ]);
  assert.equal(
    sourceManifest.tensors[0].sha256,
    digest(fixture.file.subarray(fixture.dataOffset, fixture.dataOffset + 8))
  );
  assert.ok(sourceManifest.tensors.every(({chunks}) => chunks.every(({length}) => length <= 8 * 1024 * 1024)));
  assert.deepEqual(sourceManifest.conversionVerification, {
    enabled: true,
    sourceDtype: 'BF16',
    targetDtype: 'float16',
    rounding: 'nearest-even',
    valid: true,
    invalidValueCount: 0,
    overflowValueCount: 0,
    nonFiniteValueCount: 0,
    failures: []
  });
  assert.deepEqual(sourceManifest.tensors[0].conversion, {
    sourceDtype: 'BF16',
    targetDtype: 'float16',
    verified: true,
    rounding: 'nearest-even',
    valid: true,
    invalidValueCount: 0,
    overflowValueCount: 0,
    nonFiniteValueCount: 0
  });

  const requiredWeights = [
    {name: 'scale.f32', shape: [], dtype: 'float32', length: 4},
    {name: 'weights.bf16', shape: [4], dtype: 'float16', length: 8}
  ];

  const cacheEntries = new Map();
  const cache = {
    async match(key) {
      return cacheEntries.get(key)?.clone();
    },
    async put(key, response) {
      cacheEntries.set(key, response.clone());
    }
  };
  const requests = [];
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const range = init.headers.Range;
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    assert.ok(match, 'source should issue a byte Range request');
    const start = Number(match[1]);
    const end = Number(match[2]);
    requests.push({url: String(url), range});
    const bytes = fixture.file.subarray(start, end + 1);
    return new Response(bytes, {
      status: 206,
      headers: {
        'Content-Range': `bytes ${start}-${end}/${fixture.file.byteLength}`,
        'Content-Length': String(bytes.byteLength)
      }
    });
  };

  try {
    const source = await openSafetensorsWeightSource({
      base: new URL('https://fixture.invalid/checkpoint/'),
      manifest: sourceManifest,
      requiredWeights,
      cache
    });
    const converted = await source.readTensor('weights.bf16', {dtype: 'float16'});
    const float = await source.readTensor('scale.f32');
    assert.deepEqual(
      source.records.map(({name}) => name),
      ['scale.f32', 'weights.bf16']
    );
    assert.equal(source.tensors, undefined);
    assert.equal(converted.record.dtype, 'float16');
    assert.deepEqual([...converted.bytes], [0x00, 0x3c, 0x00, 0xc1, 0x00, 0x30, 0x01, 0x00]);
    assert.equal(float.record.dtype, 'float32');
    assert.deepEqual([...float.bytes], [0x00, 0x00, 0x80, 0x3f]);
    await assert.rejects(source.readTensor('model.visual.patch_embed.proj.weight'), {code: 'INVALID_ASSET'});
    await assert.rejects(source.readTensor('model.language_model.layers.0.self_attn.q_norm.weight'), {
      code: 'INVALID_ASSET'
    });
    await assert.rejects(source.readTensor('empty.f16'), {code: 'INVALID_ASSET'});
    assert.equal(await source.commit(), true);
    const markerPaths = [...cacheEntries.keys()].map((key) => new URL(key).pathname);
    assert.ok(markerPaths.includes('/checkpoint/kanana-offline-complete-v1.json'));
    assert.equal(markerPaths.includes('/checkpoint/strands-offline-complete-v1.json'), false);
    assert.equal(requests.length, 3);
    assert.equal(requests[0].range, `bytes=0-${fixture.dataOffset - 1}`);

    const requestCount = requests.length;
    const offline = await openSafetensorsWeightSource({
      base: new URL('https://fixture.invalid/checkpoint/'),
      manifest: sourceManifest,
      requiredWeights,
      cache,
      offline: true
    });
    await offline.readTensor('weights.bf16', {dtype: 'float16'});
    await offline.readTensor('scale.f32');
    assert.equal(requests.length, requestCount);
  } finally {
    globalThis.fetch = previousFetch;
    await rm(directory, {recursive: true, force: true});
  }
});

test('requires the pinned Kanana source manifest to verify finite FP16 nearest-even conversion', async () => {
  const pinnedManifest = {
    ...KANANA_SAFETENSORS_SPEC,
    tensors: [
      {
        dtype: 'BF16',
        conversion: {
          sourceDtype: 'BF16',
          targetDtype: 'float16',
          verified: true,
          rounding: 'nearest-even',
          valid: false,
          invalidValueCount: 1,
          overflowValueCount: 1,
          nonFiniteValueCount: 0
        }
      }
    ],
    conversionVerification: {
      enabled: true,
      sourceDtype: 'BF16',
      targetDtype: 'float16',
      rounding: 'nearest-even',
      valid: false,
      invalidValueCount: 1,
      overflowValueCount: 1,
      nonFiniteValueCount: 0,
      failures: [{tensor: 'bad', fileOffset: 83_928}]
    }
  };
  await assert.rejects(
    openKananaWeightSource({manifest: pinnedManifest, requiredWeights: []}),
    (error) => error.code === 'INVALID_ASSET' && /finite BF16 to FP16 nearest-even/.test(error.message)
  );

  pinnedManifest.conversionVerification.valid = true;
  pinnedManifest.conversionVerification.invalidValueCount = 0;
  pinnedManifest.conversionVerification.overflowValueCount = 0;
  pinnedManifest.conversionVerification.nonFiniteValueCount = 0;
  pinnedManifest.conversionVerification.failures = [];
  pinnedManifest.tensors[0].conversion.valid = true;
  pinnedManifest.tensors[0].conversion.invalidValueCount = 0;
  pinnedManifest.tensors[0].conversion.overflowValueCount = 0;
  pinnedManifest.tensors[0].conversion.nonFiniteValueCount = 0;
  await assert.rejects(
    openKananaWeightSource({manifest: pinnedManifest, requiredWeights: []}),
    (error) => error.code === 'INVALID_ASSET' && /exactly 290 weights/.test(error.message)
  );
});

test('verifies Content-Range and chunk integrity, and aborts source setup', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'kanana-safetensors-'));
  const fixture = tinyModel();
  const filePath = join(directory, 'model.safetensors');
  await writeFile(filePath, fixture.file);
  const manifest = await hashSafetensorsFile(filePath, {path: 'model.safetensors', chunkBytes: 8});
  const previousFetch = globalThis.fetch;

  try {
    globalThis.fetch = async () =>
      new Response(fixture.headerRange, {
        status: 206,
        headers: {'Content-Range': `bytes 0-0/${fixture.file.byteLength}`}
      });
    await assert.rejects(
      openSafetensorsWeightSource({
        base: new URL('https://fixture.invalid/'),
        manifest,
        requiredWeights: [{name: 'weights.bf16', shape: [4], dtype: 'float16', length: 8}]
      }),
      {code: 'INVALID_ASSET'}
    );

    const badChunkManifest = structuredClone(manifest);
    badChunkManifest.tensors[0].chunks[0].sha256 = '0'.repeat(64);
    globalThis.fetch = async (url, init) => {
      const match = /^bytes=(\d+)-(\d+)$/.exec(init.headers.Range);
      assert.ok(match);
      const start = Number(match[1]);
      const end = Number(match[2]);
      const bytes = fixture.file.subarray(start, end + 1);
      return new Response(bytes, {
        status: 206,
        headers: {
          'Content-Range': `bytes ${start}-${end}/${fixture.file.byteLength}`,
          'Content-Length': String(bytes.byteLength)
        }
      });
    };
    const source = await openSafetensorsWeightSource({
      base: new URL('https://fixture.invalid/'),
      manifest: badChunkManifest,
      requiredWeights: [{name: 'weights.bf16', shape: [4], dtype: 'float16', length: 8}]
    });
    await assert.rejects(source.readTensor('weights.bf16'), {code: 'INVALID_ASSET'});

    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      openSafetensorsWeightSource({
        base: new URL('https://fixture.invalid/'),
        manifest,
        requiredWeights: [{name: 'weights.bf16', shape: [4], dtype: 'float16', length: 8}],
        signal: controller.signal
      }),
      {name: 'AbortError'}
    );
  } finally {
    globalThis.fetch = previousFetch;
    await rm(directory, {recursive: true, force: true});
  }
});
