import assert from 'node:assert/strict';
import {spawn, spawnSync, type SpawnSyncReturns} from 'node:child_process';
import {createHash} from 'node:crypto';
import {
  chmod,
  copyFile,
  lstat,
  link,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  readlink,
  rm,
  symlink,
  writeFile
} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {test} from 'node:test';

const configSha = '40ce1c875acf016ab7742ecdc45d3bad0fe8c51d3ead0ca2263ec5eb4e96f42d';

type FetchFixture = {
  readonly root: string;
  readonly target: string;
  readonly good: Buffer;
  run(
    content: string,
    fail?: boolean,
    args?: string[],
    extraEnv?: NodeJS.ProcessEnv,
    cwd?: string
  ): SpawnSyncReturns<string>;
};

async function fixture(): Promise<FetchFixture> {
  const root = await mkdtemp(join(tmpdir(), 'kev-fetch-test-'));
  await mkdir(join(root, 'scripts'));
  await mkdir(join(root, 'bin'));
  const good = Buffer.alloc(2267, 42);
  const hash = createHash('sha256').update(good).digest('hex');
  const source = await readFile(resolve('scripts/fetch-model.sh'), 'utf8');
  assert.equal(source.split(configSha).length, 2);
  const legacyHash = createHash('sha256').update('legacy').digest('hex');
  const fixtureSource = source
    .replace(configSha, hash)
    .replace('graph_size=3338522', 'graph_size=6')
    .replace('26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8', legacyHash);
  await writeFile(join(root, 'scripts/fetch-model.sh'), fixtureSource);
  await writeFile(join(root, 'good'), good);
  await writeFile(join(root, 'bad'), Buffer.alloc(good.length, 43));
  await writeFile(join(root, 'short'), 'truncated');
  const curl = join(root, 'bin/curl');
  await writeFile(
    curl,
    `#!/usr/bin/env bash
set -eu
url=''
out=''
while (( $# )); do
  case "$1" in
    http*) url="$1" ;;
    -o) shift; out="$1" ;;
  esac
  shift
done
case "$url" in */config.json) cp "$FAKE_CONTENT" "$out"; if [[ "$FAKE_FAIL" == 1 ]]; then exit 22; fi ;; *) exit 22 ;; esac
`
  );
  await chmod(curl, 0o755);
  const target = join(root, 'models/kev-0.6b-onnx/config.json');
  await mkdir(join(root, 'models/kev-0.6b-onnx/onnx'), {recursive: true});
  return {
    root,
    target,
    good,
    run(
      content: string,
      fail = false,
      args: string[] = [],
      extraEnv: NodeJS.ProcessEnv = {},
      cwd?: string
    ): SpawnSyncReturns<string> {
      return spawnSync('bash', [join(root, 'scripts/fetch-model.sh'), ...args], {
        encoding: 'utf8',
        cwd,
        env: {
          ...process.env,
          PATH: `${join(root, 'bin')}:${process.env.PATH}`,
          FAKE_CONTENT: join(root, content),
          FAKE_FAIL: fail ? '1' : '',
          ...extraEnv
        }
      });
    }
  };
}

test('repairs a corrupt cache only after validation and preserves its old bytes', async () => {
  const f = await fixture();
  try {
    await copyFile(join(f.root, 'bad'), f.target);
    await writeFile(`${f.target}.part`, 'do not overwrite');
    const result = f.run('good');
    // The fake downloader refuses the next file; the first replacement has completed.
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Preserved existing model file/);
    assert.deepEqual(await readFile(f.target), f.good);
    const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
    const backups = names.filter((name) => name.startsWith('config.json.backup.'));
    assert.equal(backups.length, 1);
    const [backup] = backups;
    assert.ok(backup);
    assert.deepEqual(
      await readFile(join(f.root, 'models/kev-0.6b-onnx', backup, 'config.json')),
      Buffer.alloc(2267, 43)
    );
    assert.equal(await readFile(`${f.target}.part`, 'utf8'), 'do not overwrite');
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('truncated and failed transfers preserve the existing cache', async () => {
  const f = await fixture();
  try {
    await writeFile(f.target, 'old data');
    const short = f.run('short');
    assert.notEqual(short.status, 0);
    assert.match(short.stderr, /size mismatch/);
    const failed = f.run('good', true);
    assert.notEqual(failed.status, 0);
    assert.match(failed.stderr, /Download failed/);
    assert.equal(await readFile(f.target, 'utf8'), 'old data');
    const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
    assert.equal(names.filter((name) => name.startsWith('config.json.download.')).length, 2);
    assert.equal(
      names.some((name) => name.startsWith('config.json.backup.')),
      false
    );
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('does not follow an existing symlink to another file', async () => {
  const f = await fixture();
  try {
    await symlink(join(f.root, 'bad'), f.target);
    const result = f.run('good');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /not a regular file/);
    assert.deepEqual(await readFile(join(f.root, 'bad')), Buffer.alloc(2267, 43));
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('legacy migration preserves existing destination paths including dangling symlinks', async () => {
  const f = await fixture();
  try {
    const directory = join(f.root, 'models/kev-0.6b-onnx');
    const legacy = join(directory, 'model_q4f16.onnx');
    const current = join(directory, 'onnx/model_q4f16.onnx');
    await writeFile(legacy, 'legacy');
    await symlink('missing-target', current);
    f.run('bad');
    assert.equal(await readFile(legacy, 'utf8'), 'legacy');
    assert((await lstat(current)).isSymbolicLink());
    assert.equal(await readlink(current), 'missing-target');
    await rm(current);
    await writeFile(current, 'existing');
    f.run('bad');
    assert.equal(await readFile(legacy, 'utf8'), 'legacy');
    assert.equal(await readFile(current, 'utf8'), 'existing');
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('legacy source symlinks are rejected and regular legacy paths are retained', async () => {
  const f = await fixture();
  try {
    const directory = join(f.root, 'models/kev-0.6b-onnx');
    const legacy = join(directory, 'model_q4f16.onnx');
    const current = join(directory, 'onnx/model_q4f16.onnx');
    await symlink('missing-target', legacy);
    const result = f.run('bad');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Legacy model path must not be a symlink/);
    assert((await lstat(legacy)).isSymbolicLink());
    await assert.rejects(lstat(current), {code: 'ENOENT'});
    await rm(legacy);
    await writeFile(legacy, 'legacy');
    f.run('bad');
    assert.equal(await readFile(legacy, 'utf8'), 'legacy');
    assert.equal(await readFile(current, 'utf8'), 'legacy');
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('invalid legacy source is preserved without creating an unverified canonical path', async () => {
  const f = await fixture();
  try {
    const directory = join(f.root, 'models/kev-0.6b-onnx');
    const legacy = join(directory, 'model_q4f16.onnx');
    const current = join(directory, 'onnx/model_q4f16.onnx');
    await writeFile(legacy, 'badbad');
    const result = f.run('bad');
    assert.match(result.stderr, /Legacy model file differs/);
    assert.equal(await readFile(legacy, 'utf8'), 'badbad');
    await assert.rejects(lstat(current), {code: 'ENOENT'});
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('a competing file after backup is never overwritten', async () => {
  const f = await fixture();
  try {
    await writeFile(f.target, 'old data');
    const python = join(f.root, 'bin/python3');
    await writeFile(
      python,
      '#!/usr/bin/env bash\nif [[ $# == 2 ]]; then exit 0; fi\nprintf concurrent > "$4"\nexit 1\n'
    );
    await chmod(python, 0o755);
    const result = f.run('good');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /without overwriting/);
    assert.equal(await readFile(f.target, 'utf8'), 'concurrent');
    const dir = join(f.root, 'models/kev-0.6b-onnx');
    const names = await readdir(dir);
    const backup = names.find((name) => name.startsWith('config.json.backup.'));
    const partial = names.find((name) => name.startsWith('config.json.download.'));
    assert.ok(backup);
    assert.ok(partial);
    assert.equal(await readFile(join(dir, backup, 'config.json'), 'utf8'), 'old data');
    assert.deepEqual(await readFile(join(dir, partial)), f.good);
    assert.ok(result.stderr.includes(backup));
    assert.ok(result.stderr.includes(partial));
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('missing Python link support fails before changing existing files', async () => {
  const f = await fixture();
  try {
    await writeFile(f.target, 'old data');
    const python = join(f.root, 'bin/python3');
    await writeFile(python, '#!/usr/bin/env bash\nexit 1\n');
    await chmod(python, 0o755);
    const result = f.run('good');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /python3 with no-follow os\.link support/);
    assert.equal(await readFile(f.target, 'utf8'), 'old data');
    const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
    assert.equal(
      names.some((name) => name.startsWith('config.json.download.')),
      false
    );
    assert.equal(
      names.some((name) => name.startsWith('config.json.backup.')),
      false
    );
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('failed replacements leave the old cache intact and use unique partial paths', async () => {
  const f = await fixture();
  try {
    await writeFile(f.target, 'old data');
    for (let i = 0; i < 2; i++) {
      const result = f.run('bad');
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /SHA-256 mismatch/);
      assert.equal(await readFile(f.target, 'utf8'), 'old data');
    }
    const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
    assert.equal(names.filter((name) => name.startsWith('config.json.download.')).length, 2);
    assert.equal(names.filter((name) => name.startsWith('config.json.backup.')).length, 0);
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('verified existing files are reused without a download or backup', async () => {
  const f = await fixture();
  try {
    await writeFile(f.target, f.good);
    const result = f.run('bad');
    assert.notEqual(result.status, 0);
    assert.match(result.stdout, /cached config\.json/);
    assert.deepEqual(await readFile(f.target), f.good);
    const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
    assert.equal(
      names.some((name) => name.startsWith('config.json.backup.')),
      false
    );
    assert.equal(
      names.some((name) => name.startsWith('config.json.download.')),
      false
    );
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('all pinned files, including optional Q4, download and are reused', async () => {
  const root = await mkdtemp(join(tmpdir(), 'kev-fetch-all-'));
  try {
    await mkdir(join(root, 'scripts'));
    await mkdir(join(root, 'bin'));
    const sha = createHash('sha256').update('abc').digest('hex');
    const source = await readFile(resolve('scripts/fetch-model.sh'), 'utf8');
    let count = 0;
    const fixtureSource = source
      .replace(/^(\s*)'(\S+) \d+ [a-f0-9]{64}'$/gm, (_all, indent, file) => {
        count++;
        return `${indent}'${file} 3 ${sha}'`;
      })
      .replace('graph_size=3338522', 'graph_size=3')
      .replace('26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8', sha);
    assert.equal(count, 6);
    await writeFile(join(root, 'scripts/fetch-model.sh'), fixtureSource);
    const curl = join(root, 'bin/curl');
    await writeFile(
      curl,
      `#!/usr/bin/env bash
set -eu
if [[ "$FAKE_FAIL" == 1 ]]; then exit 22; fi
url=''
while (( $# )); do
  if [[ "$1" == http* ]]; then url="$1"; fi
  if [[ "$1" == -o ]]; then
    shift
    if [[ "$FAKE_TAMPER" == 1 && "$url" == */tokenizer.json ]]; then
      printf xyz > models/kev-0.6b-onnx/config.json
    fi
    printf abc > "$1"
    exit 0
  fi
  shift
done
exit 1
`
    );
    await chmod(curl, 0o755);
    const run = (fail: boolean, tamper = false, args: string[] = []): SpawnSyncReturns<string> =>
      spawnSync('bash', [join(root, 'scripts/fetch-model.sh'), '--with-q4', ...args], {
        encoding: 'utf8',
        env: {
          ...process.env,
          PATH: `${join(root, 'bin')}:${process.env.PATH}`,
          FAKE_FAIL: fail ? '1' : '',
          FAKE_TAMPER: tamper ? '1' : ''
        }
      });
    const first = run(false);
    assert.equal(first.status, 0, first.stderr);
    assert.equal(await readFile(join(root, 'models/kev-0.6b-onnx/onnx/model_q4.onnx_data'), 'utf8'), 'abc');
    const second = run(true);
    assert.equal(second.status, 0, second.stderr);
    assert.equal((second.stdout.match(/^cached /gm) ?? []).length, 7);
    const q4Partial = join(root, 'retained-q4');
    await writeFile(q4Partial, 'abc');
    await rm(join(root, 'models/kev-0.6b-onnx/onnx/model_q4.onnx_data'));
    const resumed = run(true, false, ['--resume', 'onnx/model_q4.onnx_data', q4Partial]);
    assert.equal(resumed.status, 0, resumed.stderr);
    assert.equal(await readFile(join(root, 'models/kev-0.6b-onnx/onnx/model_q4.onnx_data'), 'utf8'), 'abc');
    assert.equal(await readFile(q4Partial, 'utf8'), 'abc');
    await rm(join(root, 'models/kev-0.6b-onnx/tokenizer.json'));
    const changed = run(false, true);
    assert.notEqual(changed.status, 0);
    assert.match(changed.stderr, /Final model file SHA-256 mismatch:.*config\.json/);
  } finally {
    await rm(root, {recursive: true, force: true});
  }
});

test('preview setup uses pinned direct Kev inputs and the direct converter', async () => {
  const setup = await readFile(resolve('scripts/setup_preview.sh'), 'utf8');
  for (const revision of [
    'abe7110dedfd68a185839550079bcf7c26f561ac',
    'da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
    '37cb2f6155afec1054de2cc78dfd7c7dd24a956d',
    '7b5acb32bd271342a24130f4ef439ff6c2e52c6c',
    'd7b4130688bfa348392b921ab04015aa15c4f154'
  ]) {
    assert.ok(setup.includes(revision), `Preview setup is missing pinned revision ${revision}.`);
  }
  assert.match(setup, /"\$hf" download --quiet --revision "\$kev_checkpoint_revision" jaredpalmer\/kev-0\.6b/);
  assert.match(setup, /"\$hf" download --quiet --revision "\$base_revision" Qwen\/Qwen3-0\.6B-Base/);
  assert.match(setup, /from build_tvm import verify_inputs/);
  assert.match(setup, /"\$converter\/build_tvm\.py"/);
  assert.ok(
    setup.indexOf('from build_tvm import verify_inputs') < setup.indexOf('git clone https://github.com/apache/tvm.git'),
    'Preview setup must verify the input hashes before setting up TVM.'
  );
  for (const input of ['--kev-dir', '--base-dir', '--source-dir', '--tvm-home', '--libatomic-path', '--output-dir']) {
    assert.ok(setup.includes(input), `Preview setup is missing direct-converter argument ${input}.`);
  }
  for (const legacyStep of [
    'scripts/fetch-model.sh',
    'scripts/specialize_q4_direct.py',
    'scripts/build_tvm_q4.py',
    'scripts/verify_q4_cache.py',
    'scripts/prepare_web_decision_assets.py'
  ]) {
    assert.equal(setup.includes(legacyStep), false, `Preview setup still invokes ${legacyStep}.`);
  }
});

test('preview setup refuses symlinked direct-conversion parents and candidate output', async (t) => {
  const cases = [
    {name: 'model output parent', path: 'models', kind: 'parent'},
    {name: 'candidate parent', path: 'models/web-decision-assets', kind: 'parent'},
    {name: 'toolchains parent', path: '.toolchains', kind: 'parent'},
    {name: 'converter parent', path: 'tools/kev-converter', kind: 'parent'},
    {name: 'candidate output', path: 'models/web-decision-assets/direct-kev-0.6b-q4f16', kind: 'output'}
  ];
  for (const target of cases) {
    await t.test(target.name, async () => {
      const root = await mkdtemp(join(tmpdir(), 'kev-preview-symlink-'));
      const outside = join(root, 'outside');
      const setup = join(root, 'scripts/setup_preview.sh');
      try {
        await mkdir(join(root, 'scripts'));
        await copyFile(resolve('scripts/setup_preview.sh'), setup);
        await mkdir(outside);
        const sentinel = join(outside, 'sentinel');
        await writeFile(sentinel, 'preserve me');

        if (target.kind === 'parent' && target.path.includes('/')) {
          await mkdir(join(root, target.path.split('/')[0]!));
          await symlink(outside, join(root, target.path), 'dir');
        } else if (target.kind === 'parent') {
          await symlink(outside, join(root, target.path), 'dir');
        } else {
          await mkdir(join(root, 'models/web-decision-assets'), {recursive: true});
          await symlink(sentinel, join(root, target.path));
        }

        const result = spawnSync('bash', [setup], {encoding: 'utf8'});
        assert.notEqual(result.status, 0);
        assert.match(result.stderr, target.kind === 'output' ? /Refusing to overwrite/ : /symlinked directory/);
        assert.equal(await readFile(sentinel, 'utf8'), 'preserve me');
      } finally {
        await rm(root, {recursive: true, force: true});
      }
    });
  }
});

test('model fetch refuses a symlinked ONNX output directory', async () => {
  const f = await fixture();
  try {
    const modelRoot = join(f.root, 'models/kev-0.6b-onnx');
    const onnx = join(modelRoot, 'onnx');
    const outside = join(f.root, 'outside');
    const sentinel = join(outside, 'sentinel');
    await mkdir(outside);
    await writeFile(sentinel, 'preserve me');
    await rm(onnx, {recursive: true});
    await symlink(outside, onnx, 'dir');

    const result = f.run('good');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Model output directory must not be a symlink/);
    assert.equal(await readFile(sentinel, 'utf8'), 'preserve me');
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('resume rejects invalid selections and changing or unsafe partial files', async () => {
  const f = await fixture();
  try {
    const partial = join(f.root, 'partial');
    await writeFile(partial, f.good.subarray(0, 32));
    const badSelection = f.run('good', false, ['--resume', 'onnx/model_q4.onnx_data', partial]);
    assert.match(badSelection.stderr, /not selected/);
    const q4Selection = f.run('good', true, ['--with-q4', '--resume', 'onnx/model_q4.onnx_data', partial]);
    assert.doesNotMatch(q4Selection.stderr, /not selected/);
    assert.match(q4Selection.stderr, /Download failed/);
    const symlinkPath = join(f.root, 'partial-link');
    await symlink(partial, symlinkPath);
    const symlinkResult = f.run('good', false, ['--resume', 'config.json', symlinkPath]);
    assert.match(symlinkResult.stderr, /Cannot copy partial file/);
    assert.deepEqual(await readFile(partial), f.good.subarray(0, 32));

    await writeFile(
      join(f.root, 'sitecustomize.py'),
      `import os, shutil
original = shutil.copyfileobj
def change_source(source, destination, *args):
    original(source, destination, *args)
    with open(os.environ['CHANGE_PARTIAL'], 'ab') as f:
        f.write(b'x')
shutil.copyfileobj = change_source
`
    );
    const changed = f.run('good', false, ['--resume', 'config.json', partial], {
      PYTHONPATH: f.root,
      CHANGE_PARTIAL: partial
    });
    assert.match(changed.stderr, /partial changed while copying/);
    await assert.rejects(lstat(f.target), {code: 'ENOENT'});
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

async function rangeServer(
  mode: 'correct' | 'ignored' | 'wrong-range' | 'corrupt' | 'failure' | 'oversized-chunked',
  bytes: Buffer
): Promise<{readonly server: ReturnType<typeof spawn>; readonly port: number}> {
  const script = `
    const http = require('node:http');
    const bytes = Buffer.from(process.env.BODY, 'base64');
    http.createServer((req, res) => {
      const range = req.headers.range;
      if (!req.url.endsWith('/config.json')) return res.writeHead(404).end();
      if (process.env.MODE === 'failure') return res.writeHead(500).end();
      if (process.env.MODE === 'ignored') return res.writeHead(200).end(bytes);
      if (process.env.MODE === 'oversized-chunked') {
        res.writeHead(200, {'Transfer-Encoding': 'chunked'});
        return res.end(bytes);
      }
      const start = Number(/^bytes=(\\d+)-$/.exec(range)?.[1]);
      if (!Number.isSafeInteger(start)) return res.writeHead(400).end();
      const headerStart = process.env.MODE === 'wrong-range' ? start + 1 : start;
      const body = Buffer.from(bytes.subarray(start));
      if (process.env.MODE === 'corrupt') body[0] ^= 1;
      res.writeHead(206, {'Content-Range': 'bytes ' + headerStart + '-' + (bytes.length - 1) + '/' + bytes.length,
        'Content-Length': body.length});
      res.end(body);
    }).listen(0, '127.0.0.1', function () { console.log(this.address().port); });
  `;
  const server = spawn(process.execPath, ['-e', script], {
    env: {...process.env, MODE: mode, BODY: bytes.toString('base64')},
    stdio: ['ignore', 'pipe', 'pipe']
  });
  const port = await new Promise<number>((resolvePort, reject) => {
    server.stdout.once('data', (chunk) => resolvePort(Number(String(chunk).trim())));
    server.once('error', reject);
    server.once('exit', (code) => reject(new Error(`Range server exited: ${code}`)));
  });
  return {server, port};
}

test('resumes with a real HTTP Range response and verifies before replacing', async () => {
  const modes = ['correct', 'ignored', 'wrong-range', 'corrupt', 'failure'] as const;
  for (const mode of modes) {
    const f = await fixture();
    const {server, port} = await rangeServer(mode, f.good);
    try {
      const scriptPath = join(f.root, 'scripts/fetch-model.sh');
      const source = await readFile(scriptPath, 'utf8');
      await writeFile(
        scriptPath,
        source.replace(
          'https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/$revision',
          `http://127.0.0.1:${port}`
        )
      );
      const partial = join(f.root, 'retained');
      const bytes = f.good.subarray(0, 100);
      await writeFile(partial, bytes);
      await writeFile(f.target, 'existing');
      const result = spawnSync('bash', [scriptPath, '--resume', 'config.json', partial, '--with-q4'], {
        encoding: 'utf8',
        env: {...process.env}
      });
      assert.notEqual(result.status, 0); // The next model file returns 404.
      assert.deepEqual(await readFile(partial), bytes, mode);
      if (mode === 'correct') {
        assert.deepEqual(await readFile(f.target), f.good);
        assert.match(result.stderr, /Preserved existing model file/);
      } else {
        assert.equal(await readFile(f.target, 'utf8'), 'existing', mode);
        assert.match(result.stderr, mode === 'corrupt' ? /SHA-256 mismatch/ : /Download failed/, mode);
      }
    } finally {
      server.kill('SIGTERM');
      await rm(f.root, {recursive: true, force: true});
    }
  }
});

test('caps chunked downloads without Content-Length at the pinned file size', async () => {
  const f = await fixture();
  const {server, port} = await rangeServer('oversized-chunked', Buffer.alloc(f.good.length * 8, 42));
  try {
    const scriptPath = join(f.root, 'scripts/fetch-model.sh');
    const source = await readFile(scriptPath, 'utf8');
    await writeFile(
      scriptPath,
      source.replace(
        'https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/$revision',
        `http://127.0.0.1:${port}`
      )
    );
    await writeFile(f.target, 'existing');

    const result = spawnSync('bash', [scriptPath], {encoding: 'utf8', env: {...process.env}});
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Download failed/);
    assert.equal(await readFile(f.target, 'utf8'), 'existing');

    const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
    const partial = names.find((name) => name.startsWith('config.json.download.'));
    assert.ok(partial);
    const partialBytes = (await lstat(join(f.root, 'models/kev-0.6b-onnx', partial))).size;
    const maxBytes = Math.ceil(f.good.length / 512) * 512;
    assert.ok(partialBytes <= maxBytes);
    assert.ok(partialBytes < f.good.length * 8);
  } finally {
    server.kill('SIGTERM');
    await rm(f.root, {recursive: true, force: true});
  }
});

test('does not start curl when the hard file-size limit is below the pinned size', async () => {
  const f = await fixture();
  try {
    const curlMarker = join(f.root, 'curl-called');
    const curl = join(f.root, 'bin/curl');
    await writeFile(curl, '#!/usr/bin/env bash\n: > "$CURL_MARKER"\nexit 98\n');
    await chmod(curl, 0o755);

    const result = spawnSync(
      'bash',
      [
        '-c',
        'ulimit -S -f 1 && ulimit -H -f 1 && exec bash "$1"',
        'hard-limit-test',
        join(f.root, 'scripts/fetch-model.sh')
      ],
      {
        encoding: 'utf8',
        env: {
          ...process.env,
          PATH: `${join(f.root, 'bin')}:${process.env.PATH}`,
          CURL_MARKER: curlMarker
        }
      }
    );
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Download failed/);
    await assert.rejects(lstat(curlMarker), {code: 'ENOENT'});
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('complete retained files are checked without a download', async () => {
  const f = await fixture();
  try {
    const partial = join(f.root, 'partial');
    await writeFile(partial, f.good);
    const result = f.run('bad', true, ['--resume', 'config.json', partial]);
    assert.match(result.stdout, /Checking complete retained file/);
    assert.deepEqual(await readFile(f.target), f.good);
    assert.deepEqual(await readFile(partial), f.good);
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('resume rejects the destination itself and its hard links without changing files', async () => {
  for (const alias of ['same', 'hardlink']) {
    const f = await fixture();
    try {
      const bytes = f.good.subarray(0, 32);
      await writeFile(f.target, bytes);
      const partial = alias === 'same' ? f.target : join(f.root, 'partial-hardlink');
      if (alias === 'hardlink') await link(f.target, partial);
      const before = await lstat(f.target);
      const result = f.run('good', false, ['--resume', 'config.json', partial]);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /partial.*same file.*destination/i);
      assert.deepEqual(await readFile(f.target), bytes);
      assert.deepEqual(await readFile(partial), bytes);
      assert.equal((await lstat(f.target)).ino, before.ino);
      assert.equal(
        (await readdir(join(f.root, 'models/kev-0.6b-onnx'))).some((name) => name.startsWith('config.json.backup.')),
        false
      );
    } finally {
      await rm(f.root, {recursive: true, force: true});
    }
  }
});

test('relative resume path is resolved against the caller directory', async () => {
  const f = await fixture();
  try {
    const caller = join(f.root, 'caller');
    await mkdir(caller);
    const partial = join(caller, 'partial');
    const bytes = f.good.subarray(0, 32);
    await writeFile(partial, bytes);
    await writeFile(f.target, 'existing');
    const result = f.run('good', false, ['--resume', 'config.json', 'partial'], {}, caller);
    assert.doesNotMatch(result.stderr, /Cannot copy partial file/);
    assert.deepEqual(await readFile(partial), bytes);
    assert.deepEqual(await readFile(f.target), f.good);
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('invalid later resume input fails before downloads or legacy migration', async () => {
  for (const kind of ['missing', 'directory', 'too-large']) {
    const f = await fixture();
    try {
      const partial = join(f.root, 'invalid');
      if (kind === 'directory') await mkdir(partial);
      if (kind === 'too-large') await writeFile(partial, Buffer.alloc(2268));
      await writeFile(f.target, 'existing');
      const legacy = join(f.root, 'models/kev-0.6b-onnx/model_q4f16.onnx');
      await writeFile(legacy, 'legacy');
      const result = f.run('good', false, ['--resume', 'config.json', partial]);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /partial/i);
      assert.equal(await readFile(f.target, 'utf8'), 'existing');
      await assert.rejects(lstat(join(f.root, 'models/kev-0.6b-onnx/onnx/model_q4f16.onnx')), {code: 'ENOENT'});
      const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
      assert.equal(
        names.some((name) => name.includes('.backup.') || name.includes('.download.')),
        false
      );
    } finally {
      await rm(f.root, {recursive: true, force: true});
    }
  }
});

test('invalid partial for a later selected model fails before earlier files change', async () => {
  const f = await fixture();
  try {
    await writeFile(f.target, 'existing');
    const result = f.run('good', false, ['--resume', 'onnx/model_q4f16.onnx_data', join(f.root, 'missing')]);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /partial/i);
    assert.equal(await readFile(f.target, 'utf8'), 'existing');
    assert.equal(
      (await readdir(join(f.root, 'models/kev-0.6b-onnx'))).some((name) => name.startsWith('config.json.backup.')),
      false
    );
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('resuming from the legacy graph does not migrate it into the destination first', async () => {
  const f = await fixture();
  try {
    const scriptPath = join(f.root, 'scripts/fetch-model.sh');
    const source = await readFile(scriptPath, 'utf8');
    const singleGraph = source.replace(
      /files=\(\n[\s\S]*?\n\)\nif \[\[ "\$with_q4"/,
      'files=(\n  "onnx/model_q4f16.onnx $graph_size $graph_sha"\n)\nif [[ "$with_q4"'
    );
    assert.notEqual(singleGraph, source);
    await writeFile(scriptPath, singleGraph);
    const legacy = join(f.root, 'models/kev-0.6b-onnx/model_q4f16.onnx');
    const target = join(f.root, 'models/kev-0.6b-onnx/onnx/model_q4f16.onnx');
    await writeFile(legacy, 'legacy');
    const result = f.run('bad', true, ['--resume', 'onnx/model_q4f16.onnx', legacy]);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(await readFile(legacy, 'utf8'), 'legacy');
    assert.equal(await readFile(target, 'utf8'), 'legacy');
    assert.notEqual((await lstat(legacy)).ino, (await lstat(target)).ino);
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});

test('resume rejects partials that are any other selected destination or its hard link', async () => {
  for (const alias of ['same', 'hardlink', 'symlink', 'later']) {
    const f = await fixture();
    try {
      const other = alias === 'later' ? join(f.root, 'models/kev-0.6b-onnx/onnx/model_q4f16.onnx_data') : f.target;
      const bytes = Buffer.from('partial bytes');
      const partial = ['hardlink', 'symlink'].includes(alias) ? join(f.root, 'partial-alias') : other;
      if (alias === 'symlink') {
        await writeFile(partial, bytes);
        await symlink(partial, other);
      } else {
        await writeFile(other, bytes);
        if (alias === 'hardlink') await link(other, partial);
      }
      const before = await lstat(other);
      const resume = alias === 'later' ? 'config.json' : 'onnx/model_q4f16.onnx_data';
      const result = f.run('good', false, ['--resume', resume, partial]);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /partial.*same file.*destination/i);
      assert.equal((await lstat(other)).ino, before.ino);
      assert.deepEqual(await readFile(other), bytes);
      assert.deepEqual(await readFile(partial), bytes);
      const names = await readdir(join(f.root, 'models/kev-0.6b-onnx'));
      assert.equal(
        names.some((name) => name.includes('.backup.') || name.includes('.download.')),
        false
      );
    } finally {
      await rm(f.root, {recursive: true, force: true});
    }
  }
});

test('legacy migration does not alias a partial for another selected model', async () => {
  const f = await fixture();
  try {
    const legacy = join(f.root, 'models/kev-0.6b-onnx/model_q4f16.onnx');
    const graph = join(f.root, 'models/kev-0.6b-onnx/onnx/model_q4f16.onnx');
    await writeFile(legacy, 'legacy');
    const result = f.run('bad', true, ['--resume', 'onnx/model_q4f16.onnx_data', legacy]);
    assert.notEqual(result.status, 0);
    assert.equal(await readFile(legacy, 'utf8'), 'legacy');
    await assert.rejects(lstat(graph), {code: 'ENOENT'});
  } finally {
    await rm(f.root, {recursive: true, force: true});
  }
});
