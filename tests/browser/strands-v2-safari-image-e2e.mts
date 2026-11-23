/** Local-only Safari automation for the private single-image model probe. */
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createServer} from 'node:net';
import {resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';
import {createServer as createViteServer, type Plugin} from 'vite';
import {startLocalRangeServer} from './strands-v2-local-range-server.mts';

async function unusedPort(): Promise<number> {
  const server = createServer();
  await new Promise<void>((resolve, reject) => server.once('error', reject).listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Could not allocate a local port.');
  await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
  return address.port;
}

type DriverRequest = (path: string, body?: unknown, method?: string) => Promise<unknown>;
type Evaluate = (script: string, args?: unknown[]) => Promise<unknown>;

async function waitForDriver(driver: ReturnType<typeof spawn>, request: DriverRequest, signal: AbortSignal) {
  let ready = false;
  for (let attempt = 0; attempt < 100 && !ready; attempt++) {
    if (driver.exitCode !== null) throw new Error(`Safari driver exited: ${driver.exitCode}`);
    try {
      await request('/status', undefined, 'GET');
      ready = true;
    } catch (error) {
      if (signal.aborted) throw error;
      await delay(100, undefined, {signal: signal});
    }
  }
  if (!ready) throw new Error('Safari driver did not start.');
}

async function checkLocalRanges(evaluate: Evaluate) {
  const assets = (await evaluate(
    `const done=arguments[arguments.length-1]; Promise.all(['/strands-v2-image-e2e-assets/linked.wasm','/strands-v2-image-e2e-source/model.safetensors'].map(async url=>{const r=await fetch(url,{headers:{Range:'bytes=0-3'}});return {status:r.status,range:r.headers.get('Content-Range'),bytes:Array.from(new Uint8Array(await r.arrayBuffer()))}})).then(value=>done({ok:true,value}),error=>done({ok:false,message:String(error)}));`
  )) as {
    ok: boolean;
    value?: Array<{status: number; range: string | null; bytes: number[]}>;
  };
  if (
    !assets.ok ||
    assets.value?.length !== 2 ||
    assets.value.some((item) => item.status !== 206 || !item.range?.startsWith('bytes 0-3/')) ||
    assets.value[0]?.bytes.join(',') !== '0,97,115,109'
  )
    throw new Error(`Safari local asset bootstrap failed: ${JSON.stringify(assets)}`);
  console.log(`SAFARI_LOCAL_RANGES ${JSON.stringify(assets.value)}`);
}

async function checkVisionRanges(evaluate: Evaluate, base: string) {
  const result = (await evaluate(
    `const done=arguments[arguments.length-1]; Promise.all(['graph.wasm','input-pixels.bin','input-positions.bin','model.safetensors'].map(async name=>{const r=await fetch(new URL(name,arguments[0]),{headers:{Range:'bytes=0-3'}});return {status:r.status,range:r.headers.get('Content-Range'),bytes:Array.from(new Uint8Array(await r.arrayBuffer()))}})).then(value=>done({ok:true,value}),error=>done({ok:false,message:String(error)}));`,
    [base]
  )) as {ok: boolean; value?: Array<{status: number; range: string | null; bytes: number[]}>};
  if (
    !result.ok ||
    result.value?.length !== 4 ||
    result.value.some((item) => item.status !== 206 || !item.range?.startsWith('bytes 0-3/')) ||
    result.value[0]?.bytes.join(',') !== '0,97,115,109'
  )
    throw new Error(`Safari vision asset bootstrap failed: ${JSON.stringify(result)}`);
  console.log(`SAFARI_VISION_RANGES ${JSON.stringify(result.value)}`);
}

async function runVisionPerformance(evaluate: Evaluate, base: string, hashes: Record<string, string>) {
  const result = (await evaluate(
    `const done=arguments[arguments.length-1]; window.runVisionPerformance(arguments[0],3).then(value=>done({ok:true,value}),error=>done({ok:false,message:String(error),stack:error?.stack}));`,
    [base]
  )) as {
    ok: boolean;
    message?: string;
    stack?: string;
    value?: {
      failure: unknown;
      trials: Array<{warmup: boolean; interval: number; elapsedMs: number; maxAbsoluteError: number}>;
    };
  };
  console.log(`SAFARI_VISION_SYNC_PERFORMANCE ${JSON.stringify({hashes, result})}`);
  if (!result.ok || !result.value)
    throw new Error(`Safari vision performance failed: ${result.message}\n${result.stack}`);
  if (result.value.failure)
    throw new Error(`Safari vision performance reported failure: ${JSON.stringify(result.value.failure)}`);
  for (const interval of [2, 8]) {
    const measured = result.value.trials.filter((trial) => !trial.warmup && trial.interval === interval);
    if (
      measured.length !== 3 ||
      measured.some((trial) => !(trial.elapsedMs > 0) || !(trial.maxAbsoluteError < 0.00001))
    )
      throw new Error(`Safari vision performance results invalid for interval ${interval}.`);
  }
}

async function main(): Promise<void> {
  const root = process.env.WEB_DECISION_STRANDS_V2_CHAIN_ROOT;
  const wasm = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM;
  const expectedHash = process.env.WEB_DECISION_STRANDS_V2_IMAGE_WASM_SHA256;
  if (!root || !wasm || !expectedHash || !/^[0-9a-f]{64}$/u.test(expectedHash))
    throw new Error('Set WEB_DECISION_STRANDS_V2_CHAIN_ROOT, IMAGE_WASM and IMAGE_WASM_SHA256.');
  const graph = resolve(root, wasm);
  const hash = createHash('sha256')
    .update(await readFile(graph))
    .digest('hex');
  if (hash !== expectedHash) throw new Error(`Graph SHA-256 mismatch: ${hash}`);
  const input = JSON.parse(await readFile(resolve(root, 'e2b-v2-vision-reference/tokens.json'), 'utf8'));
  const official = JSON.parse(await readFile(resolve(root, 'e2b-v2-vision-reference/result.json'), 'utf8'));
  const files = new Map([
    ['/strands-v2-image-e2e-source/model.safetensors', resolve(root, 'strands-e2b-v2/model.safetensors')],
    ['/model.safetensors', resolve(root, 'strands-e2b-v2/model.safetensors')],
    ['/graph.wasm', graph],
    ['/input-pixels.bin', resolve(root, 'e2b-v2-vision-reference/input-pixels.bin')],
    ['/input-positions.bin', resolve(root, 'e2b-v2-vision-reference/input-positions.bin')],
    ['/strands-v2-image-e2e-assets/linked.wasm', graph],
    ['/strands-v2-image-e2e-assets/readout.bin', resolve(root, 'e2b-v2-dynamic-readout/input-2.bin')],
    ['/strands-v2-image-e2e-assets/local.bin', resolve(root, 'e2b-v2-positions/local.bin')],
    ['/strands-v2-image-e2e-assets/global.bin', resolve(root, 'e2b-v2-positions/global.bin')]
  ]);
  const abort = new AbortController();
  const interrupt = () => abort.abort(new Error('Interrupted'));
  process.once('SIGINT', interrupt);
  process.once('SIGTERM', interrupt);
  const range = await startLocalRangeServer(files);
  const redirect: Plugin = {
    name: 'safari-local-probe-assets',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const path = new URL(request.url ?? '/', 'http://localhost').pathname;
        if (!files.has(path)) return next();
        response.writeHead(307, {Location: new URL(path.slice(1), range.base).href}).end();
      });
    }
  };
  let vite: Awaited<ReturnType<typeof createViteServer>> | undefined;
  let driver: ReturnType<typeof spawn> | undefined;
  let session: string | undefined;
  const port = await unusedPort();
  const driverPort = await unusedPort();
  const driverBase = `http://127.0.0.1:${driverPort}`;
  const request = async (path: string, body?: unknown, method = 'POST') => {
    const response = await fetch(driverBase + path, {
      method,
      ...(body === undefined ? {} : {headers: {'content-type': 'application/json'}, body: JSON.stringify(body)}),
      signal: AbortSignal.any([abort.signal, AbortSignal.timeout(620000)])
    });
    const parsed = (await response.json()) as {value: unknown};
    const value = parsed.value as {error?: string; message?: string};
    if (!response.ok || value?.error) throw new Error(`Safari WebDriver: ${JSON.stringify(parsed)}`);
    return parsed.value;
  };
  try {
    vite = await createViteServer({
      configFile: resolve('tests/browser/vite.config.ts'),
      plugins: [redirect],
      server: {host: '127.0.0.1', port, strictPort: true, hmr: false}
    });
    await vite.listen();
    driver = spawn('safaridriver', ['-p', String(driverPort)], {stdio: 'ignore'});
    await waitForDriver(driver, request, abort.signal);
    const created = (await request('/session', {capabilities: {alwaysMatch: {browserName: 'safari'}}})) as {
      sessionId: string;
      capabilities: unknown;
    };
    session = created.sessionId;
    const base = `/session/${session}`;
    console.log(`SAFARI_SESSION ${JSON.stringify(created.capabilities)}`);
    await request(`${base}/timeouts`, {script: 600000});
    const performanceProbe = process.argv.includes('--vision-performance');
    const page = performanceProbe ? 'strands-v2-vision-performance.html' : 'strands-v2-image-e2e-probe.html';
    await request(`${base}/url`, {url: `http://127.0.0.1:${port}/${page}`});
    const evaluate = async (script: string, args: unknown[] = []) => request(`${base}/execute/async`, {script, args});
    const bootstrap = await evaluate(
      `const done=arguments[arguments.length-1]; let n=0; function poll(){if(typeof window[argumentsName]==='function') done({ready:true}); else if(++n>=200) done({ready:false}); else setTimeout(poll,50)} const argumentsName=arguments[0]; poll();`,
      [performanceProbe ? 'runVisionPerformance' : 'runGemma4V2ImageE2E']
    );
    if (!(bootstrap as {ready: boolean}).ready) throw new Error('Safari probe did not load.');
    console.log(`SAFARI_BOOTSTRAP ${JSON.stringify(bootstrap)}`);
    if (process.argv.includes('--bootstrap')) {
      await checkLocalRanges(evaluate);
      if (performanceProbe) await checkVisionRanges(evaluate, range.base);
      return;
    }
    if (performanceProbe) {
      const hashes = {
        graph: hash,
        pixels: createHash('sha256')
          .update(await readFile(files.get('/input-pixels.bin')!))
          .digest('hex'),
        positions: createHash('sha256')
          .update(await readFile(files.get('/input-positions.bin')!))
          .digest('hex')
      };
      await runVisionPerformance(evaluate, range.base, hashes);
      return;
    }
    const result = (await evaluate(
      `const done=arguments[arguments.length-1]; window.runGemma4V2ImageE2E(arguments[0]).then(value=>done({ok:true,value}),error=>done({ok:false,message:String(error),stack:error?.stack}));`,
      [input]
    )) as {
      ok: boolean;
      message?: string;
      stack?: string;
      value?: {tokens: number; logits: number[]; answer: {choice: string; probabilities: Record<string, number>}};
    };
    if (!result.ok || !result.value)
      throw new Error(`Safari image inference failed: ${result.message}\n${result.stack}`);
    const actual = result.value;
    const delta = Math.abs(actual.answer.probabilities.red! - official.answers.color.probabilities.red);
    if (
      actual.tokens !== 256 ||
      actual.logits.length !== 255 ||
      !actual.logits.every(Number.isFinite) ||
      actual.answer.choice !== official.answers.color.choice ||
      !(delta < 0.02)
    )
      throw new Error(`Safari image output mismatch: ${JSON.stringify({actual, delta})}`);
    console.log(
      `SAFARI_IMAGE_E2E ${JSON.stringify({choice: actual.answer.choice, red: actual.answer.probabilities.red, officialRed: official.answers.color.probabilities.red, delta, range: range.stats()})}`
    );
  } finally {
    if (session) {
      try {
        await fetch(`${driverBase}/session/${session}`, {method: 'DELETE', signal: AbortSignal.timeout(10000)});
      } catch {
        /* Continue cleanup. */
      }
    }
    driver?.kill('SIGTERM');
    await vite?.close();
    await range.close();
    process.off('SIGINT', interrupt);
    process.off('SIGTERM', interrupt);
  }
}

await main();
