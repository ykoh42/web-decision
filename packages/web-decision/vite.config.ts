import {defineConfig, type Plugin} from 'vite';
import {fileURLToPath} from 'node:url';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

// The package builds against the exact pinned upstream TVM WebGPU source.
export function assertPinnedWebgpuHash(actual: string): void {
  if (actual !== '77003ab2bc6c7ae103b38fb52c12a5a7a3d49597260b43f010deaacc5d65b7c1') {
    throw new Error('The TVM WebGPU source differs from the pinned upstream revision.');
  }
}

export function tvmRuntimeImportWithoutTypeMarker(source: string): string | null {
  const marker = '?web-decision-runtime';
  const prefix = '../../../../.toolchains/apache-tvm-src/web/src/';
  if (
    source !== `${prefix}runtime.ts${marker}` &&
    source !== `${prefix}compact.ts${marker}` &&
    source !== `${prefix}webgpu.ts${marker}`
  )
    return null;
  return source.slice(0, -marker.length);
}

export function resolveTvmTypeBoundary(): Plugin {
  return {
    name: 'resolve-tvm-type-boundary',
    enforce: 'pre',
    resolveId(source, importer) {
      const target = tvmRuntimeImportWithoutTypeMarker(source);
      return target ? this.resolve(target, importer, {skipSelf: true}) : null;
    }
  };
}

// The generated TVM WASI glue is imported by this browser-only bundle. Its
// CommonJS export is in the Node branch and has no browser consumer. Change
// only the exact pinned glue sequence; if that sequence changes, the build
// must be reviewed rather than silently rewritten.
export function stripNodeOnlyWasiExport(code: string): string {
  const target = 'if(typeof module!="undefined"){module["exports"]=Module}';
  if (code.split(target).length !== 2) {
    throw new Error('Expected exactly one Node-only export in the TVM WASI glue.');
  }
  return code.replace(target, '');
}

export function browserOnlyTvmWasiExport(): Plugin {
  let seen = false;
  return {
    name: 'browser-only-tvm-wasi-export',
    enforce: 'pre',
    buildStart() {
      seen = false;
    },
    transform(code, id) {
      const path = (id.split('?', 1)[0] ?? '').replaceAll('\\', '/');
      if (!path.endsWith('/.toolchains/apache-tvm-src/web/src/tvmjs_runtime_wasi.js')) return null;
      seen = true;
      return {code: stripNodeOnlyWasiExport(code), map: null};
    },
    generateBundle() {
      if (!seen) throw new Error('Expected TVM WASI glue in bundle');
    }
  };
}

export default defineConfig({
  base: './',
  // TVM's generated Emscripten glue probes Node globals. This package is
  // browser-only; leave those globals undefined before a consumer bundler can
  // replace `typeof __filename` with "string" and make the branch unconditional.
  define: {__filename: 'undefined', __dirname: 'undefined'},
  plugins: [
    resolveTvmTypeBoundary(),
    browserOnlyTvmWasiExport(),
    {
      name: 'verify-pinned-webgpu-source',
      buildStart() {
        const path = fileURLToPath(new URL('../../.toolchains/apache-tvm-src/web/src/webgpu.ts', import.meta.url));
        assertPinnedWebgpuHash(createHash('sha256').update(readFileSync(path)).digest('hex'));
      }
    },
    {
      name: 'normalize-tvm-source-markers',
      generateBundle(_options, bundle) {
        let sawMarker = false;
        for (const item of Object.values(bundle)) {
          if (item.type !== 'chunk') continue;
          // Rolldown emits source paths as comments. Their number of ../ segments
          // depends on the checkout location, even when the code is identical.
          const marker = /^\/\/ (?:\.\.\/)*\.toolchains\/apache-tvm-src\/web\/src\//m;
          if (!marker.test(item.code)) continue;
          sawMarker = true;
          item.code = item.code.replace(/^\/\/ (?:\.\.\/)*\.toolchains\/apache-tvm-src\/web\/src\//gm, '// [TVM]/');
        }
        if (!sawMarker) throw new Error('Expected TVM source markers in bundle');
      }
    }
  ],
  resolve: {alias: [{find: 'ws', replacement: fileURLToPath(new URL('./src/browser-only-ws.ts', import.meta.url))}]},
  build: {
    lib: {
      entry: fileURLToPath(new URL('./src/curated/index.ts', import.meta.url)),
      formats: ['es'],
      fileName: () => 'index.js'
    },
    target: 'es2022',
    minify: false,
    rollupOptions: {external: ['@huggingface/tokenizers']}
  }
});
