import {defineConfig} from 'vite';
import {fileURLToPath} from 'node:url';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

// This preview uses the exact pinned original TVM WebGPU source.
export function assertPreviewWebgpuHash(actual) {
  if (actual !== '77003ab2bc6c7ae103b38fb52c12a5a7a3d49597260b43f010deaacc5d65b7c1') {
    throw new Error(
      'This preview requires its verified TVM WebGPU source; use a clean pinned checkout for this build.'
    );
  }
}

// The generated TVM WASI glue is imported by this browser-only bundle. Its
// CommonJS export is in the Node branch and has no browser consumer. Change
// only the exact pinned glue sequence; if that sequence changes, the build
// must be reviewed rather than silently rewritten.
export function stripNodeOnlyWasiExport(code) {
  const target = 'if(typeof module!="undefined"){module["exports"]=Module}';
  if (code.split(target).length !== 2) {
    throw new Error('Expected exactly one Node-only export in the TVM WASI glue.');
  }
  return code.replace(target, '');
}

export function browserOnlyTvmWasiExport() {
  let seen = false;
  return {
    name: 'browser-only-tvm-wasi-export',
    enforce: 'pre',
    buildStart() {
      seen = false;
    },
    transform(code, id) {
      const path = id.split('?')[0].replaceAll('\\', '/');
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
  // TVM's generated Emscripten glue probes Node globals. This package is
  // browser-only; leave those globals undefined before a consumer bundler can
  // replace `typeof __filename` with "string" and make the branch unconditional.
  define: {__filename: 'undefined', __dirname: 'undefined'},
  plugins: [
    browserOnlyTvmWasiExport(),
    {
      name: 'verify-preview-webgpu-source',
      buildStart() {
        const path = fileURLToPath(new URL('../../.toolchains/apache-tvm-src/web/src/webgpu.ts', import.meta.url));
        assertPreviewWebgpuHash(createHash('sha256').update(readFileSync(path)).digest('hex'));
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
          const marker = /^\/\/ (?:\.\.\/)+\.toolchains\/apache-tvm-src\/web\/src\//m;
          if (!marker.test(item.code)) continue;
          sawMarker = true;
          item.code = item.code.replace(/^\/\/ (?:\.\.\/)+\.toolchains\/apache-tvm-src\/web\/src\//gm, '// [TVM]/');
        }
        if (!sawMarker) throw new Error('Expected TVM source markers in bundle');
      }
    }
  ],
  resolve: {alias: [{find: 'ws', replacement: fileURLToPath(new URL('./src/browser-only-ws.js', import.meta.url))}]},
  build: {
    lib: {
      entry: fileURLToPath(new URL('./src/index.js', import.meta.url)),
      formats: ['es'],
      fileName: () => 'index.js'
    },
    target: 'es2022',
    minify: false,
    rollupOptions: {external: ['@huggingface/tokenizers', 'onnxruntime-web/wasm', 'onnxruntime-web/webgpu']}
  }
});
