import {defineConfig, type Connect} from 'vite';
import {createReadStream, realpathSync} from 'node:fs';
import {stat} from 'node:fs/promises';
import {dirname, extname, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const workspace = fileURLToPath(new URL('../../', import.meta.url));
const openDeciderAssets = process.env.WEB_DECISION_OPENDECIDER_NANO_ASSETS
  ? resolve(process.env.WEB_DECISION_OPENDECIDER_NANO_ASSETS)
  : fileURLToPath(
      new URL('../../experiments/opendecider-nano-tvm-20261102/webgpu-build-int32-export/', import.meta.url)
    );
const publicAssetsRoot = process.env.WEB_DECISION_PUBLIC_ASSET_ROOT
  ? resolve(process.env.WEB_DECISION_PUBLIC_ASSET_ROOT)
  : fileURLToPath(new URL('../../models/web-decision-assets/', import.meta.url));
const tinyDecideAssets = process.env.WEB_DECISION_TINYDECIDE_ASSETS
  ? resolve(process.env.WEB_DECISION_TINYDECIDE_ASSETS)
  : fileURLToPath(new URL('../../models/tinydecide-aa745b73/', import.meta.url));
const webDecisionBundle = process.env.WEB_DECISION_INSTALLED_BUNDLE
  ? resolve(process.env.WEB_DECISION_INSTALLED_BUNDLE)
  : fileURLToPath(new URL('../../packages/web-decision/dist/index.js', import.meta.url));
const packageConsumerRoot = process.env.WEB_DECISION_INSTALLED_BUNDLE
  ? realpathSync(resolve(dirname(webDecisionBundle), '../../..'))
  : null;
const port = Number(process.env.WEB_DECISION_BROWSER_PORT || 41773);
const contentTypes = new Map([
  ['.bin', 'application/octet-stream'],
  ['.json', 'application/json'],
  ['.wasm', 'application/wasm']
]);

function serveModelAsset(assets: string): Connect.NextHandleFunction {
  const root = resolve(assets);
  return async function serve(request, response, next) {
    if (request.method !== 'GET' && request.method !== 'HEAD') return next();
    let file;
    try {
      const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
      file = resolve(assets, `.${pathname}`);
      if (file !== root && !file.startsWith(`${root}${sep}`)) return next();
      const info = await stat(file);
      if (!info.isFile()) return next();
      response.statusCode = 200;
      response.setHeader('Content-Length', info.size);
      response.setHeader('Content-Type', contentTypes.get(extname(file)) ?? 'application/octet-stream');
      if (request.method === 'HEAD') return response.end();
      createReadStream(file).pipe(response);
    } catch {
      next();
    }
  };
}

const serveOpenDeciderAsset = serveModelAsset(openDeciderAssets);
const serveTinyDecideAsset = serveModelAsset(tinyDecideAssets);

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  publicDir: publicAssetsRoot,
  // Use the fresh workspace build by default, or an explicit installed package
  // copy for package-consumer browser checks.
  optimizeDeps: {noDiscovery: true, include: []},
  resolve: {
    alias: {
      '@huggingface/tokenizers': realpathSync(
        fileURLToPath(
          new URL(
            '../../packages/web-decision/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs',
            import.meta.url
          )
        )
      ),
      'web-decision': webDecisionBundle
    }
  },
  server: {
    host: '127.0.0.1',
    port,
    strictPort: true,
    fs: {allow: [workspace, ...(packageConsumerRoot ? [packageConsumerRoot] : [])]}
  },
  plugins: [
    {
      name: 'browser-test-manifests',
      configureServer(server) {
        server.middlewares.use('/opendecider-nano-assets', serveOpenDeciderAsset);
        server.middlewares.use('/tinydecide-assets', serveTinyDecideAsset);
        server.middlewares.use('/test-missing/manifest.json', (_request, response) => {
          response.statusCode = 404;
          response.end('Not found');
        });
        server.middlewares.use('/test-missing', (_request, response) => {
          response.statusCode = 404;
          response.end('Not found');
        });
        server.middlewares.use('/test-stall/manifest.json', (request, response) => {
          request.on('close', () => {
            if (!response.writableEnded) response.end();
          });
        });
        server.middlewares.use('/test-stall', (request, response) => {
          request.on('close', () => {
            if (!response.writableEnded) response.end();
          });
        });
      }
    }
  ]
});
