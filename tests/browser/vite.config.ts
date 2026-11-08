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
const decimaSmallAssets = fileURLToPath(new URL('../../models/decima-small-1.1-480c4ea/onnx/int8/', import.meta.url));
const decimaBaseCleanAssets = fileURLToPath(
  new URL('../../models/decima-base-clean-a37e006/onnx/int8/', import.meta.url)
);
const decimaAgentAssets = process.env.WEB_DECISION_DECIMA_AGENT_ASSETS
  ? resolve(process.env.WEB_DECISION_DECIMA_AGENT_ASSETS)
  : fileURLToPath(new URL('../../models/decima-agent-2.3-1acd2c4/onnx/int8/', import.meta.url));
const tamevMicroAssets = process.env.WEB_DECISION_TAMEV_MICRO_ASSETS
  ? resolve(process.env.WEB_DECISION_TAMEV_MICRO_ASSETS)
  : fileURLToPath(new URL('../../models/tamev-micro-c1b956/', import.meta.url));
const tamevNanoAssets = process.env.WEB_DECISION_TAMEV_NANO_ASSETS
  ? resolve(process.env.WEB_DECISION_TAMEV_NANO_ASSETS)
  : fileURLToPath(new URL('../../models/tamev-nano-ad15b45f/', import.meta.url));
const tamevNanoFlashAssets = process.env.WEB_DECISION_TAMEV_NANO_FLASH_ASSETS
  ? resolve(process.env.WEB_DECISION_TAMEV_NANO_FLASH_ASSETS)
  : fileURLToPath(new URL('../../models/tamev-nano-flash-8049ddb/', import.meta.url));
const skejjaAssets =
  process.env.WEB_DECISION_SKEJJA_ASSETS ?? fileURLToPath(new URL('../../models/skejja-edge32k-ft6/', import.meta.url));
const ettinAssets =
  process.env.WEB_DECISION_ETTIN_ASSETS ??
  fileURLToPath(new URL('../../models/decider-ettin-df8600e1/', import.meta.url));
const tinyDecideAssets = process.env.WEB_DECISION_TINYDECIDE_ASSETS
  ? resolve(process.env.WEB_DECISION_TINYDECIDE_ASSETS)
  : fileURLToPath(new URL('../../models/tinydecide-aa745b73/', import.meta.url));
const webDecisionBundle = process.env.WEB_DECISION_INSTALLED_BUNDLE
  ? resolve(process.env.WEB_DECISION_INSTALLED_BUNDLE)
  : fileURLToPath(new URL('../../packages/web-decision/dist/index.js', import.meta.url));
const packageConsumerRoot = process.env.WEB_DECISION_INSTALLED_BUNDLE
  ? realpathSync(resolve(dirname(webDecisionBundle), '../../..'))
  : null;
const layaAssets = fileURLToPath(new URL('../../models/laya-multilingual-onnx-46b77bb/', import.meta.url));
const layaTypedAssets = fileURLToPath(new URL('../../models/laya-typed-decisions-onnx-79c441b/', import.meta.url));
const xdecisionAssets = fileURLToPath(
  new URL('../../experiments/xdecision-conversion-20261103/output/assets/', import.meta.url)
);
const glinerMobileAssets = process.env.WEB_DECISION_GLINER_MOBILE_ASSETS
  ? resolve(process.env.WEB_DECISION_GLINER_MOBILE_ASSETS)
  : fileURLToPath(new URL('../../models/gliner2.5-decide-mobile-onnx-880d192/', import.meta.url));
const safeDecideAssets = process.env.WEB_DECISION_SAFEDECIDE_ASSETS
  ? resolve(process.env.WEB_DECISION_SAFEDECIDE_ASSETS)
  : fileURLToPath(new URL('../../models/safedecide-140m-onnx/', import.meta.url));
const crivoAssets = process.env.WEB_DECISION_CRIVO_ASSETS
  ? resolve(process.env.WEB_DECISION_CRIVO_ASSETS)
  : fileURLToPath(new URL('../../models/crivo-32m-v2-candidate-20261104/', import.meta.url));
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
const serveDecimaSmallAsset = serveModelAsset(decimaSmallAssets);
const serveDecimaBaseCleanAsset = serveModelAsset(decimaBaseCleanAssets);
const serveDecimaAgentAsset = serveModelAsset(decimaAgentAssets);
const serveTamevMicroAsset = serveModelAsset(tamevMicroAssets);
const serveTamevNanoAsset = serveModelAsset(tamevNanoAssets);
const serveTamevNanoFlashAsset = serveModelAsset(tamevNanoFlashAssets);
const serveSkejjaAsset = serveModelAsset(skejjaAssets);
const serveEttinAsset = serveModelAsset(ettinAssets);
const serveTinyDecideAsset = serveModelAsset(tinyDecideAssets);
const serveLayaAsset = serveModelAsset(layaAssets);
const serveLayaTypedAsset = serveModelAsset(layaTypedAssets);
const serveXDecisionAsset = serveModelAsset(xdecisionAssets);
const serveGlinerMobileAsset = serveModelAsset(glinerMobileAssets);
const serveSafeDecideAsset = serveModelAsset(safeDecideAssets);
const serveCrivoAsset = serveModelAsset(crivoAssets);

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  publicDir: publicAssetsRoot,
  // Use the fresh workspace build by default, or an explicit installed package
  // copy for package-consumer browser checks.
  optimizeDeps: {noDiscovery: true, include: []},
  resolve: {
    alias: {
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
        server.middlewares.use('/decima-small-assets', serveDecimaSmallAsset);
        server.middlewares.use('/decima-base-clean-assets', serveDecimaBaseCleanAsset);
        server.middlewares.use('/decima-agent-assets', serveDecimaAgentAsset);
        server.middlewares.use('/tamev-micro-assets', serveTamevMicroAsset);
        server.middlewares.use('/tamev-nano-assets', serveTamevNanoAsset);
        server.middlewares.use('/tamev-nano-flash-assets', serveTamevNanoFlashAsset);
        server.middlewares.use('/skejja-assets', serveSkejjaAsset);
        server.middlewares.use('/ettin-assets', serveEttinAsset);
        server.middlewares.use('/tinydecide-assets', serveTinyDecideAsset);
        server.middlewares.use('/laya-assets', serveLayaAsset);
        server.middlewares.use('/laya-typed-assets', serveLayaTypedAsset);
        server.middlewares.use('/xdecision-assets', serveXDecisionAsset);
        server.middlewares.use('/gliner-mobile-assets', serveGlinerMobileAsset);
        server.middlewares.use('/safedecide-140m-assets', serveSafeDecideAsset);
        server.middlewares.use('/crivo-assets', serveCrivoAsset);
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
