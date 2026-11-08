import {readFile} from 'node:fs/promises';

let testConfig;

export function initialize(data) {
  testConfig = data;
}

function moduleDataUrl(source) {
  return `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
}

export async function resolve(specifier, context, nextResolve) {
  if (
    [testConfig.bundledAssetsUrl, testConfig.stableBundledAssetsUrl].includes(context.parentURL) &&
    specifier.endsWith('?url&inline')
  ) {
    const assetUrl = new URL(specifier.slice(0, specifier.indexOf('?')), context.parentURL);
    const isStableWasm =
      context.parentURL === testConfig.stableBundledAssetsUrl && assetUrl.pathname.includes('/kev06-stable/');
    const assetPath = assetUrl.pathname.split(isStableWasm ? '/kev06-stable/' : '/experimental-kev-upstream/').at(-1);
    const assets = isStableWasm ? testConfig.stableInlineAssets : testConfig.inlineAssets;
    const value = assets[assetPath];
    if (typeof value !== 'string') throw new Error(`No inline asset fixture for ${assetPath}.`);
    return {
      url: moduleDataUrl(`export default ${JSON.stringify(value)};`),
      shortCircuit: true
    };
  }

  if (context.parentURL === testConfig.upstreamSourceUrl && specifier.endsWith('source-shard-map.json?raw')) {
    const pathUrl = new URL(specifier.slice(0, specifier.indexOf('?')), context.parentURL);
    const sourceMapText = await readFile(pathUrl, 'utf8');
    return {
      url: moduleDataUrl(`export default ${JSON.stringify(sourceMapText)};`),
      shortCircuit: true
    };
  }

  return nextResolve(specifier, context);
}
