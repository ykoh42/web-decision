import {readFile} from 'node:fs/promises';

interface LoaderTestConfig {
  bundledAssetsUrl: string;
  stableBundledAssetsUrl: string;
  upstreamSourceUrl: string;
  inlineAssets: Readonly<Record<string, string>>;
  stableInlineAssets: Readonly<Record<string, string>>;
}

interface ResolveContext {
  parentURL: string | undefined;
}

interface ResolveResult {
  url: string;
  shortCircuit?: boolean;
}

type NextResolve = (specifier: string, context?: Partial<ResolveContext>) => ResolveResult | Promise<ResolveResult>;

let testConfig: LoaderTestConfig;

export function initialize(data: unknown): void {
  testConfig = data as LoaderTestConfig;
}

function moduleDataUrl(source: string): string {
  return `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
}

export async function resolve(
  specifier: string,
  context: ResolveContext,
  nextResolve: NextResolve
): Promise<ResolveResult> {
  if (
    [testConfig.bundledAssetsUrl, testConfig.stableBundledAssetsUrl].includes(context.parentURL!) &&
    specifier.endsWith('?url&inline')
  ) {
    const assetUrl = new URL(specifier.slice(0, specifier.indexOf('?')), context.parentURL!);
    const isStableWasm =
      context.parentURL === testConfig.stableBundledAssetsUrl && assetUrl.pathname.includes('/kev06-stable/');
    const assetPath = assetUrl.pathname.split(isStableWasm ? '/kev06-stable/' : '/experimental-kev-upstream/').at(-1);
    const assets = isStableWasm ? testConfig.stableInlineAssets : testConfig.inlineAssets;
    const value = assets[assetPath!];
    if (typeof value !== 'string') throw new Error(`No inline asset fixture for ${assetPath}.`);
    return {
      url: moduleDataUrl(`export default ${JSON.stringify(value)};`),
      shortCircuit: true
    };
  }

  if (context.parentURL === testConfig.upstreamSourceUrl && specifier.endsWith('source-shard-map.json?raw')) {
    const pathUrl = new URL(specifier.slice(0, specifier.indexOf('?')), context.parentURL!);
    const sourceMapText = await readFile(pathUrl, 'utf8');
    return {
      url: moduleDataUrl(`export default ${JSON.stringify(sourceMapText)};`),
      shortCircuit: true
    };
  }

  return nextResolve(specifier, context);
}
