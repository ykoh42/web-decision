import {readFile} from 'node:fs/promises';
import {register, type LoadHook} from 'node:module';
import {pathToFileURL} from 'node:url';

let builtModuleUrl: string;

export function initialize(url: string): void {
  builtModuleUrl = url;
}

export const load: LoadHook = async (url, context, nextLoad) => {
  if (url !== builtModuleUrl) return nextLoad(url, context);
  return {format: 'module', shortCircuit: true, source: await readFile(new URL(url))};
};

export async function importBuiltModule(path: string): Promise<unknown> {
  const url = pathToFileURL(path).href;
  // Load emitted JavaScript directly to preserve its Vite source map.
  register(import.meta.url, {parentURL: import.meta.url, data: url});
  return import(url);
}
