import {WebDecisionError} from '../kev.js';

const TVM_POOL_PAGE_BYTES = 4096;
const BOOL_I32_PAGE_BYTES = TVM_POOL_PAGE_BYTES * 2;
const I8_ARRAY = /\barray\s*<\s*i8\s*>/g;
const BOOL_STORE = /(\]\s*=\s*)bool\(([^;]*?)\);/g;
const stripComments = (code: string): string => code.replace(/\/\*[\s\S]*?\*\/|\/\/[^\r\n]*/g, '');

interface RewriteResult {
  readonly code: string;
  readonly changed: boolean;
  readonly arrayCount: number;
  readonly storeCount: number;
}

interface MutableCompatibilityReport {
  shaderModules: number;
  i8ArrayDeclarations: number;
  boolStores: number;
  expandedStorageAllocations: number;
}

interface CompatibilityReport extends Readonly<MutableCompatibilityReport> {}

interface Compatibility {
  readonly report: CompatibilityReport;
  restore(): void;
}

type RuntimeMethod = (this: unknown, ...args: unknown[]) => unknown;
type MethodWrapper = (original: RuntimeMethod) => RuntimeMethod;

function compatibilityError(message: string, cause?: unknown): WebDecisionError {
  return new WebDecisionError('UNSUPPORTED_ENVIRONMENT', message, cause ? {cause} : undefined);
}

export function rewriteOpenDeciderNanoShader(code: unknown): RewriteResult {
  if (typeof code !== 'string') throw new TypeError('A WebGPU shader source string is required.');
  const uncommented = stripComments(code);
  if (!/\bi8\b/.test(uncommented)) return {code, changed: false, arrayCount: 0, storeCount: 0};

  const arrays = uncommented.match(I8_ARRAY) || [];
  const unsupportedTypes = uncommented.replace(I8_ARRAY, '').match(/\bi8\b/);
  if (arrays.length === 0 || unsupportedTypes)
    throw compatibilityError('OpenDecider-nano encountered an unsupported i8 shader pattern.');

  const rewrittenArrays = code.replace(I8_ARRAY, 'array<i32>');
  let storeCount = 0;
  const rewritten = rewrittenArrays.replace(BOOL_STORE, (_match: string, target: string, value: string) => {
    storeCount++;
    return `${target}i32(bool(${value}));`;
  });
  return {code: rewritten, changed: true, arrayCount: arrays.length, storeCount};
}

function installMethod(target: object, name: string, wrap: MethodWrapper): () => void {
  const methodTarget = target as Record<string, unknown>;
  const previous = Object.getOwnPropertyDescriptor(target, name);
  if (previous && !previous.configurable)
    throw compatibilityError(`Cannot install OpenDecider-nano TVM compatibility for ${name}.`);
  const original = methodTarget[name];
  if (typeof original !== 'function')
    throw compatibilityError(`OpenDecider-nano requires the pinned TVM ${name} method.`);
  const replacement = wrap(original as RuntimeMethod);
  try {
    Object.defineProperty(target, name, {
      configurable: true,
      enumerable: previous?.enumerable ?? false,
      writable: true,
      value: replacement
    });
  } catch (cause) {
    throw compatibilityError(`Cannot install OpenDecider-nano TVM compatibility for ${name}.`, cause);
  }
  return () => {
    if (methodTarget[name] !== replacement) return;
    if (previous) Object.defineProperty(target, name, previous);
    else delete methodTarget[name];
  };
}

function wrapShader(report: MutableCompatibilityReport, original: RuntimeMethod): RuntimeMethod {
  return function (this: unknown, info: unknown, code: unknown, ...args: unknown[]) {
    if (typeof code !== 'string') return original.call(this, info, code, ...args);
    const result = rewriteOpenDeciderNanoShader(code);
    if (!result.changed) return original.call(this, info, code, ...args);
    report.shaderModules++;
    report.i8ArrayDeclarations += result.arrayCount;
    report.boolStores += result.storeCount;
    return original.call(this, info, result.code, ...args);
  };
}

/** Keep the compatibility changes on the pinned TVM runtime context only. */
export function installOpenDeciderNanoGpuCompatibility(context: unknown): Compatibility {
  if (!context || typeof context !== 'object')
    throw compatibilityError('OpenDecider-nano needs the pinned TVM WebGPU context.');
  const runtimeContext = context as object;
  const report: MutableCompatibilityReport = {
    shaderModules: 0,
    i8ArrayDeclarations: 0,
    boolStores: 0,
    expandedStorageAllocations: 0
  };
  const restores: Array<() => void> = [];
  try {
    restores.push(installMethod(runtimeContext, 'createShader', (original) => wrapShader(report, original)));
    restores.push(installMethod(runtimeContext, 'createShaderAsync', (original) => wrapShader(report, original)));
    restores.push(
      installMethod(
        runtimeContext,
        'deviceAllocDataSpace',
        (original) =>
          function (this: unknown, nbytes: unknown, ...args: unknown[]) {
            if (
              Number.isSafeInteger(nbytes as number) &&
              (nbytes as number) >= TVM_POOL_PAGE_BYTES &&
              (nbytes as number) < BOOL_I32_PAGE_BYTES
            ) {
              report.expandedStorageAllocations++;
              return original.call(this, BOOL_I32_PAGE_BYTES, ...args);
            }
            return original.call(this, nbytes, ...args);
          }
      )
    );
  } catch (cause) {
    for (const restore of restores.reverse()) restore();
    throw cause;
  }

  let restored = false;
  return {
    report,
    restore() {
      if (restored) return;
      restored = true;
      for (const restore of restores.reverse()) restore();
    }
  };
}
