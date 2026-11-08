import {WebDecisionError} from '../kev.js';

const TVM_POOL_PAGE_BYTES = 4096;
const BOOL_I32_PAGE_BYTES = TVM_POOL_PAGE_BYTES * 2;
const I8_ARRAY = /\barray\s*<\s*i8\s*>/g;
const BOOL_STORE = /(\]\s*=\s*)bool\(([^;]*?)\);/g;
const stripComments = (code) => code.replace(/\/\*[\s\S]*?\*\/|\/\/[^\r\n]*/g, '');

function compatibilityError(message, cause) {
  return new WebDecisionError('UNSUPPORTED_ENVIRONMENT', message, cause ? {cause} : undefined);
}

export function rewriteOpenDeciderNanoShader(code) {
  if (typeof code !== 'string') throw new TypeError('A WebGPU shader source string is required.');
  const uncommented = stripComments(code);
  if (!/\bi8\b/.test(uncommented)) return {code, changed: false, arrayCount: 0, storeCount: 0};

  const arrays = uncommented.match(I8_ARRAY) || [];
  const unsupportedTypes = uncommented.replace(I8_ARRAY, '').match(/\bi8\b/);
  if (arrays.length === 0 || unsupportedTypes)
    throw compatibilityError('OpenDecider-nano encountered an unsupported i8 shader pattern.');

  const rewrittenArrays = code.replace(I8_ARRAY, 'array<i32>');
  let storeCount = 0;
  const rewritten = rewrittenArrays.replace(BOOL_STORE, (_, target, value) => {
    storeCount++;
    return `${target}i32(bool(${value}));`;
  });
  return {code: rewritten, changed: true, arrayCount: arrays.length, storeCount};
}

function installMethod(target, name, wrap) {
  const previous = Object.getOwnPropertyDescriptor(target, name);
  if (previous && !previous.configurable)
    throw compatibilityError(`Cannot install OpenDecider-nano TVM compatibility for ${name}.`);
  const original = target[name];
  if (typeof original !== 'function')
    throw compatibilityError(`OpenDecider-nano requires the pinned TVM ${name} method.`);
  const replacement = wrap(original);
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
    if (target[name] !== replacement) return;
    if (previous) Object.defineProperty(target, name, previous);
    else delete target[name];
  };
}

function wrapShader(report, original) {
  return function (info, code, ...args) {
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
export function installOpenDeciderNanoGpuCompatibility(context) {
  if (!context || typeof context !== 'object')
    throw compatibilityError('OpenDecider-nano needs the pinned TVM WebGPU context.');
  const report = {shaderModules: 0, i8ArrayDeclarations: 0, boolStores: 0, expandedStorageAllocations: 0};
  const restores = [];
  try {
    restores.push(installMethod(context, 'createShader', (original) => wrapShader(report, original)));
    restores.push(installMethod(context, 'createShaderAsync', (original) => wrapShader(report, original)));
    restores.push(
      installMethod(
        context,
        'deviceAllocDataSpace',
        (original) =>
          function (nbytes, ...args) {
            if (Number.isSafeInteger(nbytes) && nbytes >= TVM_POOL_PAGE_BYTES && nbytes < BOOL_I32_PAGE_BYTES) {
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
