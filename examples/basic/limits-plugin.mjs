import {choices} from './model-options.mjs';

// Keep the browser library lazy: extract only the installed package's limits
// when the example server or build starts.
export function limitsPlugin(describeModel) {
  const id = 'virtual:web-decision-token-limits';
  return {
    name: 'web-decision-token-limits',
    resolveId(source) {
      if (source === id) return `\0${id}`;
    },
    load(source) {
      if (source !== `\0${id}`) return;
      const limits = Object.fromEntries(
        choices.map(({model}) => [
          model,
          describeModel(model).features.textChoice.limits.filter((limit) => limit.resource === 'tokens')
        ])
      );
      return `export default ${JSON.stringify(limits)};`;
    }
  };
}
