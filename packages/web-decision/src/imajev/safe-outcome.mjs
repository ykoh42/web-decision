// Standalone design prototype, not exported by the package.
// Input follows the isolated one-pass Imajev protocol's decoded result.
export function safeImajevAnswer(answer) {
  if (
    !answer ||
    typeof answer !== 'object' ||
    typeof answer.abstained !== 'boolean' ||
    !Number.isFinite(answer.unknownProbability) ||
    answer.unknownProbability < 0 ||
    answer.unknownProbability > 1 ||
    !Array.isArray(answer.probabilities) ||
    answer.probabilities.length < 2 ||
    answer.probabilities.length > 255
  )
    throw new TypeError('Invalid Imajev answer');
  const probabilities = answer.probabilities.map((entry) => {
    if (
      !entry ||
      typeof entry.option !== 'string' ||
      !Number.isFinite(entry.probability) ||
      entry.probability < 0 ||
      entry.probability > 1 ||
      !Number.isFinite(entry.logit)
    )
      throw new TypeError('Invalid known-option score');
    return Object.freeze({option: entry.option, probability: entry.probability, logit: entry.logit});
  });
  const max = Math.max(...probabilities.map((entry) => entry.probability));
  const firstMaximum = probabilities.findIndex((entry) => entry.probability === max);
  const index = answer.selectedIndex;
  if (
    !Number.isSafeInteger(index) ||
    index !== firstMaximum ||
    answer.selected !== probabilities[index]?.option ||
    Math.abs(probabilities.reduce((sum, entry) => sum + entry.probability, 0) - 1) > 1e-9
  )
    throw new TypeError('Invalid best-known choice');
  const shared = {unknownProbability: answer.unknownProbability, probabilities: Object.freeze(probabilities)};
  if (answer.abstained)
    return Object.freeze({
      abstained: true,
      selected: null,
      selectedIndex: null,
      bestKnown: Object.freeze({option: answer.selected, index}),
      ...shared
    });
  return Object.freeze({abstained: false, selected: answer.selected, selectedIndex: index, ...shared});
}
