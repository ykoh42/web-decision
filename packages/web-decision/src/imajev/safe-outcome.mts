// Standalone design prototype, not exported by the package.
// Input follows the isolated one-pass Imajev protocol's decoded result.
export interface ImajevAnswerProbability {
  readonly option: string;
  readonly probability: number;
  readonly logit: number;
}

export type ImajevAnswer = Readonly<{
  unknownProbability: number;
  probabilities: readonly ImajevAnswerProbability[];
}> &
  (
    | Readonly<{
        abstained: true;
        selected: null;
        selectedIndex: null;
        bestKnown: Readonly<{option: string; index: number}>;
      }>
    | Readonly<{
        abstained: false;
        selected: string;
        selectedIndex: number;
      }>
  );

interface ImajevAnswerCandidate {
  readonly abstained?: unknown;
  readonly unknownProbability?: unknown;
  readonly probabilities?: unknown;
  readonly selected?: unknown;
  readonly selectedIndex?: unknown;
}

interface ImajevProbabilityCandidate {
  readonly option?: unknown;
  readonly probability?: unknown;
  readonly logit?: unknown;
}

export function safeImajevAnswer(answer: unknown): ImajevAnswer {
  if (!answer || typeof answer !== 'object') throw new TypeError('Invalid Imajev answer');
  const candidate = answer as ImajevAnswerCandidate;
  if (
    typeof candidate.abstained !== 'boolean' ||
    !Number.isFinite(candidate.unknownProbability) ||
    (candidate.unknownProbability as number) < 0 ||
    (candidate.unknownProbability as number) > 1 ||
    !Array.isArray(candidate.probabilities) ||
    candidate.probabilities.length < 2 ||
    candidate.probabilities.length > 255
  )
    throw new TypeError('Invalid Imajev answer');
  const probabilities = (candidate.probabilities as unknown[]).map((rawEntry: unknown) => {
    const entry = rawEntry as ImajevProbabilityCandidate | null | undefined;
    if (
      !entry ||
      typeof entry.option !== 'string' ||
      !Number.isFinite(entry.probability) ||
      (entry.probability as number) < 0 ||
      (entry.probability as number) > 1 ||
      !Number.isFinite(entry.logit)
    )
      throw new TypeError('Invalid known-option score');
    return Object.freeze({
      option: entry.option,
      probability: entry.probability as number,
      logit: entry.logit as number
    });
  });
  const max = Math.max(...probabilities.map((entry) => entry.probability));
  const firstMaximum = probabilities.findIndex((entry) => entry.probability === max);
  const index = candidate.selectedIndex;
  if (
    !Number.isSafeInteger(index) ||
    index !== firstMaximum ||
    candidate.selected !== probabilities[index as number]?.option ||
    Math.abs(probabilities.reduce((sum, entry) => sum + entry.probability, 0) - 1) > 1e-9
  )
    throw new TypeError('Invalid best-known choice');
  const shared = {
    unknownProbability: candidate.unknownProbability as number,
    probabilities: Object.freeze(probabilities)
  };
  if (candidate.abstained)
    return Object.freeze({
      abstained: true,
      selected: null,
      selectedIndex: null,
      bestKnown: Object.freeze({option: candidate.selected as string, index: index as number}),
      ...shared
    });
  return Object.freeze({
    abstained: false,
    selected: candidate.selected as string,
    selectedIndex: index as number,
    ...shared
  });
}
