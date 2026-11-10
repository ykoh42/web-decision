import {listModels, loadDecisionModel, type LoadedDecisionModel} from 'web-decision';
import {loadOptions} from './model-options.ts';

function element<T extends Element>(selector: string, kind: {new (): T}): T {
  const found = document.querySelector(selector);
  if (!(found instanceof kind)) throw new Error(`Missing element: ${selector}`);
  return found;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

const load = element('#load', HTMLButtonElement);
const unload = element('#unload', HTMLButtonElement);
const form = element('#form', HTMLFormElement);
const decide = element('#decide', HTMLButtonElement);
const status = element('#status', HTMLElement);
const result = element('#result', HTMLElement);
const selector = element('#model', HTMLSelectElement);
const kananaAttribution = element('#kanana-attribution', HTMLParagraphElement);
const models = listModels();

for (const {modelId} of models) {
  const option = document.createElement('option');
  option.value = modelId;
  option.textContent = modelId;
  selector.append(option);
}
function updateKananaAttribution() {
  kananaAttribution.hidden = selector.value !== 'kanana-2-v-decide-3b-2611';
}
selector.addEventListener('change', updateKananaAttribution);
updateKananaAttribution();
if (models.length === 0) {
  load.disabled = true;
  status.textContent = 'No models are available.';
}

let model: LoadedDecisionModel | undefined;
let sessionBlocked = false;

async function runDecision(model: LoadedDecisionModel) {
  const state = element('#state', HTMLTextAreaElement).value;
  const instructions = element('#question', HTMLInputElement).value;
  const options = element('#options', HTMLTextAreaElement).value.split('\n');
  if (options.some((option) => !option.trim())) throw new Error('Each option line must contain text.');

  if (model.modelId === 'kanana-2-v-decide-3b-2611') {
    const criteria = Object.fromEntries(options.map((option, index) => [`option-${index}`, option]));
    const answer = await model.decide({
      state,
      questions: {decision: {type: 'choice', instructions, criteria}}
    });
    const selected = answer.answers.decision;
    if (!selected || selected.type !== 'choice') throw new Error('The model returned no choice answer.');
    const selectedIndex = Number(selected.choice.slice('option-'.length));
    const selectedOption = options[selectedIndex];
    if (selectedOption === undefined) throw new Error('The model returned an unknown option.');
    return {selected: selectedOption, answer};
  }

  const answer = await model.decide({
    state,
    questions: [{instructions, options}]
  });
  const selected = answer.answers[0];
  if (!selected) throw new Error('The model returned no answer.');
  return {selected: selected.selected, answer};
}

load.addEventListener('click', async () => {
  load.disabled = true;
  selector.disabled = true;
  status.textContent = 'Loading model...';
  let loading = true;
  try {
    const options = loadOptions(models, selector.value, ({fraction}) => {
      if (loading) status.textContent = `Loading: ${Math.round(fraction * 100)}%`;
    });
    model = await loadDecisionModel(options);
    status.textContent = `${selector.value} ready.`;
    decide.disabled = false;
    unload.disabled = false;
  } catch (error) {
    status.textContent = `Could not load the model: ${message(error)}. Refresh the page before retrying.`;
    sessionBlocked = true;
  } finally {
    loading = false;
  }
});

unload.addEventListener('click', async () => {
  if (!model) return;
  const current = model;
  model = undefined;
  unload.disabled = true;
  decide.disabled = true;
  result.textContent = '';
  status.textContent = 'Unloading model...';
  try {
    await current.dispose();
    status.textContent = 'Model unloaded.';
    load.disabled = false;
    selector.disabled = false;
  } catch (error) {
    model = current;
    sessionBlocked = true;
    status.textContent = `Could not unload the model: ${message(error)}. Refresh the page before trying again.`;
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!model || sessionBlocked) return;
  decide.disabled = true;
  unload.disabled = true;
  result.textContent = '';
  try {
    const answer = await runDecision(model);
    status.textContent = `Selected: ${answer.selected}`;
    result.textContent = JSON.stringify(answer.answer, null, 2);
  } catch (error) {
    status.textContent = `Could not make a decision: ${message(error)}`;
  } finally {
    decide.disabled = false;
    unload.disabled = false;
  }
});
