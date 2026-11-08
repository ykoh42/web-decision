import {choices, findChoice, isH2oChoice, isOnlineOnlyChoice, loadOptions} from './model-options.mjs';
import tokenLimits from 'virtual:web-decision-token-limits';

const load = document.querySelector('#load');
const unload = document.querySelector('#unload');
const form = document.querySelector('#form');
const decide = document.querySelector('#decide');
const status = document.querySelector('#status');
const result = document.querySelector('#result');
const selector = document.querySelector('#model');
const assets = document.querySelector('#assets');
const help = document.querySelector('#asset-help');
const query = new URLSearchParams(location.search);
const requested = query.get('model') || 'kev';
const packedReadback = query.get('readback') === 'packed';
for (const choice of choices) {
  const option = document.createElement('option');
  option.value = choice.key;
  option.textContent = choice.label;
  selector.append(option);
}
function assetHelp(key, choice) {
  if (key === 'clef')
    return 'Clef requires separately prepared assets on an HTTP range server. Online only; no cache or progress. The complete request is limited to 512 tokens and this example accepts 2–64 distinct text options. Each decision reads selected table ranges; transfer size varies and the server can observe the requested ranges.';
  if (isH2oChoice(key))
    return 'H2O assets are not distributed; enter the URL of the separately prepared matching asset server. Caching and progress are not supported.';
  if (isOnlineOnlyChoice(key))
    return 'Model assets are not distributed; enter a matching HTTP range server URL. Caching and progress are not supported.';
  if (key === 'strands')
    return 'Strands assets are not distributed; enter a matching HTTP range server URL. This example disables cache.';
  return choice.assets
    ? 'This local URL requires separately prepared matching files in the example asset directory.'
    : 'Enter a URL serving the matching, separately prepared model files.';
}
function selectChoice(key) {
  const choice = findChoice(key);
  if (!choice) return false;
  selector.value = key;
  assets.value = choice.assets;
  const limits = tokenLimits[choice.model].map(
    ({scope, max, description}) => `${scope}: ${max} tokens. ${description}`
  );
  help.textContent = `${assetHelp(key, choice)} Token limits — ${limits.join(' ')}`;
  return true;
}
if (selectChoice(requested)) {
  if (query.has('assetsUrl')) assets.value = query.get('assetsUrl');
} else {
  selector.value = '';
  load.disabled = true;
  status.textContent = 'Unknown model. Select a supported model.';
}
selector.addEventListener('change', () => {
  if (selectChoice(selector.value)) {
    load.disabled = false;
    status.textContent = 'Load the model to begin.';
  }
});
let model;
let sessionBlocked = false;

load.addEventListener('click', async () => {
  load.disabled = true;
  selector.disabled = true;
  assets.disabled = true;
  status.textContent = 'Loading model...';
  let loaderStarted = false;
  let loading = true;
  try {
    const options = loadOptions(selector.value, assets.value, {
      packedReadback: packedReadback && selector.value === 'deberta',
      onProgress: ({fraction}) => {
        if (loading) status.textContent = `Loading: ${Math.round(fraction * 100)}%`;
      }
    });
    const {loadDecisionModel} = await import('web-decision');
    loaderStarted = true;
    model = await loadDecisionModel(options);
    status.textContent = `${findChoice(selector.value).label} ready.`;
    decide.disabled = false;
    unload.disabled = false;
  } catch (error) {
    const retainGuard = loaderStarted && (isOnlineOnlyChoice(selector.value) || selector.value === 'strands');
    status.textContent = `Could not load the model: ${error.message}${
      retainGuard
        ? ' GPU resources may remain. Refresh the page before another attempt.'
        : loaderStarted
          ? ' GPU resources may remain. Consider refreshing the page before retrying.'
          : ''
    }`;
    if (!retainGuard) {
      load.disabled = false;
      selector.disabled = false;
      assets.disabled = false;
    }
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
  let disposed = false;
  try {
    await current.dispose();
    disposed = true;
    status.textContent = 'Model unloaded.';
  } catch (error) {
    model = current;
    sessionBlocked = true;
    status.textContent = `Could not unload the model: ${error.message}. GPU resources may remain. Refresh the page before loading another model.`;
  } finally {
    if (disposed) {
      load.disabled = false;
      selector.disabled = false;
      assets.disabled = false;
    }
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!model || sessionBlocked) return;
  decide.disabled = true;
  unload.disabled = true;
  result.textContent = '';
  try {
    const options = document.querySelector('#options').value.split('\n');
    if (options.some((option) => !option.trim()))
      throw new Error('Each option line must contain text. Remove blank lines.');
    if (selector.value === 'imajev' && new Set(options).size !== options.length)
      throw new Error('Imajev requires distinct options. Remove duplicate lines.');
    const answer = await model.decide({
      state: document.querySelector('#state').value,
      questions: [
        {
          instructions: document.querySelector('#question').value,
          options
        }
      ]
    });
    status.textContent = answer.answers[0].abstained
      ? 'Abstained: no option selected.'
      : `Selected: ${answer.answers[0].selected}`;
    result.textContent = JSON.stringify(answer, null, 2);
  } catch (error) {
    status.textContent = `Could not make a decision: ${error.message}`;
  } finally {
    decide.disabled = false;
    unload.disabled = false;
  }
});
