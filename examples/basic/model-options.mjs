// Local example choices; each route still requires its own matching assets.
export const choices = Object.freeze(
  [
    {key: 'kev', label: 'Kev 0.6B Q4', model: 'kev-0.6b-q4f16', assets: '/kev-0.6b-q4f16/'},
    {
      key: 'k32',
      label: 'Kev 0.6B Q4 (experimental K32)',
      model: 'kev-0.6b-q4f16',
      assets: '/kev-0.6b-q4f16-k32-experimental/'
    },
    {
      key: 'reflexes',
      label: 'Kev 0.6B reflexes Q4',
      model: 'kev-0.6b-reflexes-q4f16',
      assets: '/kev-0.6b-reflexes-q4f16-preview/'
    },
    {
      key: 'deberta',
      label: 'DeBERTa-v3-large Q4',
      model: 'open-jev-deberta-v3-large-q4f16',
      assets: '/open-jev-deberta-v3-large-q4f16/'
    },
    {key: 'mapika-f16', label: 'Mapika 0.8B FP16', model: 'mapika-0.8b-f16', assets: ''},
    {key: 'mapika-q4', label: 'Mapika 0.8B Q4', model: 'mapika-0.8b-q4f16', assets: ''},
    {key: 'kev08', label: 'Kev 0.8B FP16', model: 'kev-0.8b-f16', assets: ''},
    {key: 'h2o', label: 'H2O Lightning 4B Q4 (experimental)', model: 'experimental-h2o-lightning-4b-q4f16', assets: ''},
    {
      key: 'h2o-dlight',
      label: 'H2O Lightning 4B Q4 DLight (experimental)',
      model: 'experimental-h2o-lightning-4b-q4f16-dlight',
      assets: ''
    },
    {
      key: 'h2o-dual',
      label: 'H2O Lightning 4B Q4 dual projection (experimental)',
      model: 'experimental-h2o-lightning-4b-q4f16-dual',
      assets: ''
    },
    {
      key: 'nari',
      label: 'Nari Decision 4B Q4 (experimental)',
      model: 'experimental-nari-decision-4b-q4f16',
      assets: ''
    },
    {
      key: 'nari-mixed',
      label: 'Nari Decision 4B mixed FP16/Q4 (experimental)',
      model: 'experimental-nari-decision-4b-mixed16-q4f16',
      assets: ''
    },
    {key: 'imajev', label: 'Imajev 4B Q4 (experimental)', model: 'experimental-imajev-4b-candidate', assets: ''},
    {
      key: 'pplx',
      label: 'Perplexity Decider Flash 4B Q4 DLight (experimental)',
      model: 'experimental-pplx-decider-flash-v1.2-q4f16-dlight',
      assets: ''
    },
    {key: 'mapika4b', label: 'Mapika 4B Q4 (experimental)', model: 'experimental-mapika-4b-q4f16', assets: ''},
    {
      key: 'mapika4b-dlight',
      label: 'Mapika 4B Q4 DLight (experimental)',
      model: 'experimental-mapika-4b-q4f16-dlight',
      assets: ''
    },
    {
      key: 'mapika4b-dlight-all',
      label: 'Mapika 4B Q4 DLight all (experimental)',
      model: 'experimental-mapika-4b-q4f16-dlight-all',
      assets: ''
    },
    {
      key: 'elephant',
      label: 'Elephant 9B Q4 (experimental)',
      model: 'experimental-elephant-9b-q4f16-dlight',
      assets: ''
    },
    {
      key: 'elephant-512',
      label: 'Elephant 9B Q4 512 tokens (experimental)',
      model: 'experimental-elephant-9b-q4f16-dlight-512',
      assets: ''
    },
    {
      key: 'clef',
      label: 'Clef 9B selected-row Q4 (experimental)',
      model: 'experimental-clef-9b-q4f16-selected',
      assets: ''
    },
    {key: 'strands', label: 'Strands 2B FP16 (experimental)', model: 'experimental-strands-qwen-2b-fp16', assets: ''},
    {
      key: 'opendecider-nano',
      label: 'OpenDecider-nano FP16 (experimental)',
      model: 'experimental-opendecider-nano-fp16',
      assets: ''
    },
    {
      key: 'd1-omni',
      label: 'd1-omni 600M Q8 text (experimental)',
      model: 'experimental-d1-omni-600m-q8',
      assets: '/d1-omni-600m-q8/'
    }
  ].map(Object.freeze)
);

export function findChoice(key) {
  return choices.find((choice) => choice.key === key);
}

export function isH2oChoice(key) {
  return key === 'h2o' || key === 'h2o-dlight' || key === 'h2o-dual';
}

export function isOnlineOnlyChoice(key) {
  return (
    isH2oChoice(key) ||
    key === 'nari' ||
    key === 'nari-mixed' ||
    key === 'imajev' ||
    key === 'pplx' ||
    key === 'mapika4b' ||
    key === 'mapika4b-dlight' ||
    key === 'mapika4b-dlight-all' ||
    key === 'elephant' ||
    key === 'elephant-512' ||
    key === 'clef' ||
    key === 'opendecider-nano'
  );
}

export function loadOptions(
  key,
  assetsUrl,
  {onProgress, packedReadback = false, baseUrl = globalThis.location?.href} = {}
) {
  const choice = findChoice(key);
  if (!choice) throw new Error('Select a supported model.');
  if (typeof assetsUrl !== 'string' || !assetsUrl.trim()) throw new Error('Enter the matching asset directory URL.');
  if (packedReadback && key !== 'deberta') throw new Error('Packed readback is available only for DeBERTa.');
  const trimmedUrl = assetsUrl.trim();
  let parsedUrl;
  try {
    parsedUrl = new URL(trimmedUrl, baseUrl);
  } catch {
    throw new Error('Enter a valid HTTP or HTTPS asset directory URL.');
  }
  if (
    !['http:', 'https:'].includes(parsedUrl.protocol) ||
    parsedUrl.username ||
    parsedUrl.password ||
    parsedUrl.search ||
    parsedUrl.hash
  )
    throw new Error('Asset URL must use HTTP or HTTPS without credentials, query, or fragment.');
  // Preserve relative paths while removing bare query and fragment delimiters.
  const normalizedUrl = trimmedUrl.replace(/[?#]+$/, '');
  if (!normalizedUrl) {
    parsedUrl.search = '';
    parsedUrl.hash = '';
  }
  return {
    model: choice.model,
    assetsUrl: normalizedUrl || parsedUrl.href,
    ...(isOnlineOnlyChoice(key) || key === 'strands' ? {cache: false} : {}),
    ...((!isOnlineOnlyChoice(key) || key === 'opendecider-nano') && onProgress ? {onProgress} : {}),
    ...(packedReadback ? {debertaReadback: 'packed'} : {})
  };
}
