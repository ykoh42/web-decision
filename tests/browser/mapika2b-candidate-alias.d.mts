declare module 'mapika2b-qa-candidate' {
  export const loadMapika2bCandidate: typeof import('../../packages/web-decision/src/mapika2b/loader.mts').loadMapika2bCandidate;
  export const loadMapika2bCuratedForTest: typeof import('../../scripts/mapika/webgpu-candidate-entry.mts').loadMapika2bCuratedForTest;
  export const warmMapika2bPackagedCode: () => Promise<void>;
  export const mapika2bPackagedAsset: typeof import('../../scripts/mapika/webgpu-candidate-entry.mts').mapika2bPackagedAsset;
}
