export const ELEPHANT_MODEL = 'experimental-elephant-9b-q4f16-dlight';
export const ELEPHANT_512_MODEL = 'experimental-elephant-9b-q4f16-dlight-512';
const ELEPHANT_GRAPH_SHA256 = 'f7a8bb9cc2e5fe70c07edb05e678a854ebf83e9c78dc8bb1559185472c643181';
const ELEPHANT_512_GRAPH_SHA256 = 'a7d9904be86cb1ab54a7778fe0c3315c90761e0f4760df1a04de83c88c1189c5';
export const ELEPHANT_INVENTORY_SHA256 = 'd68385516ef60ec2feafd5d7996534356ed7531de8491c45c2fd00073831366b';
const VARIANTS = Object.freeze({
  [ELEPHANT_MODEL]: Object.freeze({graphSha256: ELEPHANT_GRAPH_SHA256, graphBytes: 5093840, maxBranchTokens: 128}),
  [ELEPHANT_512_MODEL]: Object.freeze({
    graphSha256: ELEPHANT_512_GRAPH_SHA256,
    graphBytes: 5094092,
    maxBranchTokens: 512
  })
});

export function elephantVariant(model) {
  return typeof model === 'string' && Object.hasOwn(VARIANTS, model) ? VARIANTS[model] : undefined;
}
export const ELEPHANT_LIMITS = Object.freeze({
  maxBufferSize: 134217728,
  maxStorageBufferBindingSize: 134217728,
  maxStorageBuffersPerShaderStage: 8
});
