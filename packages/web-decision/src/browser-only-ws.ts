// TVM's Node-only RPC fallback is not part of this browser-only package.
export default class UnsupportedNodeWebSocket {
  constructor() {
    throw new Error('Node.js WebSocket fallback is not supported.');
  }
}
