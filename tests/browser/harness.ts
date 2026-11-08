import * as webDecision from 'web-decision';

declare global {
  interface Window {
    webDecision: typeof webDecision;
  }
}

window.webDecision = webDecision;
