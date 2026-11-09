import assert from 'node:assert/strict';
import test from 'node:test';
import {createFixtureImajevSession} from '../src/imajev/fixture-session.mjs';
import {createImajevResultProcessor} from '../src/imajev/result-processor.mjs';
import {safeImajevAnswer} from '../src/imajev/safe-outcome.mjs';

test('fixture session preserves the destructuring TypeError for undefined options', () => {
  assert.throws(() => Reflect.apply(createFixtureImajevSession, undefined, [undefined]), TypeError);
});

test('Imajev manifest and calibration getters preserve validation order and thrown errors', () => {
  const accesses: string[] = [];
  const codes = Array.from({length: 256}, (_, index) => ({code: `C${index}`, token_id: index}));
  const manifest = {
    get version(): number {
      accesses.push('manifest.version');
      return 1;
    },
    get codes(): typeof codes {
      accesses.push('manifest.codes');
      return codes;
    }
  };
  const calibration = {
    get schema_version(): string {
      accesses.push('calibration.schema_version');
      return '1.0';
    },
    get calibration_version(): string {
      accesses.push('calibration.calibration_version');
      return 'trace';
    },
    get unknown_offsets(): undefined {
      accesses.push('calibration.unknown_offsets');
      return undefined;
    },
    get temperatures(): Record<string, number> {
      accesses.push('calibration.temperatures');
      return {};
    }
  };
  const tokenizer = {encode: () => ({ids: [1]})};
  createImajevResultProcessor({tokenizer, manifest, calibration, maxInputTokens: 8});
  assert.deepEqual(accesses, [
    'manifest.version',
    'manifest.codes',
    'manifest.codes',
    'manifest.codes',
    'calibration.schema_version',
    'calibration.calibration_version',
    'calibration.calibration_version',
    'calibration.unknown_offsets',
    'calibration.temperatures',
    'calibration.temperatures',
    'calibration.temperatures',
    'calibration.temperatures',
    'calibration.calibration_version'
  ]);

  const marker = new Error('metadata getter');
  assert.throws(
    () =>
      createImajevResultProcessor({
        tokenizer,
        maxInputTokens: 8,
        manifest: {
          get version() {
            throw marker;
          }
        }
      }),
    (error: unknown) => error === marker
  );
  assert.throws(
    () =>
      createImajevResultProcessor({
        tokenizer,
        maxInputTokens: 8,
        manifest: {version: 1, codes},
        calibration: {
          get schema_version() {
            throw marker;
          }
        }
      }),
    (error: unknown) => error === marker
  );
});

test('Imajev answer getters preserve access order and thrown error identity', () => {
  const accesses: string[] = [];
  const probabilities = [0, 1].map((index) => ({
    get option(): string {
      accesses.push(`p${index}.option`);
      return index ? 'B' : 'A';
    },
    get probability(): number {
      accesses.push(`p${index}.probability`);
      return index ? 0.4 : 0.6;
    },
    get logit(): number {
      accesses.push(`p${index}.logit`);
      return index ? 1 : 2;
    }
  }));
  const answer = {
    get abstained(): boolean {
      accesses.push('answer.abstained');
      return false;
    },
    get unknownProbability(): number {
      accesses.push('answer.unknownProbability');
      return 0.2;
    },
    get probabilities(): typeof probabilities {
      accesses.push('answer.probabilities');
      return probabilities;
    },
    get selectedIndex(): number {
      accesses.push('answer.selectedIndex');
      return 0;
    },
    get selected(): string {
      accesses.push('answer.selected');
      return 'A';
    }
  };
  safeImajevAnswer(answer);
  assert.deepEqual(accesses, [
    'answer.abstained',
    'answer.unknownProbability',
    'answer.unknownProbability',
    'answer.unknownProbability',
    'answer.probabilities',
    'answer.probabilities',
    'answer.probabilities',
    'answer.probabilities',
    'p0.option',
    'p0.probability',
    'p0.probability',
    'p0.probability',
    'p0.logit',
    'p0.option',
    'p0.probability',
    'p0.logit',
    'p1.option',
    'p1.probability',
    'p1.probability',
    'p1.probability',
    'p1.logit',
    'p1.option',
    'p1.probability',
    'p1.logit',
    'answer.selectedIndex',
    'answer.selected',
    'answer.unknownProbability',
    'answer.abstained',
    'answer.selected'
  ]);

  const marker = new Error('answer getter');
  assert.throws(
    () =>
      safeImajevAnswer({
        get abstained() {
          throw marker;
        }
      }),
    (error: unknown) => error === marker
  );
});
