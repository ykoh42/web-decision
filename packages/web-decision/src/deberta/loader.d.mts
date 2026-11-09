import type {DebertaLoadOptions} from '../deberta-single-preview.js';
import type {ExperimentalDebertaSession} from './model.mjs';

/**
 * Temporary type boundary for loader.mjs while it still imports pinned TVM TypeScript sources.
 * Remove this declaration when the maintained loader is migrated to loader.mts.
 */
export function loadExperimentalDeberta(options: DebertaLoadOptions): Promise<ExperimentalDebertaSession>;
