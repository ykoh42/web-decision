import {assertCommittedPackageProvenance} from './model-provenance.ts';
import {fileURLToPath} from 'node:url';

try {
  const workspaceRoot = fileURLToPath(new URL('../../', import.meta.url));
  const result = await assertCommittedPackageProvenance(workspaceRoot);
  process.stdout.write(
    `Verified model provenance for ${result.modelIds.length} model(s); manifest ${result.manifestSha256}.\n`
  );
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Model provenance gate failed: ${message}\n`);
  process.exitCode = 1;
}
