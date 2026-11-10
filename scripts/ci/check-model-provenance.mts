import {assertCommittedPackageProvenance} from './model-provenance.ts';
import {fileURLToPath} from 'node:url';

try {
  const args = process.argv.slice(2);
  if (
    args.some((argument) => argument !== '--allow-pending') ||
    args.filter((argument) => argument === '--allow-pending').length > 1
  ) {
    throw new TypeError('The only supported option is --allow-pending.');
  }
  const workspaceRoot = fileURLToPath(new URL('../../', import.meta.url));
  const result = await assertCommittedPackageProvenance(workspaceRoot, {
    allowPending: args.includes('--allow-pending')
  });
  const pending =
    result.pendingModelIds.length > 0 ? ` Pending and unreleasable: ${result.pendingModelIds.join(', ')}.` : '';
  process.stdout.write(
    `Verified model provenance for ${result.modelIds.length} model(s); manifest ${result.manifestSha256}.${pending}\n`
  );
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Model provenance gate failed: ${message}\n`);
  process.exitCode = 1;
}
