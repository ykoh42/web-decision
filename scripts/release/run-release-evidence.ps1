param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('verify', 'verify-kanana', 'test')]
  [string]$Mode,

  [string]$Archive,

  [string]$InstalledPackage,

  [string]$PlaywrightReport,

  [string]$BuildEvidence
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$buildDirectory = Join-Path $env:TEMP "web-decision-release-evidence-$PID"
$previousWorkspaceRoot = $env:WEB_DECISION_WORKSPACE_ROOT

New-Item -ItemType Directory -Path $buildDirectory | Out-Null
$exitCode = 0
try {
  $compiler = Join-Path $workspaceRoot 'node_modules/typescript/bin/tsc'
  $config = Join-Path $workspaceRoot 'scripts/release/tsconfig.runtime.json'
  & node.exe $compiler --project $config --outDir $buildDirectory
  $exitCode = $LASTEXITCODE
  if ($exitCode -eq 0) {
    $env:WEB_DECISION_WORKSPACE_ROOT = $workspaceRoot
    $releaseDirectory = Join-Path $buildDirectory 'scripts/release'
    if ($Mode -eq 'test') {
      & node.exe --test (Join-Path $releaseDirectory 'verify-release-evidence.test.js') (Join-Path $releaseDirectory 'tvm-build-evidence.test.js')
    } elseif ($Mode -eq 'verify-kanana') {
      if (-not $Archive -or -not $InstalledPackage -or -not $PlaywrightReport -or -not $BuildEvidence) {
        throw 'Kanana verify mode requires Archive, InstalledPackage, PlaywrightReport, and BuildEvidence.'
      }
      & node.exe (Join-Path $releaseDirectory 'verify-release-evidence.js') `
        --kanana `
        --archive $Archive `
        --installed-package $InstalledPackage `
        --playwright-report $PlaywrightReport `
        --build-evidence $BuildEvidence
    } else {
      if (-not $Archive -or -not $InstalledPackage -or -not $PlaywrightReport -or -not $BuildEvidence) {
        throw 'Verify mode requires Archive, InstalledPackage, PlaywrightReport, and BuildEvidence.'
      }
      & node.exe (Join-Path $releaseDirectory 'verify-release-evidence.js') `
        --archive $Archive `
        --installed-package $InstalledPackage `
        --playwright-report $PlaywrightReport `
        --build-evidence $BuildEvidence
    }
    $exitCode = $LASTEXITCODE
  }
} catch {
  Write-Host $_ -ForegroundColor Red
  $exitCode = 1
} finally {
  Remove-Item -LiteralPath $buildDirectory -Recurse -Force -ErrorAction SilentlyContinue
  if ($null -eq $previousWorkspaceRoot) {
    Remove-Item Env:WEB_DECISION_WORKSPACE_ROOT -ErrorAction SilentlyContinue
  } else {
    $env:WEB_DECISION_WORKSPACE_ROOT = $previousWorkspaceRoot
  }
}
exit $exitCode
