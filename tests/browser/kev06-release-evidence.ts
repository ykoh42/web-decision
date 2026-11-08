import {execFileSync} from 'node:child_process';
import {arch, release, version} from 'node:os';
export interface ObservedProbability {
  option: string;
  logit: number;
  probability: number;
}

export interface ObservedInference {
  modelId: string;
  backend: string;
  assetManifestSha256: string;
  selected: string;
  probabilities: readonly ObservedProbability[];
}

export interface ExpectedInference {
  selected?: string;
  optionOrder?: readonly string[];
  logits: readonly number[];
  probabilities: readonly number[];
  maxAbsoluteLogitDifference: number;
  maxAbsoluteProbabilityDifference: number;
}

interface WmiComputerSystem {
  Manufacturer?: unknown;
  Model?: unknown;
  HypervisorPresent?: unknown;
}

interface WmiVideoController {
  AdapterCompatibility?: unknown;
  Name?: unknown;
  DriverVersion?: unknown;
  PNPDeviceID?: unknown;
}

interface WmiSnapshot {
  computerSystems?: unknown;
  videoControllers?: unknown;
}

interface NodeOsMetadata {
  version: string;
  release: string;
  architecture: string;
}

export interface HardwareEvidence {
  device: {
    manufacturer: string;
    model: string;
    os: string;
    osVersion: string;
    architecture: string;
    isVirtual: false;
    evidence: {
      host: {
        source: 'windows-wmi/Win32_ComputerSystem';
        raw: {Manufacturer: string; Model: string; HypervisorPresent: boolean};
      };
      os: {
        source: 'node:os';
        raw: NodeOsMetadata;
      };
    };
  };
  gpu: {
    hardware: {
      manufacturer: string;
      model: string;
      driverVersion: string;
      pnpDeviceId: string;
    };
    hardwareEvidence: {
      source: 'windows-wmi/Win32_VideoController';
      raw: {
        AdapterCompatibility: string;
        Name: string;
        DriverVersion: string;
        PNPDeviceID: string;
      };
    };
  };
}

export interface BaselineFixtureEvidence {
  scope: string;
  packageSha256: string;
  fixtureSha256: string;
}

export interface UpstreamFixtureEvidence {
  scope: string;
  fixtureSha256: string;
  modelId: string;
  graphSha256: string;
  weightsSha256: string;
}

export function matchesInferenceReference(actual: ObservedInference, expected: ExpectedInference): boolean {
  if (expected.selected !== undefined && actual.selected !== expected.selected) return false;
  if (
    expected.optionOrder !== undefined &&
    !sameStrings(
      actual.probabilities.map(({option}) => option),
      expected.optionOrder
    )
  ) {
    return false;
  }

  const logits = actual.probabilities.map(({logit}) => logit);
  const probabilities = actual.probabilities.map(({probability}) => probability);
  return (
    withinTolerance(logits, expected.logits, expected.maxAbsoluteLogitDifference) &&
    withinTolerance(probabilities, expected.probabilities, expected.maxAbsoluteProbabilityDifference)
  );
}

export function matchesOfflineToOnline(online: ObservedInference, offline: ObservedInference): boolean {
  return (
    online.modelId === offline.modelId &&
    online.backend === offline.backend &&
    online.assetManifestSha256 === offline.assetManifestSha256 &&
    online.selected === offline.selected &&
    sameStrings(
      online.probabilities.map(({option}) => option),
      offline.probabilities.map(({option}) => option)
    ) &&
    sameNumbers(
      online.probabilities.map(({logit}) => logit),
      offline.probabilities.map(({logit}) => logit)
    ) &&
    sameNumbers(
      online.probabilities.map(({probability}) => probability),
      offline.probabilities.map(({probability}) => probability)
    )
  );
}

export function createInferenceEvidence(
  online: ObservedInference,
  offline: ObservedInference,
  offlineNetworkAttempts: readonly string[],
  baseline: BaselineFixtureEvidence,
  upstream: UpstreamFixtureEvidence,
  baselineExpected: ExpectedInference,
  upstreamExpected: ExpectedInference
) {
  return {
    online: {
      completed: true,
      ...toInferenceOutput(online),
      referenceMatched: matchesInferenceReference(online, baselineExpected)
    },
    offline: {
      completed: true,
      ...toInferenceOutput(offline),
      matchesOnline: matchesOfflineToOnline(online, offline),
      networkAttempts: [...offlineNetworkAttempts]
    },
    reference: baseline,
    upstreamReference: {
      ...upstream,
      matched: matchesInferenceReference(online, upstreamExpected)
    }
  };
}

export function parseWmiHardwareEvidence(
  input: unknown,
  osMetadata: NodeOsMetadata = {version: version(), release: release(), architecture: arch()}
): HardwareEvidence {
  const snapshot = asRecord(input, 'Windows WMI output') as WmiSnapshot;
  const host = onlyWmiRecord(snapshot.computerSystems, 'Win32_ComputerSystem') as WmiComputerSystem;
  const controllers = asArray(snapshot.videoControllers, 'Win32_VideoController');
  if (controllers.length !== 1) {
    throw new Error(`Expected exactly one visible Win32_VideoController; found ${controllers.length}.`);
  }
  const gpu = asRecord(controllers[0], 'Win32_VideoController') as WmiVideoController;
  const manufacturer = requiredText(host.Manufacturer, 'Win32_ComputerSystem.Manufacturer');
  const model = requiredText(host.Model, 'Win32_ComputerSystem.Model');
  if (typeof host.HypervisorPresent !== 'boolean') {
    throw new Error('Win32_ComputerSystem.HypervisorPresent is missing from measured system data.');
  }
  const gpuManufacturer = requiredText(gpu.AdapterCompatibility, 'Win32_VideoController.AdapterCompatibility');
  const gpuModel = requiredText(gpu.Name, 'Win32_VideoController.Name');
  const driverVersion = requiredText(gpu.DriverVersion, 'Win32_VideoController.DriverVersion');
  const pnpDeviceId = requiredText(gpu.PNPDeviceID, 'Win32_VideoController.PNPDeviceID');
  const os = {
    version: requiredText(osMetadata.version, 'node:os version'),
    release: requiredText(osMetadata.release, 'node:os release'),
    architecture: requiredText(osMetadata.architecture, 'node:os architecture')
  };

  if (looksVirtual(`${manufacturer} ${model} ${gpuManufacturer} ${gpuModel} ${pnpDeviceId}`)) {
    throw new Error('WMI hardware identifiers indicate a virtual host or display adapter.');
  }

  return {
    device: {
      manufacturer,
      model,
      os: os.version,
      osVersion: os.release,
      architecture: os.architecture,
      isVirtual: false,
      evidence: {
        host: {
          source: 'windows-wmi/Win32_ComputerSystem',
          raw: {Manufacturer: manufacturer, Model: model, HypervisorPresent: host.HypervisorPresent}
        },
        os: {source: 'node:os', raw: os}
      }
    },
    gpu: {
      hardware: {
        manufacturer: gpuManufacturer,
        model: gpuModel,
        driverVersion,
        pnpDeviceId
      },
      hardwareEvidence: {
        source: 'windows-wmi/Win32_VideoController',
        raw: {
          AdapterCompatibility: gpuManufacturer,
          Name: gpuModel,
          DriverVersion: driverVersion,
          PNPDeviceID: pnpDeviceId
        }
      }
    }
  };
}

export function collectWindowsHardwareEvidence(): HardwareEvidence {
  if (process.platform !== 'win32') {
    throw new Error('Release hardware evidence requires Windows WMI measurements.');
  }
  const output = execFileSync('powershell.exe', ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', WMI_QUERY], {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024,
    windowsHide: true
  }).replace(/^\uFEFF/u, '');
  return parseWmiHardwareEvidence(JSON.parse(output) as unknown);
}

function toInferenceOutput(inference: ObservedInference) {
  return {
    modelId: inference.modelId,
    backend: inference.backend,
    assetManifestSha256: inference.assetManifestSha256,
    selected: inference.selected,
    optionOrder: inference.probabilities.map(({option}) => option),
    logits: inference.probabilities.map(({logit}) => logit),
    probabilities: inference.probabilities.map(({probability}) => probability)
  };
}

function withinTolerance(actual: readonly number[], expected: readonly number[], tolerance: number): boolean {
  return (
    actual.length === expected.length &&
    actual.every((value, index) => {
      const expectedValue = expected[index];
      return expectedValue !== undefined && Number.isFinite(value) && Math.abs(value - expectedValue) <= tolerance;
    })
  );
}

function sameNumbers(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function asRecord(value: unknown, description: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${description} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function asArray(value: unknown, description: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${description} must be an array.`);
  return value;
}

function onlyWmiRecord(value: unknown, description: string): unknown {
  const records = asArray(value, description);
  if (records.length !== 1) throw new Error(`Expected exactly one ${description}; found ${records.length}.`);
  return records[0];
}

function requiredText(value: unknown, description: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${description} is missing from measured system data.`);
  }
  return value;
}

function looksVirtual(value: string): boolean {
  return /virtual machine|vmware|virtualbox|qemu|\bkvm\b|\bxen\b|hyper-v|parallels|bhyve|microsoft basic render driver|microsoft remote display adapter|virtio/iu.test(
    value
  );
}

const WMI_QUERY = String.raw`
$ErrorActionPreference = 'Stop'
try {
  $computerSystems = @(Get-CimInstance -ClassName Win32_ComputerSystem | Select-Object Manufacturer, Model, HypervisorPresent)
  $videoControllers = @(Get-CimInstance -ClassName Win32_VideoController | Select-Object AdapterCompatibility, Name, DriverVersion, PNPDeviceID)
  [PSCustomObject]@{ computerSystems = $computerSystems; videoControllers = $videoControllers } | ConvertTo-Json -Depth 4 -Compress
} catch {
  [Console]::Error.WriteLine($_.Exception.Message)
  exit 1
}
`;
