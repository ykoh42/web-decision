import assert from 'node:assert/strict';
import test from 'node:test';
import {
  matchesInferenceReference,
  matchesOfflineToOnline,
  parseWmiHardwareEvidence,
  type ExpectedInference,
  type ObservedInference
} from './kev06-release-evidence.ts';

const baseline: ObservedInference = {
  modelId: 'kev-0.6b-q4f16',
  backend: 'tvm-webgpu',
  assetManifestSha256: 'a'.repeat(64),
  selected: 'billing',
  probabilities: [
    {option: 'billing', logit: 4.62, probability: 0.9955},
    {option: 'shipping', logit: -0.78, probability: 0.0045}
  ]
};

const baselineExpected: ExpectedInference = {
  selected: 'billing',
  optionOrder: ['billing', 'shipping'],
  logits: [4.621, -0.785],
  probabilities: [0.9955, 0.0045],
  maxAbsoluteLogitDifference: 0.01,
  maxAbsoluteProbabilityDifference: 0.001
};

const physicalWmiSnapshot = {
  computerSystems: [{Manufacturer: 'AZW', Model: 'MINI S', HypervisorPresent: false}],
  videoControllers: [
    {
      AdapterCompatibility: 'Intel Corporation',
      Name: 'Intel(R) UHD Graphics',
      DriverVersion: '32.0.101.6737',
      PNPDeviceID: 'PCI\\VEN_8086&DEV_46A8'
    }
  ]
};

test('reference comparisons use the captured output and preserve option order', () => {
  assert.equal(matchesInferenceReference(baseline, baselineExpected), true);
  assert.equal(matchesInferenceReference({...baseline, selected: 'shipping'}, baselineExpected), false);
  assert.equal(
    matchesInferenceReference(
      {
        ...baseline,
        probabilities: [...baseline.probabilities].reverse()
      },
      baselineExpected
    ),
    false
  );
  assert.equal(
    matchesInferenceReference(
      {
        ...baseline,
        probabilities: [{...baseline.probabilities[0]!, logit: Number.NaN}, baseline.probabilities[1]!]
      },
      baselineExpected
    ),
    false
  );
});

test('offline comparison requires exact IDs, selection, logits, and probabilities', () => {
  assert.equal(matchesOfflineToOnline(baseline, structuredClone(baseline)), true);
  assert.equal(
    matchesOfflineToOnline(baseline, {
      ...baseline,
      probabilities: [
        {...baseline.probabilities[0]!, logit: baseline.probabilities[0]!.logit + 0.000001},
        baseline.probabilities[1]!
      ]
    }),
    false
  );
  assert.equal(matchesOfflineToOnline(baseline, {...baseline, modelId: 'different-model'}), false);
});

test('WMI provenance stays separate from browser adapter information', () => {
  const evidence = parseWmiHardwareEvidence(physicalWmiSnapshot, {
    version: 'Windows 11 Pro',
    release: '10.0.26100',
    architecture: 'x64'
  });

  assert.deepEqual(evidence.device, {
    manufacturer: 'AZW',
    model: 'MINI S',
    os: 'Windows 11 Pro',
    osVersion: '10.0.26100',
    architecture: 'x64',
    isVirtual: false,
    evidence: {
      host: {
        source: 'windows-wmi/Win32_ComputerSystem',
        raw: {Manufacturer: 'AZW', Model: 'MINI S', HypervisorPresent: false}
      },
      os: {
        source: 'node:os',
        raw: {version: 'Windows 11 Pro', release: '10.0.26100', architecture: 'x64'}
      }
    }
  });
  assert.ok(evidence.gpu);
  assert.equal(evidence.gpu.hardware.model, 'Intel(R) UHD Graphics');
  assert.equal(evidence.gpu.hardwareEvidence.source, 'windows-wmi/Win32_VideoController');
  assert.equal('adapterInfo' in evidence.gpu, false);
  const hostWithHypervisor = parseWmiHardwareEvidence(
    {
      ...physicalWmiSnapshot,
      computerSystems: [{Manufacturer: 'AZW', Model: 'MINI S', HypervisorPresent: true}]
    },
    {
      version: 'Windows 11 Pro',
      release: '10.0.26100',
      architecture: 'x64'
    }
  );
  assert.equal(hostWithHypervisor.device.evidence.host.source, 'windows-wmi/Win32_ComputerSystem');
  assert.equal(hostWithHypervisor.device.evidence.host.raw.HypervisorPresent, true);
});

test('WMI evidence fails closed for known virtual signatures, missing, or ambiguous hardware', () => {
  assert.throws(
    () =>
      parseWmiHardwareEvidence({
        ...physicalWmiSnapshot,
        computerSystems: [{Manufacturer: 'Microsoft Corporation', Model: 'Virtual Machine', HypervisorPresent: false}]
      }),
    /virtual host/u
  );
  assert.throws(
    () =>
      parseWmiHardwareEvidence({
        ...physicalWmiSnapshot,
        computerSystems: [{Manufacturer: 'AZW', Model: 'MINI S'}]
      }),
    /HypervisorPresent is missing/u
  );
  assert.throws(
    () =>
      parseWmiHardwareEvidence({
        ...physicalWmiSnapshot,
        videoControllers: [...physicalWmiSnapshot.videoControllers, physicalWmiSnapshot.videoControllers[0]]
      }),
    /exactly one visible/u
  );
  assert.throws(
    () =>
      parseWmiHardwareEvidence({
        ...physicalWmiSnapshot,
        videoControllers: [{...physicalWmiSnapshot.videoControllers[0], Name: 'VMware SVGA 3D'}]
      }),
    /virtual host or display adapter/u
  );
  assert.throws(
    () =>
      parseWmiHardwareEvidence({
        ...physicalWmiSnapshot,
        videoControllers: [{...physicalWmiSnapshot.videoControllers[0], DriverVersion: ''}]
      }),
    /DriverVersion is missing/u
  );
});
