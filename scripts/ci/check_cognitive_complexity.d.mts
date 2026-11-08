export interface GeneratedExclusion {
  readonly path: string;
  readonly reason: string;
}

export interface ComplexityFinding {
  readonly file: string;
  readonly identity: string;
  readonly score: number;
}

export interface ComplexityBaseline {
  readonly schemaVersion: 1;
  readonly threshold: 15;
  readonly versions: {
    readonly oxlint: string;
    readonly sonarjs: string;
    readonly typescriptParser: string;
  };
  readonly generatedExclusions: readonly GeneratedExclusion[];
  readonly findings: readonly ComplexityFinding[];
}

export interface OxlintDiagnostic {
  readonly message: string;
  readonly filename: string;
  readonly labels?: readonly {
    readonly span?: {
      readonly offset?: number;
      readonly line?: number;
      readonly column?: number;
    };
  }[];
}

export interface ComplexitySuppression {
  readonly offset: number;
  readonly directive: string;
}

export declare const GENERATED_EXCLUSIONS: readonly GeneratedExclusion[];
export declare const ROOT: string;
export declare const TYPESCRIPT_PARSER_VERSION: string;

export declare function selectTrackedSources(
  trackedNames: readonly string[],
  exclusions?: readonly GeneratedExclusion[]
): string[];
export declare function sourceNamesInWorkingTree(root?: string): string[];
export declare function findComplexitySuppressions(source: string): ComplexitySuppression[];
export declare function diagnosticIdentity(
  diagnostic: OxlintDiagnostic,
  source: string,
  fileName: string
): ComplexityFinding;
export declare function runOxlint(files: readonly string[]): OxlintDiagnostic[];
export declare function validateFindings(findings: readonly ComplexityFinding[], label?: string): Map<string, number>;
export declare function compareFindingsToTrusted(
  currentFindings: readonly ComplexityFinding[],
  trustedFindings: readonly ComplexityFinding[]
): void;
export declare function checkBaseline(candidate: ComplexityBaseline, trusted?: ComplexityBaseline | null): true;
