export { LabShell, RailSection, type Insight, type LabShellProps } from './LabShell';
export {
  ProblemPicker,
  MethodSlots,
  StartPointFields,
  MethodCard,
  RunSummary,
  RunBadge,
  StartPointHint,
  StepBlock,
  FocusPicker,
  KAxisControl,
  type FocusRun,
  type MethodCardProps,
  type MethodCardQuantity,
  type MethodSlotsProps,
  type ProblemOption,
  type RunSummaryItem,
  type RunSummaryProps,
} from './blocks';
export { liveSpans } from './liveGrid';
export { SeedControl, MAX_SEED } from './SeedControl';
export {
  useLabRuns,
  useLabRunsState,
  useMethodSelection,
  selectionCodec,
  focusRuns,
  type LabRun,
  type LabRunsOptions,
} from './useLabRuns';
export { firstFreeSlot, sanitizeSelection, coerceParam, type MethodSelection } from './slots';
export {
  describeResult,
  divergenceReason,
  evidence,
  shortMethodName,
  type IterationNoun,
  type RunStatus,
  type StatusWording,
  type StatusWordingArg,
} from './status';
export {
  URL_KEYS,
  labQuery,
  labHref,
  tupleCodec,
  useProblemState,
  useStartPoint,
  applyPreset,
  presetActive,
  useKAxis,
  type KAxisMode,
  type LabState,
  type LabPreset,
} from './labState';
export { TryThis } from './presets';
export { pythonCall, pythonSnippet, pyFloat, type PythonCallOptions } from './python';
