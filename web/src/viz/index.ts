import { lazy } from 'react';

export { useCanvas, useElementSize, type CanvasSize } from './useCanvas';
export * from './scales';
export { drawAxes, type Frame } from './axes';
export { Plot1D, type Overlay1D, type Plot1DProps } from './Plot1D';
export { adaptiveSample } from './sampling';
export { Contour2D, type Domain2D, type View2D, type Contour2DProps } from './Contour2D';
export {
  drawOverlays2D,
  drawOverlayLabels,
  ellipsePoints,
  implicitSegments,
  type Overlay2D,
  type OverlayLabel,
  type OverlayView,
} from './overlays2d';
export { addLabelRect, hitsLabel, labelOverlap, type LabelRect } from './labelRects';
export {
  drawPathLayer,
  pathPosition,
  overlappingPaths,
  pathCoverage,
  milestoneIndices,
  clipSegment,
  type PathSpec,
  type PxBounds,
} from './PathLayer';
export {
  ConvergenceChart,
  type ConvergenceSeries,
  type ConvergenceChartProps,
  type SlopeGuide,
} from './ConvergenceChart';
export { IterationTable } from './IterationTable';
export { SeriesLegend, type SeriesLegendItem } from './SeriesLegend';
export { ViewToolbar, type ViewToolbarProps } from './ViewToolbar';
export { defaultColumns, type Column } from './columns';
export { MatrixView, type MatrixViewProps } from './MatrixView';
export { matrixColumnChars } from './matrixColumns';
export { TableView, type TableColumn, type TableViewProps } from './TableView';
export { TreeView, type TreeNode, type TreeNodeStatus, type TreeViewProps } from './TreeView';
export { layoutTree } from './treeLayout';
export { DataPlot, type DataPoint, type DataCurve, type DataPlotProps } from './DataPlot';
export { dataDomain, slopeTriangle, suggestLogK } from './chartMath';
export {
  drawMath,
  measureMath,
  pow10Runs,
  iterateRuns,
  starRuns,
  m as mathMain,
  v as mathVar,
  b as mathBold,
  sub as mathSub,
  sup as mathSup,
  sans as mathSans,
  type MathRun,
} from './mathText';
export type { Surface3DProps } from './Surface3D';

/** three.js surface, code-split: wrap in <Suspense>. */
export const LazySurface3D = lazy(() => import('./Surface3D'));
