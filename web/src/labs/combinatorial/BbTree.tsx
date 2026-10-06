/**
 * The depth-first branch-and-bound tree of the knapsack, revealed node by node. Depth d decides
 * item order[d]; the include child (xᵢ = 1, explored first) is on the left, the exclude child
 * (xᵢ = 0) on the right. Status is a shape and a word, never color alone: ● branched,
 * ○ with a bar = pruned (⌊U⌋ ≤ z), ■ leaf, a ring = new incumbent; the node at the playhead is
 * ringed in the accent color.
 */
import { useMemo, useRef, useState } from 'react';
import { layoutTree, useElementSize } from '../../viz';
import { sig } from '../../core/format';
import type { BbNode } from '../../methods/combinatorial/knapsack';
import { seriesVar } from '../../ui/colors';
import styles from './CombinatorialLab.module.css';

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (n: number) => [...String(n)].map((c) => SUBS[Number(c)]).join('');
const describe = (n: BbNode) =>
  `Node ${n.id}, depth ${n.depth}${n.item !== null ? `, x${sub(n.item)} = ${n.decision}` : ', root'}: V = ${n.value}, W = ${n.weight}, U = ${sig(n.bound, 5)}, ${n.status}${n.incumbent ? ', new incumbent' : ''}`;

export function BbTree({
  nodes,
  order,
  current,
  slot,
  zStar,
}: {
  /** Nodes bounded up to the playhead (pop order). */
  nodes: readonly BbNode[];
  order: readonly number[];
  current: number | null;
  slot: number;
  zStar: number | null;
}) {
  const box = useRef<HTMLDivElement>(null);
  const size = useElementSize(box);
  const [hover, setHover] = useState<number | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const { placed, width, depth } = useMemo(
    () =>
      layoutTree(
        nodes.map((n) => ({
          id: String(n.id),
          parent: n.parent === null ? null : String(n.parent),
        })),
      ),
    [nodes],
  );
  const maxDepth = Math.max(depth, 1);
  const labelW = 34,
    pad = 14,
    legendH = 26;
  const W = Math.max(size.width, 1),
    H = Math.max(size.height - legendH, 1);
  const pitchX = Math.max(18, (W - labelW - 2 * pad - 64) / Math.max(1, width));
  const pitchY = Math.min(44, Math.max(13, (H - 2 * pad) / maxDepth));
  const svgW = Math.max(W, labelW + 2 * pad + pitchX * width + 64);
  const svgH = Math.max(H, 2 * pad + pitchY * (maxDepth - 1) + 8);
  const pos = (id: number) => {
    const p = placed.get(String(id));
    return p ? { x: labelW + pad + pitchX * (p.col + 0.5), y: pad + 4 + pitchY * p.depth } : null;
  };
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const color = seriesVar(slot);
  const showBounds = pitchX >= 54 && pitchY >= 22;
  const r = Math.min(5.5, pitchY / 3.2);
  const hovered = hover === null ? undefined : byId.get(hover);
  // Labels: every bound when there is room; otherwise only the node at the playhead and the
  // latest incumbent. Labels are drawn after every node, on an opaque chip, so a value is never
  // struck through by a sibling marker.
  let lastIncumbent: number | null = null;
  for (const n of nodes) if (n.incumbent) lastIncumbent = n.id;
  const labelled = (n: BbNode) =>
    showBounds || n.id === current || n.id === lastIncumbent || (n.incumbent && pitchX >= 60);
  const labelText = (n: BbNode) => (n.incumbent ? `z = ${n.value}` : `U ${sig(n.bound, 4)}`);
  const cur = current === null ? undefined : byId.get(current);
  const info = hovered ?? cur;

  return (
    <div className={styles.tree}>
      <div ref={box} className={styles.treeScroll}>
        <svg
          width={svgW}
          height={svgH}
          role="img"
          aria-label={`Branch-and-bound tree: ${nodes.length} nodes bounded so far, ${nodes.filter((n) => n.status === 'pruned').length} pruned.`}
        >
          {/* Depth rows: the item decided at each depth. */}
          {Array.from({ length: maxDepth }, (_, d) => (
            <g key={d}>
              <line
                x1={labelW}
                x2={svgW - 4}
                y1={pad + 4 + pitchY * d}
                y2={pad + 4 + pitchY * d}
                className={styles.treeRule}
              />
              {d > 0 && pitchY >= 14 && (
                <text
                  x={4}
                  y={pad + 4 + pitchY * d}
                  className={styles.treeRowLabel}
                  dominantBaseline="middle"
                >
                  {`x${sub(order[d - 1])}`}
                </text>
              )}
            </g>
          ))}
          {nodes.map((n) => {
            if (n.parent === null) return null;
            const a = pos(n.parent),
              b = pos(n.id);
            if (!a || !b) return null;
            return (
              <line
                key={`e${n.id}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                className={styles.treeEdge}
                data-zero={n.decision === 0 || undefined}
                data-pruned={n.status === 'pruned' || undefined}
              />
            );
          })}
          {nodes.map((n) => {
            const p = pos(n.id);
            if (!p) return null;
            const isCur = n.id === current;
            return (
              <g
                key={n.id}
                transform={`translate(${p.x} ${p.y})`}
                className={styles.treeNode}
                tabIndex={0}
                role="img"
                aria-label={describe(n)}
                onPointerEnter={() => setHover(n.id)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => {
                  setHover(n.id);
                  setFocused(n.id);
                }}
                onBlur={() => {
                  setHover(null);
                  setFocused(null);
                }}
              >
                <title>{describe(n)}</title>
                {n.incumbent && <circle r={r + 4} fill="none" stroke={color} strokeWidth={1.6} />}
                {n.status === 'leaf' ? (
                  <rect x={-r} y={-r} width={2 * r} height={2 * r} fill="var(--color-text-2)" />
                ) : n.status === 'pruned' ? (
                  <>
                    <circle r={r} className={styles.treePruned} />
                    <line x1={-r - 2} x2={r + 2} y1={0} y2={0} className={styles.treePrunedBar} />
                  </>
                ) : (
                  <circle r={r} fill={color} stroke="var(--chart-halo)" strokeWidth={1.2} />
                )}
                {isCur && (
                  <circle r={r + 7} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
                )}
                {focused === n.id && <circle r={r + 10} className={styles.treeFocus} />}
              </g>
            );
          })}
          {nodes.map((n) => {
            const p = pos(n.id);
            if (!p || !labelled(n)) return null;
            const off = n.id === current ? r + 10 : n.incumbent ? r + 7 : r + 5;
            const text = labelText(n);
            const w = text.length * 6.7 + 6; // JetBrains Mono at --text-2xs (11 px)
            return (
              <g key={`l${n.id}`} transform={`translate(${p.x + off} ${p.y})`} aria-hidden="true">
                <rect x={-2} y={-8} width={w} height={16} rx={3} className={styles.treeLabelBg} />
                <text
                  x={1}
                  y={0}
                  dominantBaseline="middle"
                  className={styles.treeLabel}
                  data-strong={n.id === lastIncumbent || n.id === current || undefined}
                >
                  {text}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className={styles.treeLegend}>
        <span>
          <svg width="12" height="12" aria-hidden="true">
            <circle cx="6" cy="6" r="4.5" fill={color} />
          </svg>
          branched
        </span>
        <span>
          <svg width="14" height="12" aria-hidden="true">
            <circle cx="7" cy="6" r="4.5" className={styles.treePruned} />
            <line x1="0" x2="14" y1="6" y2="6" className={styles.treePrunedBar} />
          </svg>
          pruned, ⌊U⌋ ≤ z
        </span>
        <span>
          <svg width="12" height="12" aria-hidden="true">
            <rect x="2" y="2" width="8" height="8" fill="var(--color-text-2)" />
          </svg>
          leaf
        </span>
        <span>
          <svg width="14" height="14" aria-hidden="true">
            <circle cx="7" cy="7" r="6" fill="none" stroke={color} strokeWidth="1.5" />
          </svg>
          new incumbent
        </span>
        <span className={styles.treeLegendNote}>left xᵢ = 1 · right xᵢ = 0 (dashed)</span>
        {info && (
          <span className={styles.treeInfo}>
            node {info.id}
            {info.item !== null ? ` (x${sub(info.item)} = ${info.decision})` : ' (root)'}: V ={' '}
            {info.value}, W = {info.weight}, U = {sig(info.bound, 5)}, {info.status}
            {info.incumbent ? ` · incumbent z = ${info.value}` : ''}
            {zStar !== null && info.incumbent ? ` · z⋆ = ${zStar}` : ''}
          </span>
        )}
      </div>
    </div>
  );
}
