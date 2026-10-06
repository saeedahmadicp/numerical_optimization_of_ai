import { useMemo, type ReactNode } from 'react';
import { motion } from 'motion/react';
import { usePrefersReducedMotion } from '../play/reducedMotion';
import { layoutTree } from './treeLayout';
import styles from './TreeView.module.css';
import { baseTransition } from '../ui/motion';

/**
 * Status of a search-tree node. Each is drawn with a word and a shape, never color alone:
 * open (dashed, waiting), branched (solid), pruned (struck through, faint), infeasible (×),
 * incumbent (double border: best feasible so far), optimal (double border + ✓).
 */
export type TreeNodeStatus =
  'open' | 'branched' | 'pruned' | 'infeasible' | 'incumbent' | 'optimal';

export interface TreeNode {
  id: string;
  parent: string | null;
  /** Main line (a bound, a value): short, tabular. */
  label: ReactNode;
  /** Second line (e.g. "z ≤ 12.5"). */
  detail?: ReactNode;
  /** Text on the edge from the parent (the branching decision, e.g. "x₁ ≤ 2"). */
  edge?: ReactNode;
  status?: TreeNodeStatus;
  /** Step at which the node appears (shown when `step ≤ k`). Default 0. */
  step?: number;
  /** Plain-text description for assistive tech. */
  description?: string;
}

export interface TreeViewProps {
  nodes: readonly TreeNode[];
  /** Playhead: nodes with `step > k` are hidden (default: show all). */
  k?: number;
  /** The node being processed (iris ring). */
  current?: string | null;
  onSelect?: (id: string) => void;
  ariaLabel: string;
  /** Node box size in px. */
  nodeWidth?: number;
  nodeHeight?: number;
  className?: string;
}

const STATUS_WORD: Record<TreeNodeStatus, string> = {
  open: 'open',
  branched: 'branched',
  pruned: 'pruned',
  infeasible: 'infeasible',
  incumbent: 'incumbent',
  optimal: 'optimal',
};

/** A search tree (branch and bound) revealed step by step, laid out tidily. */
export function TreeView({
  nodes,
  k = Infinity,
  current,
  onSelect,
  ariaLabel,
  nodeWidth = 112,
  nodeHeight = 48,
  className,
}: TreeViewProps) {
  const reduced = usePrefersReducedMotion();
  const visible = useMemo(() => nodes.filter((n) => (n.step ?? 0) <= k), [nodes, k]);
  const { placed, width, depth } = useMemo(() => layoutTree(visible), [visible]);
  const gapX = 18,
    gapY = 46;
  const pitchX = nodeWidth + gapX,
    pitchY = nodeHeight + gapY;
  const W = width * pitchX - gapX + 8,
    H = depth * pitchY - gapY + 8;
  const pos = (id: string) => {
    const p = placed.get(id)!;
    return { x: 4 + p.col * pitchX, y: 4 + p.depth * pitchY };
  };

  return (
    <div className={`${styles.scroll} ${className ?? ''}`} role="group" aria-label={ariaLabel}>
      <div className={styles.canvas} style={{ width: W, height: H }}>
        <svg className={styles.edges} width={W} height={H} aria-hidden="true">
          {visible.map((n) => {
            if (!n.parent || !placed.has(n.parent)) return null;
            const a = pos(n.parent),
              b = pos(n.id);
            const x1 = a.x + nodeWidth / 2,
              y1 = a.y + nodeHeight,
              x2 = b.x + nodeWidth / 2,
              y2 = b.y;
            const my = (y1 + y2) / 2;
            return (
              <path
                key={n.id}
                d={`M${x1} ${y1} C${x1} ${my} ${x2} ${my} ${x2} ${y2}`}
                className={styles.edge}
                data-status={n.status}
              />
            );
          })}
        </svg>
        {visible.map((n) => {
          if (!n.parent || !n.edge || !placed.has(n.parent)) return null;
          const a = pos(n.parent),
            b = pos(n.id);
          return (
            <span
              key={`e-${n.id}`}
              className={styles.edgeLabel}
              style={{
                left: (a.x + b.x) / 2 + nodeWidth / 2,
                top: (a.y + nodeHeight + b.y) / 2,
              }}
            >
              {n.edge}
            </span>
          );
        })}
        {visible.map((n) => {
          const { x, y } = pos(n.id);
          const status = n.status ?? 'open';
          const Tag = onSelect ? motion.button : motion.div;
          return (
            <Tag
              key={n.id}
              type={onSelect ? 'button' : undefined}
              className={styles.node}
              data-status={status}
              data-current={current === n.id || undefined}
              style={{ left: x, top: y, width: nodeWidth, height: nodeHeight }}
              initial={reduced ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={baseTransition}
              onClick={onSelect ? () => onSelect(n.id) : undefined}
              aria-label={n.description ?? `Node ${n.id}, ${STATUS_WORD[status]}`}
              aria-current={current === n.id ? 'step' : undefined}
            >
              <span className={styles.label}>{n.label}</span>
              <span className={styles.meta}>
                {n.detail && <span className={styles.detail}>{n.detail}</span>}
                <span className={styles.status}>
                  <span aria-hidden="true">{STATUS_MARK[status]}</span> {STATUS_WORD[status]}
                </span>
              </span>
            </Tag>
          );
        })}
      </div>
    </div>
  );
}

const STATUS_MARK: Record<TreeNodeStatus, string> = {
  open: '○',
  branched: '◇',
  pruned: '–',
  infeasible: '×',
  incumbent: '★',
  optimal: '✓',
};
