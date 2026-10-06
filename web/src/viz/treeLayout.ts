/**
 * Tidy layout for rooted trees (branch and bound, recursion trees): leaves take consecutive
 * columns in depth-first order and every parent sits centered over its children. Pure; tested.
 */

export interface TreeLayoutNode {
  id: string;
  parent: string | null;
}

export interface PlacedNode {
  id: string;
  parent: string | null;
  depth: number;
  /** Column (leaves are 0, 1, 2, …; parents are the mean of their first and last child). */
  col: number;
}

/** Positions in grid units (multiply by the node pitch); unknown parents become roots. */
export function layoutTree<T extends TreeLayoutNode>(
  nodes: readonly T[],
): { placed: Map<string, PlacedNode>; width: number; depth: number } {
  const ids = new Set(nodes.map((n) => n.id));
  const children = new Map<string | null, T[]>();
  for (const n of nodes) {
    const p = n.parent !== null && ids.has(n.parent) ? n.parent : null;
    const list = children.get(p) ?? [];
    list.push(n);
    children.set(p, list);
  }
  const placed = new Map<string, PlacedNode>();
  let next = 0;
  let maxDepth = 0;
  const visit = (n: T, depth: number, parent: string | null): number => {
    maxDepth = Math.max(maxDepth, depth);
    const kids = children.get(n.id) ?? [];
    let col: number;
    if (kids.length === 0) col = next++;
    else {
      const cols = kids.map((c) => visit(c, depth + 1, n.id));
      col = (cols[0] + cols[cols.length - 1]) / 2;
    }
    placed.set(n.id, { id: n.id, parent, depth, col });
    return col;
  };
  for (const root of children.get(null) ?? []) visit(root, 0, null);
  return { placed, width: Math.max(1, next), depth: maxDepth + 1 };
}
