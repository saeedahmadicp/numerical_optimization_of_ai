/**
 * Search over labs, methods and problems (pure; tested). A lab opens the lab, a method opens its
 * page (`#/method/<id>`: the live run, parameters, Python call, parity record and a link into the
 * lab), a problem opens the lab of its kind with the problem selected (`?p=<id>`).
 */
import { pickLabForKind } from './labPick';
import type { CatalogIndex } from './catalogTypes';

export type SearchKind = 'lab' | 'method' | 'problem';

export interface SearchItem {
  kind: SearchKind;
  id: string;
  title: string;
  /** Secondary line (lab title, summary, description). */
  subtitle: string;
  /** KaTeX shown beside the title (a lab's problem, a problem's formula). */
  tex?: string;
  /** Hash link (`#/lab/…`). */
  href: string;
  /** Lab status for lab rows and for the lab a result opens. */
  status: 'open' | 'preview' | 'planned';
  /** Extra context (method order, lab method count). */
  meta?: string;
  /** Normalized haystacks. */
  name: string;
  words: string;
}

/** The subset of a lab entry the index needs. */
export interface SearchLab {
  id: string;
  title: string;
  problem: string;
  pitch: string;
  families: readonly string[];
  problemKinds: readonly string[];
  status: 'open' | 'preview' | 'planned';
  methodCount: number;
}

/** Lower case, no diacritics, dashes unified (so "Bjorck" finds "Björck", "gauss-newton" finds "Gauss–Newton"). */
export function normalize(s: string): string {
  return s
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[‐-―−]/g, '-')
    .replace(/_/g, ' ')
    .toLowerCase();
}

export function buildSearchItems(labs: readonly SearchLab[], index: CatalogIndex): SearchItem[] {
  const out: SearchItem[] = [];
  const labOfFamily = (f: string) => labs.find((l) => l.families.includes(f));
  const labOfKind = (k: string) => pickLabForKind(labs, k);
  for (const l of labs)
    out.push({
      kind: 'lab',
      id: l.id,
      title: l.title,
      subtitle: l.pitch,
      tex: l.problem,
      href: `#/lab/${l.id}`,
      status: l.status,
      meta: `${l.methodCount} ${l.methodCount === 1 ? 'method' : 'methods'}`,
      name: normalize(l.title),
      words: normalize(`${l.id} ${l.families.join(' ')} ${l.pitch}`),
    });
  for (const m of index.methods) {
    const lab = labOfFamily(m.family);
    out.push({
      kind: 'method',
      id: m.id,
      title: m.name,
      subtitle: lab ? `${lab.title} · ${m.summary}` : m.summary,
      href: `#/method/${m.id}`,
      status: lab?.status ?? 'planned',
      meta: m.order || undefined,
      name: normalize(m.name),
      words: normalize(
        `${m.id} ${m.family} ${m.tags.join(' ')} ${m.summary} ${m.references.join(' ')} ${m.order}`,
      ),
    });
  }
  for (const p of index.problems) {
    const lab = labOfKind(p.kind);
    out.push({
      kind: 'problem',
      id: p.id,
      title: p.name,
      subtitle: lab ? `${lab.title} · ${p.description}` : p.description,
      tex: p.latex,
      href: lab ? `#/lab/${lab.id}?p=${p.id}` : '#/methods',
      status: lab?.status ?? 'planned',
      name: normalize(p.name),
      words: normalize(`${p.id} ${p.kind} ${p.tags.join(' ')} ${p.description}`),
    });
  }
  return out;
}

/** Score one item against query tokens (0 = no match: every token must match somewhere). */
export function scoreItem(item: SearchItem, tokens: readonly string[]): number {
  let score = 0;
  const nameWords = item.name.split(/[\s\-(),/]+/).filter(Boolean);
  for (const t of tokens) {
    let s = 0;
    if (item.name === t) s = 12;
    else if (item.name.startsWith(t)) s = 9;
    else if (nameWords.some((w) => w.startsWith(t))) s = 7;
    else if (item.name.includes(t)) s = 5;
    else if (normalize(item.id) === t) s = 8;
    else if (normalize(item.id).startsWith(t)) s = 6;
    else if (item.words.split(/\s+/).some((w) => w.startsWith(t))) s = 2;
    else if (item.words.includes(t)) s = 1;
    if (s === 0) return 0;
    score += s;
  }
  // Prefer things you can open now, then shorter names.
  if (item.status === 'open') score += 1.5;
  return score - item.name.length / 200;
}

const KIND_ORDER: Record<SearchKind, number> = { lab: 0, method: 1, problem: 2 };

/** Ranked results grouped by kind (labs, methods, problems), at most `perKind` each. */
export function searchItems(
  items: readonly SearchItem[],
  query: string,
  perKind = 7,
): SearchItem[] {
  const tokens = normalize(query).split(/\s+/).filter(Boolean);
  if (tokens.length === 0) {
    const labs = items.filter((i) => i.kind === 'lab');
    return [...labs.filter((l) => l.status === 'open'), ...labs.filter((l) => l.status !== 'open')];
  }
  const scored = items
    .map((item) => ({ item, s: scoreItem(item, tokens) }))
    .filter((x) => x.s > 0)
    .sort((a, b) => KIND_ORDER[a.item.kind] - KIND_ORDER[b.item.kind] || b.s - a.s);
  const count: Record<SearchKind, number> = { lab: 0, method: 0, problem: 0 };
  return scored.filter((x) => ++count[x.item.kind] <= perKind).map((x) => x.item);
}
