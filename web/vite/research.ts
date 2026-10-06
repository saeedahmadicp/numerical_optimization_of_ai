/**
 * Build-time rendering of the research studies (`research/<id>/README.md` and `research/README.md`).
 *
 * Markdown is parsed with `marked` and every `$…$` / `$$…$$` is typeset with KaTeX in Node, so the
 * browser receives finished HTML (only the KaTeX stylesheet and fonts are loaded at run time; the
 * markdown parser and KaTeX's JS never reach the bundle). Links are rewritten for the hash router:
 * a study folder → `#/research/<id>`, an in-page anchor → `#/research/<id>?s=<slug>`, any other
 * relative path → the file on GitHub. Figures (`figures/*.svg|png`) are returned so the plugin can
 * emit exactly the files a README shows (dev: served by middleware at the same relative URL).
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, normalize, posix } from 'node:path';
import katex from 'katex';
import { Marked, type MarkedExtension, type Tokens } from 'marked';

export interface StudyHeading {
  depth: number;
  text: string;
  slug: string;
}

export interface StudyDoc {
  id: string;
  title: string;
  /** The first paragraph under the title (HTML). */
  lede: string;
  html: string;
  toc: StudyHeading[];
  /** Figures shown, relative to the study folder (`figures/x.svg`), that exist on disk. */
  figures: string[];
  /** Figures the README shows that are not committed. */
  missing: string[];
  words: number;
}

export interface StudySummary {
  id: string;
  title: string;
  /** Short subtitle from the index table (`silver steps · long steps · OGM1 · PEP`). */
  tags: string;
  family: string;
  question: string;
  finding: string;
  promote: string;
  /** First figure of the study (`research/<id>/figures/x.svg`), or null. */
  figure: string | null;
  figureAlt: string;
  words: number;
  figures: number;
}

export const ABOUT_ID = 'about';

const escapeHtml = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** GitHub-style heading slug. */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[^\p{L}\p{N}\s_-]/gu, '')
    .trim()
    .replace(/\s/g, '-');
}

function tex(src: string, displayMode: boolean): string {
  try {
    return katex.renderToString(src, {
      displayMode,
      throwOnError: false,
      strict: 'ignore',
      output: 'htmlAndMathml',
    });
  } catch {
    return `<code>${escapeHtml(src)}</code>`;
  }
}

/** Plain text of inline markdown (headings in the TOC, alt text). */
function plain(md: string): string {
  return md
    .replace(/\$([^$]+)\$/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*?|__?/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\\([\\{}])/g, '$1')
    .trim();
}

export interface RenderContext {
  /** Study id (folder) or ABOUT_ID for research/README.md. */
  id: string;
  /** Absolute path of research/. */
  root: string;
  /** GitHub base: `<repo>/blob/main`. */
  blob: string;
  /** Known study ids (folder links → in-app pages). */
  studies: readonly string[];
}

function rewriteHref(href: string, ctx: RenderContext): { href: string; external: boolean } {
  if (/^[a-z]+:/i.test(href)) return { href, external: true };
  const [path, anchor] = href.split('#');
  if (!path)
    return { href: `#/research/${ctx.id}?s=${encodeURIComponent(anchor ?? '')}`, external: false };
  const base = ctx.id === ABOUT_ID ? 'research' : `research/${ctx.id}`;
  const target = posix.normalize(posix.join(base, path));
  const m = /^research\/([^/]+)\/?$/.exec(target);
  if (m && ctx.studies.includes(m[1]))
    return {
      href: `#/research/${m[1]}${anchor ? `?s=${encodeURIComponent(anchor)}` : ''}`,
      external: false,
    };
  if (target === 'research' || target === 'research/README.md')
    return { href: '#/research', external: false };
  const kind = /\.[a-z0-9]+$/i.test(target) ? 'blob' : 'tree';
  return {
    href: `${ctx.blob.replace(/\/blob\/main$/, `/${kind}/main`)}/${target}${anchor ? `#${anchor}` : ''}`,
    external: true,
  };
}

/** `flowchart LR` with `A["a<br/>b"] --> B[...]` → an ordered row of steps. */
function mermaidFlow(src: string): string | null {
  if (!/^\s*flowchart\s+(LR|TD|TB)/.test(src)) return null;
  const nodes: string[] = [];
  for (const m of src.matchAll(/\b[A-Za-z0-9_]+\["([^"]+)"\]/g)) {
    const [head, ...rest] = m[1].split(/<br\s*\/?>/);
    nodes.push(
      `<li><span class="flow-head">${escapeHtml(head)}</span>${rest.length ? `<span class="flow-body">${escapeHtml(rest.join(' '))}</span>` : ''}</li>`,
    );
  }
  return nodes.length ? `<ol class="flow">${nodes.join('')}</ol>` : null;
}

const MATH: MarkedExtension['extensions'] = [
  {
    name: 'mathBlock',
    level: 'block',
    start: (src: string) => src.match(/^\$\$/m)?.index,
    tokenizer(src: string) {
      const m = /^\$\$([\s\S]+?)\$\$[ \t]*(?:\n+|$)/.exec(src);
      if (m) return { type: 'mathBlock', raw: m[0], text: m[1].trim() };
      return undefined;
    },
    renderer: (t) =>
      `<div class="math-display">${tex((t as unknown as { text: string }).text, true)}</div>\n`,
  },
  {
    name: 'math',
    level: 'inline',
    start: (src: string) => (src.indexOf('$') >= 0 ? src.indexOf('$') : undefined),
    tokenizer(src: string) {
      let m = /^\$\$([\s\S]+?)\$\$/.exec(src);
      if (m) return { type: 'math', raw: m[0], text: m[1].trim(), display: true };
      // One soft line break may sit inside inline math (a README wraps long spans); a blank line
      // (a new paragraph) may not.
      m = /^\$(?![\s$])((?:\\.|[^$\\\n]|\n(?![ \t]*\n))+?)\$(?![0-9])/.exec(src);
      if (m && !/\s$/.test(m[1])) return { type: 'math', raw: m[0], text: m[1], display: false };
      return undefined;
    },
    renderer: (t) => {
      const tok = t as unknown as { text: string; display: boolean };
      if (tok.display) return `<span class="math-display">${tex(tok.text, true)}</span>`;
      // A long inline formula (a set of ten values) cannot wrap: it is marked so that on a
      // phone it scrolls sideways inside the text column instead of widening the page.
      return tok.text.replace(/\s+/g, '').length > 32
        ? `<span class="math-wide">${tex(tok.text, false)}</span>`
        : tex(tok.text, false);
    },
  },
];

function linkHtml(
  href: string,
  title: string | null | undefined,
  inner: string,
  ctx: RenderContext,
) {
  const r = rewriteHref(href, ctx);
  const t = title ? ` title="${escapeHtml(title)}"` : '';
  return r.external
    ? `<a href="${escapeHtml(r.href)}"${t} target="_blank" rel="noreferrer" class="ext">${inner}</a>`
    : `<a href="${escapeHtml(r.href)}"${t}>${inner}</a>`;
}

/** Inline markdown (with math and rewritten links) → HTML. */
export function inlineHtml(md: string, ctx: RenderContext): string {
  const m = new Marked({
    gfm: true,
    extensions: MATH,
    renderer: {
      link({ href, title, tokens }: Tokens.Link) {
        return linkHtml(href, title, this.parser.parseInline(tokens), ctx);
      },
    },
  });
  return m.parseInline(md.trim(), { async: false });
}

/** Render one README. */
export function renderStudy(markdown: string, ctx: RenderContext): StudyDoc {
  const toc: StudyHeading[] = [];
  const figures: string[] = [];
  const missing: string[] = [];
  const used = new Map<string, number>();
  const dir = ctx.id === ABOUT_ID ? ctx.root : join(ctx.root, ctx.id);

  const resolveFigure = (src: string): string | null => {
    if (/^[a-z]+:/i.test(src)) return null;
    const rel = posix.normalize(src);
    if (rel.startsWith('..')) return null;
    if (existsSync(join(dir, rel))) return rel;
    // A README may name the .svg while only the .png is committed (or the reverse).
    const alt = rel.endsWith('.svg')
      ? rel.replace(/\.svg$/, '.png')
      : rel.replace(/\.png$/, '.svg');
    return existsSync(join(dir, alt)) ? alt : null;
  };

  const figureHtml = (href: string, text: string): string => {
    const file = resolveFigure(href);
    const caption = text ? `<figcaption>${escapeHtml(text)}</figcaption>` : '';
    if (!file) {
      missing.push(href);
      return `<figure class="figure figure-missing"><div class="figure-plate" role="img" aria-label="${escapeHtml(text || href)} (figure not committed)"><span>Figure not committed</span><code>${escapeHtml(href)}</code></div>${caption}</figure>`;
    }
    if (!figures.includes(file)) figures.push(file);
    const url = ctx.id === ABOUT_ID ? `research/${file}` : `research/${ctx.id}/${file}`;
    return `<figure class="figure"><a class="figure-plate" href="${url}" target="_blank" rel="noreferrer" aria-label="Open the figure at full size: ${escapeHtml(text || file)}"><img src="${url}" alt="${escapeHtml(text)}" loading="lazy" decoding="async"></a>${caption}</figure>`;
  };

  const marked = new Marked({
    gfm: true,
    extensions: MATH,
    renderer: {
      heading({ tokens, depth, text }: Tokens.Heading) {
        const inner = this.parser.parseInline(tokens);
        if (depth === 1) return `<h1 class="study-title">${inner}</h1>\n`;
        let slug = slugify(plain(text));
        const n = used.get(slug) ?? 0;
        used.set(slug, n + 1);
        if (n) slug = `${slug}-${n}`;
        if (depth <= 3) toc.push({ depth, text: plain(text), slug });
        const level = Math.min(6, depth);
        return `<h${level} id="${slug}" data-anchor><a class="anchor" href="#/research/${ctx.id}?s=${slug}" aria-label="Link to this section">#</a>${inner}</h${level}>\n`;
      },
      link({ href, title, tokens }: Tokens.Link) {
        return linkHtml(href, title, this.parser.parseInline(tokens), ctx);
      },
      image({ href, text }: Tokens.Image) {
        return figureHtml(href, text);
      },
      paragraph({ tokens }: Tokens.Paragraph) {
        const solo = tokens.filter((t) => !(t.type === 'text' && !t.raw.trim()) && t.type !== 'br');
        if (solo.length > 0 && solo.every((t) => t.type === 'image'))
          return `<div class="figures" data-count="${solo.length}">${solo
            .map((t) => figureHtml((t as Tokens.Image).href, (t as Tokens.Image).text))
            .join('')}</div>\n`;
        return `<p>${this.parser.parseInline(tokens)}</p>\n`;
      },
      code({ text, lang }: Tokens.Code) {
        if (lang === 'mermaid') {
          const flow = mermaidFlow(text);
          if (flow) return flow;
        }
        const l = lang ? ` data-lang="${escapeHtml(lang)}"` : '';
        return `<pre class="code" tabindex="0"${l}><code>${escapeHtml(text)}</code></pre>\n`;
      },
      table(token: Tokens.Table) {
        const cell = (c: Tokens.TableCell, tag: 'th' | 'td', i: number) => {
          const align = token.align[i] ? ` style="text-align:${token.align[i]}"` : '';
          const scope = tag === 'th' ? ' scope="col"' : '';
          return `<${tag}${align}${scope}>${this.parser.parseInline(c.tokens)}</${tag}>`;
        };
        const head = `<tr>${token.header.map((c, i) => cell(c, 'th', i)).join('')}</tr>`;
        const body = token.rows
          .map((r) => `<tr>${r.map((c, i) => cell(c, 'td', i)).join('')}</tr>`)
          .join('');
        return `<div class="table-wrap" tabindex="0" role="region" aria-label="Table"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>\n`;
      },
    },
  });

  // Drop a leading centered header block (<div align="center">…</div>) and the Contents list: the
  // page has its own header and table of contents.
  let md = markdown.replace(/^<div align="center">[\s\S]*?<\/div>\s*/m, (block) => {
    const h1 = /^# (.+)$/m.exec(block);
    return h1 ? `# ${h1[1]}\n\n` : '';
  });
  md = md.replace(/^## Contents\n[\s\S]*?(?=^## )/m, '');

  const title = plain(/^# (.+)$/m.exec(md)?.[1] ?? ctx.id);
  const body = md.replace(/^# .+$/m, '');
  // The first paragraph (before any section) is the page's lede: render it once, above the body.
  const leadBlock = /^\s*([^#<|!`$\s-][^\n]*(?:\n(?!\n)[^\n]*)*)\n\n/.exec(
    body.replace(/^\s*\n/, ''),
  );
  const html = marked.parse(
    leadBlock ? body.replace(/^\s*\n/, '').slice(leadBlock[0].length) : body,
    {
      async: false,
    },
  );
  const firstPara = body
    .split(/\n{2,}/)
    .map((s) => s.trim())
    .find((s) => s && !/^(#|<|\||!\[|```|-{3,}|\$\$)/.test(s));
  const lede = firstPara ? inlineHtml(firstPara, ctx) : '';
  const words = body
    .replace(/\$[^$]*\$/g, ' x ')
    .split(/\s+/)
    .filter(Boolean).length;
  return { id: ctx.id, title, lede, html, toc, figures, missing, words };
}

/** Study folders: `research/<id>/README.md`. */
export function studyIds(root: string): string[] {
  if (!existsSync(root)) return [];
  return readdirSync(root, { withFileTypes: true })
    .filter(
      (d) =>
        d.isDirectory() && !/^[._]/.test(d.name) && existsSync(join(root, d.name, 'README.md')),
    )
    .map((d) => d.name)
    .sort();
}

/** The studies table of research/README.md: one row per study. */
function indexRows(root: string, ctx: Omit<RenderContext, 'id'>) {
  const path = join(root, 'README.md');
  const out = new Map<
    string,
    { tags: string; family: string; question: string; finding: string; promote: string }
  >();
  if (!existsSync(path)) return out;
  const md = readFileSync(path, 'utf8');
  const actx: RenderContext = { ...ctx, id: ABOUT_ID };
  const inline = (cell: string) => inlineHtml(cell, actx);
  for (const line of md.split('\n')) {
    const m =
      /^\|\s*\[\*\*(.+?)\*\*\]\(([^)/]+)\/?\)\s*(?:<br>\s*<sub>(.*?)<\/sub>)?\s*\|(.*)\|\s*$/.exec(
        line,
      );
    if (!m) continue;
    const cells = m[4].split(/(?<!\\)\|/);
    if (cells.length < 4) continue;
    const id = m[2];
    out.set(id, {
      tags: plain(m[3] ?? ''),
      family: cells[0].replace(/`/g, '').trim(),
      question: inline(cells[1]),
      finding: inline(cells[2]),
      promote: inline(cells[3]),
    });
  }
  return out;
}

export function readStudies(root: string, repo: string) {
  const studies = studyIds(root);
  const blob = `${repo}/blob/main`;
  const docs = new Map<string, StudyDoc>();
  const load = (id: string): StudyDoc => {
    let d = docs.get(id);
    if (!d) {
      const file = id === ABOUT_ID ? join(root, 'README.md') : join(root, id, 'README.md');
      d = renderStudy(existsSync(file) ? readFileSync(file, 'utf8') : `# ${id}\n`, {
        id,
        root,
        blob,
        studies,
      });
      docs.set(id, d);
    }
    return d;
  };
  const rows = indexRows(root, { root, blob, studies });
  const summaries = (): StudySummary[] =>
    studies.map((id) => {
      const d = load(id);
      const r = rows.get(id);
      const first = d.figures[0] ?? null;
      return {
        id,
        title: d.title,
        tags: r?.tags ?? '',
        family: r?.family ?? '',
        question: r?.question ?? d.lede,
        finding: r?.finding ?? '',
        promote: r?.promote ?? '',
        figure: first ? `research/${id}/${first}` : null,
        figureAlt: first ? `${d.title}: ${first.replace(/^figures\//, '')}` : '',
        words: d.words,
        figures: d.figures.length,
      };
    });
  return { studies, load, summaries };
}

/** Absolute path of a figure URL (`research/<id>/figures/x.svg`), or null when it is not one. */
export function figurePath(root: string, url: string): string | null {
  const m = /^\/?research\/(.+)$/.exec(decodeURIComponent(url.split('?')[0]));
  if (!m) return null;
  const rel = normalize(m[1]);
  if (rel.startsWith('..') || !/\.(svg|png|jpe?g|webp|gif)$/i.test(rel)) return null;
  const abs = join(root, rel);
  return abs.startsWith(root) && existsSync(abs) ? abs : null;
}
