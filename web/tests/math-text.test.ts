import { describe, expect, it } from 'vitest';
import registry from '../src/generated/registry.json';
import { hasMath, scriptsToMath, splitMath } from '../src/ui/mathProse';

describe('scriptsToMath', () => {
  it('sets TeX-style scripts as math', () => {
    expect(scriptsToMath('1 − f_{k+1}/f_k (or ½)')).toBe('1 − $f_{k+1}$/$f_{k}$ (or ½)');
    expect(scriptsToMath('O(h^{2k+2}) in row k')).toBe('O($h^{2k+2}$) in row k');
    expect(scriptsToMath('β = g_kᵀy/d_{k−1}ᵀy')).toBe('β = $g_{k}^{\\top}$y/$d_{k-1}^{\\top}$y');
    expect(scriptsToMath('η = 1/(16L_max)')).toBe('η = 1/(16$L_{\\mathrm{max}}$)');
  });
  it('leaves words with underscores and plain text alone', () => {
    expect(scriptsToMath('The CG_DESCENT direction')).toBe('The CG_DESCENT direction');
    expect(scriptsToMath('no scripts here')).toBe('no scripts here');
  });
  it('leaves no raw script in any catalog summary or rate', () => {
    for (const m of registry as { id: string; summary: string; order: string }[]) {
      for (const s of [m.summary, m.order ?? '']) {
        const prose = splitMath(scriptsToMath(s)).filter((_, i) => i % 2 === 0);
        for (const p of prose) expect(p, m.id).not.toMatch(/(?<![A-Z])[A-Za-z]_[{A-Za-z0-9]|\^\{/);
      }
    }
  });
});

describe('splitMath', () => {
  it('keeps an unpaired dollar as text', () => {
    expect(splitMath('costs $5')).toEqual(['costs $5']);
    expect(hasMath('a $x$ b')).toBe(true);
  });
});

describe('method docs', () => {
  it('pair every $ in the MethodCard prose fields (MathText typesets them)', async () => {
    await import('../src/methods');
    const { listMethods } = await import('../src/core/registry');
    const methods = listMethods();
    expect(methods.length).toBeGreaterThan(100);
    for (const { spec, doc } of methods) {
      const fields = [doc?.order ?? spec.order, doc?.intuition ?? spec.summary];
      fields.push(...(doc?.pros ?? []), ...(doc?.cons ?? []));
      for (const f of fields) {
        if (!f || !f.includes('$')) continue;
        // An unpaired `$` would be printed raw.
        expect(hasMath(f), `${spec.id}: ${f}`).toBe(true);
        expect(splitMath(f).length % 2, `${spec.id}: ${f}`).toBe(1);
      }
    }
  });
});
