/**
 * Review fixes of the linear-systems lab: what the lab states must be what it computes (preset
 * notes, ω⋆, numbers near 1), every index it shows is 1-based, and values outside a view are
 * reported, never pinned to its edge.
 */
import { describe, expect, it } from 'vitest';
import { runMethod } from '../../src/core/registry';
import { listProblems } from '../../src/problems/registry';
import type { LinalgProblem } from '../../src/problems/linalg';
import {
  clampOmega,
  displayDescription,
  offScaleIndices,
  offText,
  omegaText,
  polylinePrefix,
  snapOmega,
  structureHint,
  structureOf,
} from '../../src/labs/linalg/model';
import {
  nearTex,
  nearText,
  pivotSymbol,
  splitMath,
  thomasStageTex,
} from '../../src/labs/linalg/texfmt';
import { msgText, oneBased } from '../../src/labs/linalg/messages';
import { num3 } from '../../src/labs/linalg/columns';
import { PRESETS, POISSON_OMEGA_OPT } from '../../src/labs/linalg/presets';
import '../../src/labs/linalg/setup';

const P = (id: string) => listProblems<LinalgProblem>('linalg').find((p) => p.id === id)!;

describe("Young's ω⋆ preset", () => {
  it('uses the exact ω⋆ and its note states the counts the runs produce', () => {
    const s = structureOf(P('poisson_1d_10').A);
    expect(POISSON_OMEGA_OPT).toBeCloseTo(s.omegaOpt!, 12);
    const young = PRESETS.find((p) => p.id === 'young')!;
    const sor = young.methods!.find((m) => m.id === 'sor')!;
    expect(sor.params.omega).toBe(POISSON_OMEGA_OPT);
    const count = (id: string, params = {}) => runMethod(id, P('poisson_1d_10'), params).nIter;
    const counts = {
      sor: count('sor', { omega: POISSON_OMEGA_OPT }),
      gs: count('gauss_seidel'),
      jacobi: count('jacobi'),
    };
    expect(counts).toEqual({ sor: 49, gs: 279, jacobi: 556 });
    expect(young.note).toContain('49 sweeps, Gauss–Seidel 279, Jacobi 556');
    const rho = runMethod('sor', P('poisson_1d_10'), { omega: POISSON_OMEGA_OPT }).extra
      .spectral_radius as number;
    expect(rho).toBeCloseTo(POISSON_OMEGA_OPT - 1, 6);
    expect(young.note).toContain('ω⋆ − 1 ≈ 0.560');
  });

  it('keeps preset notes free of raw subscripts', () => {
    for (const p of PRESETS) expect(p.note ?? '').not.toMatch(/_[A-Za-z{]/);
  });
});

describe('ω of the relaxation panel', () => {
  it('Enter / a drag near ω⋆ give the exact ω⋆; elsewhere ω rounds to 0.01', () => {
    const w = POISSON_OMEGA_OPT;
    expect(snapOmega(1.562, w)).toBe(w);
    expect(snapOmega(1.5655, w)).toBe(1.57);
    expect(snapOmega(1.2345, w)).toBe(1.23);
    expect(snapOmega(1.2345, null)).toBe(1.23);
    expect(clampOmega(2.5)).toBe(1.95);
    expect(omegaText(1.5)).toBe('1.50');
    expect(omegaText(w)).toBe('1.560');
  });
});

describe('numbers that round to 1 or 2', () => {
  it('print their distance instead', () => {
    const s = structureOf(P('nearly_singular').A);
    const plain = (v: number) => String(+v.toPrecision(4));
    expect(nearText(s.rhoJ!, plain)).toMatch(/^1 − \d\.\d×10⁻¹⁰$/);
    expect(nearText(s.omegaOpt!, plain)).toMatch(/^2 − \d\.\d×10⁻⁵$/);
    expect(nearText(0.959, plain)).toBe('0.959');
    expect(nearText(1, plain)).toBe('1');
    expect(nearTex(1 - 4.7e-10, plain)).toBe('1 - 4.7\\times 10^{-10}');
  });
});

describe('Python messages, 1-based', () => {
  it('rewrites the 0-based symbols', () => {
    expect(oneBased('zero pivot |a_00| = 0 ≤ τ = 3.3e-15 at stage 1')).toBe(
      'zero pivot |a₁₁| = 0 ≤ τ = 3.3e-15 at stage 1',
    );
    expect(oneBased('|u_22| and |r_11| and |w_4| and d_2 = a_22 − Σ l_2j²')).toBe(
      '|u₃₃| and |r₂₂| and |w₅| and d₃ = a₃₃ − Σ l₃ⱼ²',
    );
    expect(oneBased('max |a_ij − a_ji| = 1; reached max_iter=5')).toBe(
      'max |aᵢⱼ − aⱼᵢ| = 1; reached max_iter=5',
    );
    expect(oneBased('ρ(G_J) = 1.1 ≥ 1')).toBe('ρ(G_J) = 1.1 ≥ 1');
  });

  it('matches what the methods report on the zero-pivot problems', () => {
    const ge = runMethod('gaussian_elimination', P('needs_pivoting'), {});
    expect(ge.message).toContain('|a_00|');
    expect(msgText(ge.message)).toContain('|a₁₁|');
    const sing = runMethod('gaussian_elimination_pivoting', P('singular_3'), {});
    expect(msgText(sing.message)).toContain('|a₃₃|');
    const jac = runMethod('jacobi', P('needs_pivoting'), {});
    expect(msgText(jac.message)).toMatch(/^a₁₁ = 0/);
  });
});

describe('notation', () => {
  it('names each pivot as the rule does', () => {
    expect(pivotSymbol('cholesky', 1)).toBe('w_{22}');
    expect(pivotSymbol('lu_decomposition', 0)).toBe('w_{11}');
    expect(pivotSymbol('thomas', 2)).toBe('w_{3}');
    expect(pivotSymbol('gaussian_elimination', 0)).toBe('a_{11}');
  });

  it('writes the first and last Thomas stages without undefined indices', () => {
    expect(thomasStageTex(1, 3, '2')).toBe(
      "w_{1} = \\delta_{1} = 2,\\quad c'_{1} = c_{1}/w_{1},\\quad d'_{1} = b_{1}/w_{1}",
    );
    const last = thomasStageTex(3, 3, '1.5');
    expect(last).not.toContain("c'_{3}");
    expect(last).toContain("d'_{3} = (b_{3} - a_{3}d'_{2})/w_{3}");
  });

  it('splits prose and inline math', () => {
    expect(splitMath('rate $\\rho(G_J)$ per step')).toEqual(['rate ', '\\rho(G_J)', ' per step']);
  });

  it('typesets table numbers as ×10ⁿ', () => {
    expect(num3(4.28e-10)).toBe('4.28×10⁻¹⁰');
    expect(num3(-1.5e6)).toBe('−1.50×10⁶');
    expect(num3(0.5)).toBe('0.500');
  });

  it("shows the lab's G notation and a correct claim in the problem descriptions", () => {
    const d = displayDescription(P('spd_2x2'));
    expect(d).toContain('$\\rho(G_J)$');
    expect(d).toContain('$\\rho(G_{GS})$');
    expect(d).not.toMatch(/T_J|T_GS/);
    expect(displayDescription(P('diag_dominant_3'))).toContain('except Thomas');
  });
});

describe('rail hint', () => {
  it('is true on a singular A', () => {
    const h = structureHint(structureOf(P('singular_3').A));
    expect(h).toContain('elimination must meet a zero pivot');
    expect(h).toContain('GMRES');
    expect(h).not.toContain('LU, QR');
    expect(h).not.toContain('every pivot test must fail');
    // The first two pivots of singular_3 pass; only the last stage fails.
    const r = runMethod('gaussian_elimination_pivoting', P('singular_3'), {});
    expect(r.nIter).toBe(3);
  });
  it('says Thomas needs a tridiagonal A when it does not apply', () => {
    expect(structureHint(structureOf(P('diag_dominant_3').A))).toContain('Thomas');
    expect(structureHint(structureOf(P('spd_2x2').A))).not.toContain('Thomas');
  });
});

describe('components outside the view', () => {
  it('are reported with their true value', () => {
    const x60 = runMethod('jacobi', P('jacobi_diverges'), { max_iter: 60 }).x as number[];
    expect(offScaleIndices(x60, -15, 20)).toEqual([0, 1, 2]);
    expect(offText(x60[1])).toBe('−1937');
    expect(offScaleIndices([1, NaN, Infinity, 2], 0, 3)).toEqual([1, 2]);
    expect(offText(Infinity)).toBe('∞');
  });
});

describe('Gauss–Seidel staircase', () => {
  it('cuts a polyline by length', () => {
    const pts: [number, number][] = [
      [0, 0],
      [2, 0],
      [2, 2],
    ];
    expect(polylinePrefix(pts, 0.25).end).toEqual([1, 0]);
    expect(polylinePrefix(pts, 0.75)).toEqual({
      points: [
        [0, 0],
        [2, 0],
        [2, 1],
      ],
      end: [2, 1],
    });
    expect(polylinePrefix(pts, 1).end).toEqual([2, 2]);
  });
});
