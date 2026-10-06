/** A readable plain-text stand-in for TeX (Formula placeholders, titles, accessible names). */

const SYMBOLS: Record<string, string> = {
  le: '≤',
  leq: '≤',
  ge: '≥',
  geq: '≥',
  in: '∈',
  nabla: '∇',
  alpha: 'α',
  beta: 'β',
  varphi: 'φ',
  phi: 'φ',
  rho: 'ρ',
  sum: 'Σ',
  int: '∫',
  top: 'ᵀ',
  star: '⋆',
  infty: '∞',
  min: 'min',
  max: 'max',
  Omega: 'Ω',
  cdot: '·',
  times: '×',
  ldots: '…',
  dots: '…',
};

/** A readable plain-text stand-in for a TeX string (placeholders, titles). */
export function texToText(tex: string): string {
  return tex
    .replace(/\\(?:mathbf|boldsymbol|mathrm|operatorname|text|textstyle)\s*\{([^}]*)\}/g, '$1')
    .replace(/\\(?:tfrac|frac)\s*\{([^}]*)\}\s*\{([^}]*)\}/g, '$1/$2')
    .replace(/\\\|/g, '‖')
    .replace(/\\([a-zA-Z]+)/g, (_, c: string) => SYMBOLS[c] ?? '')
    .replace(/\\[,;:! ]/g, ' ')
    .replace(/[{}]/g, '')
    .replace(/\^/g, '')
    .replace(/_/g, '')
    .replace(/-/g, '−')
    .replace(/\s+/g, ' ')
    .trim();
}
