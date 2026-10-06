/**
 * The numopt mark: steepest descent with exact line search on a κ = 6 quadratic, turned −45° so
 * every step is axis-aligned — a staircase into the minimizer (docs/brand/brand.md §5).
 * Below 48 px the 16-unit pixel master is used (two steps, then the 𝐱⋆ dot).
 */
export function Logo({ size = 20 }: { size?: number }) {
  if (size < 48)
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <polyline
          points="3,4 7,4 7,7"
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth="2"
          strokeLinecap="square"
          strokeLinejoin="miter"
        />
        <rect x="10" y="8" width="4" height="4" rx="1.2" fill="var(--color-text)" />
      </svg>
    );
  const tints = ['--mark-tint-1', '--mark-tint-2', '--mark-tint-3', '--mark-tint-4'];
  const radii: [number, number][] = [
    [35.024, 14.298],
    [25.017, 10.213],
    [17.869, 7.295],
    [12.764, 5.211],
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" focusable="false">
      {radii.map(([rx, ry], i) => (
        <ellipse
          key={i}
          cx="32"
          cy="32"
          rx={rx}
          ry={ry}
          transform="rotate(45 32 32)"
          fill={`var(${tints[i]})`}
        />
      ))}
      <polyline
        points="9.3,12.893 18.352,12.893 18.352,22.251 25.037,22.251 25.037,27.026 28.447,27.026 28.447,29.462 30.187,29.462"
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth="2.6"
        strokeLinejoin="miter"
      />
      <circle
        cx="5.25"
        cy="12.893"
        r="3.3"
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth="1.9"
      />
      <circle cx="32" cy="32" r="4.2" fill="var(--color-text)" />
    </svg>
  );
}

/** Mark + "numopt" in Newsreader Medium (always lower case). */
export function Lockup({ size = 20, className }: { size?: number; className?: string }) {
  return (
    <span
      className={className}
      style={{ display: 'inline-flex', alignItems: 'center', gap: size * 0.3 }}
    >
      <Logo size={size} />
      <span
        style={{
          fontFamily: 'var(--font-serif)',
          fontWeight: 500,
          fontSize: size * 1.12,
          letterSpacing: '-0.01em',
          lineHeight: 1,
          color: 'var(--color-text)',
        }}
      >
        numopt
      </span>
    </span>
  );
}
