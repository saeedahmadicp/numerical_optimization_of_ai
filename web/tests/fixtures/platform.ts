/**
 * Where the Python reference dumps under tests/**\/fixtures come from (gen_platform.py).
 *
 * The TypeScript ports replay the arithmetic of the Python environment that wrote the committed
 * parity fixtures (aarch64 Linux with the NumPy wheel's OpenBLAS), some of it bit for bit. The
 * dumps are written on the machine that runs the tests. When that Python is canonical (a fresh
 * `numopt export` is byte-identical to src/generated), the tests compare the dumps exactly, as
 * the ports are meant to be exact there. On another platform (an x86-64 CI runner, another BLAS
 * or NumPy) the last bits differ, so a test compares
 *   - floats with the tolerance it states (`CANONICAL ? exact : tolerance`),
 *   - messages with their decimal numerals masked (`sameText`): a message quotes values that are
 *     rounding noise (‖r‖₂ = 0 on one CPU, 4.97e-16 on another),
 *   - counts exactly, except for the runs that a test names as chaotic with respect to rounding
 *     (there another CPU's Python itself takes another path); those are compared over their
 *     first iterates.
 * The committed parity fixtures (src/generated) never depend on this: their tests stay exact.
 */
export interface DumpPlatform {
  canonical: boolean;
  machine: string;
  system: string;
  python: string;
  numpy: string;
  export_files_that_differ: string[];
}

// import.meta.glob (not node:fs): src/**/*.test.ts import this file under the app tsconfig.
const FOUND = import.meta.glob<DumpPlatform>('./platform_python.json', { eager: true, import: 'default' });

/** The platform record; a missing file counts as not canonical (the tolerant comparison). */
export const DUMP_PLATFORM: DumpPlatform = Object.values(FOUND)[0] ?? {
  canonical: false,
  machine: 'unknown',
  system: 'unknown',
  python: 'unknown',
  numpy: 'unknown',
  export_files_that_differ: [],
};

/** True when the dumps round exactly like the canonical export (exact comparisons). */
export const CANONICAL = DUMP_PLATFORM.canonical;

if (!CANONICAL && !(globalThis as { __numoptPlatformNoted?: boolean }).__numoptPlatformNoted) {
  (globalThis as { __numoptPlatformNoted?: boolean }).__numoptPlatformNoted = true;
  console.info(
    `Python reference dumps from ${DUMP_PLATFORM.machine} ${DUMP_PLATFORM.system}, NumPy ` +
      `${DUMP_PLATFORM.numpy}: not the canonical platform, so the bit-exact checks of the dumps ` +
      'use the tolerances stated in each test (tests/fixtures/platform.ts).',
  );
}

const NUMERAL = /[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|\b(?:nan|inf)\b/g;
const isInteger = (t: string) => /^[-+]?\d+$/.test(t);

/**
 * Equal text up to rounding: the same text with every numeral masked, and the same integer
 * wherever both texts have an integer numeral ("5 support points", "type (4, 4)"). A numeral
 * with a point or an exponent on either side is a rounded value and is not compared.
 */
export function sameText(got: string, want: string): boolean {
  if (got === want) return true;
  if (got.replace(NUMERAL, '#') !== want.replace(NUMERAL, '#')) return false;
  const a = got.match(NUMERAL) ?? [];
  const b = want.match(NUMERAL) ?? [];
  return a.every((t, i) => !(isInteger(t) && isInteger(b[i])) || Number(t) === Number(b[i]));
}

/** Equal text with every numeral masked (the outcome of a run driven by rounding). */
export function sameTemplate(got: string, want: string): boolean {
  return got.replace(NUMERAL, '#') === want.replace(NUMERAL, '#');
}

/** The message comparison of the dump tests: exact on the canonical platform, else `sameText`. */
export function messagesMatch(got: string, want: string): boolean {
  return CANONICAL ? got === want : sameText(got, want);
}
