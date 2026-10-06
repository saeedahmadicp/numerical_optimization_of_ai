import { usePrefersReducedMotion } from '../play/reducedMotion';

/**
 * Motion tokens for the `motion` library, mirroring `tokens.css` (--duration-*, --ease-out), so
 * JS transitions and CSS transitions share one clock. Seconds, as `motion` expects.
 *
 *   transition={{ duration: MOTION.base, ease: MOTION.easeOut }}
 *   exit={{ opacity: 0, transition: { duration: MOTION.exit } }}
 *
 * Motion explains a state change (brand.md §7); nothing enters with a decorative slide.
 */
export const MOTION = {
  /** --duration-fast: cross-fades, value changes. */
  fast: 0.14,
  /** --duration-base: panels, popovers, expanding rows. */
  base: 0.22,
  /** --duration-slow: sheets. */
  slow: 0.36,
  /** Exits are quicker than entrances. */
  exit: 0.14,
  /** --ease-out: cubic-bezier(0.22, 1, 0.36, 1). */
  easeOut: [0.22, 1, 0.36, 1] as [number, number, number, number],
} as const;

/** `{ duration: MOTION.base, ease: MOTION.easeOut }` */
export const baseTransition = { duration: MOTION.base, ease: MOTION.easeOut };
/** `{ duration: MOTION.fast, ease: MOTION.easeOut }` */
export const fastTransition = { duration: MOTION.fast, ease: MOTION.easeOut };

const INSTANT = { duration: 0 };

/**
 * The transitions to pass to `motion` components, honoring `prefers-reduced-motion` (the app has
 * no MotionConfig, so the motion library stays off the critical path): under reduced motion every
 * transition is instant, like the CSS duration tokens that collapse to 0 ms.
 */
export function useMotionTransitions() {
  const reduced = usePrefersReducedMotion();
  return reduced
    ? { reduced, base: INSTANT, fast: INSTANT, slow: INSTANT, exit: INSTANT }
    : {
        reduced,
        base: baseTransition,
        fast: fastTransition,
        slow: { duration: MOTION.slow, ease: MOTION.easeOut },
        exit: { duration: MOTION.exit, ease: MOTION.easeOut },
      };
}
