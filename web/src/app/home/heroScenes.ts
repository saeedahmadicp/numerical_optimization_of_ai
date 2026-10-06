/** The hero cycles through these labs' previews, in this order (web/README.md, "Home page"). */
export const HERO_SCENES = ['unconstrained', 'roots', 'lp', 'interpolation'] as const;

/** Short scene names for the scene switcher. */
export const SCENE_NAMES: Record<(typeof HERO_SCENES)[number], string> = {
  unconstrained: 'Descent',
  roots: 'Roots',
  lp: 'Simplex',
  interpolation: 'Interpolation',
};
