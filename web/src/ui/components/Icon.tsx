/** A small stroked icon set (16×16 grid, 1.5px strokes, currentColor). */
import type { SVGProps } from 'react';

const PATHS = {
  play: <path d="M5 3.5v9l7.5-4.5z" fill="currentColor" stroke="none" />,
  pause: (
    <>
      <rect x="4" y="3.5" width="2.6" height="9" rx="0.6" fill="currentColor" stroke="none" />
      <rect x="9.4" y="3.5" width="2.6" height="9" rx="0.6" fill="currentColor" stroke="none" />
    </>
  ),
  stepForward: (
    <>
      <path d="M4 4v8l6-4z" fill="currentColor" stroke="none" />
      <path d="M12 4v8" />
    </>
  ),
  stepBack: (
    <>
      <path d="M12 4v8L6 8z" fill="currentColor" stroke="none" />
      <path d="M4 4v8" />
    </>
  ),
  skipStart: <path d="M4 3.5v9M12.5 4 7 8l5.5 4z" />,
  skipEnd: <path d="M12 3.5v9M3.5 4 9 8l-5.5 4z" />,
  reset: <path d="M3.2 8a4.8 4.8 0 1 0 1.5-3.5M3 2.8v2.6h2.6" />,
  sun: (
    <>
      <circle cx="8" cy="8" r="2.8" />
      <path d="M8 1.5v1.4M8 13.1v1.4M1.5 8h1.4M13.1 8h1.4M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
    </>
  ),
  moon: <path d="M13 9.6A5.5 5.5 0 0 1 6.4 3a5.5 5.5 0 1 0 6.6 6.6z" />,
  monitor: (
    <>
      <rect x="2" y="3" width="12" height="8" rx="1.5" />
      <path d="M6 13.5h4M8 11v2.5" />
    </>
  ),
  link: (
    <path d="M6.8 9.2a2.6 2.6 0 0 0 3.7 0l2.2-2.2a2.6 2.6 0 0 0-3.7-3.7l-.8.8M9.2 6.8a2.6 2.6 0 0 0-3.7 0L3.3 9a2.6 2.6 0 0 0 3.7 3.7l.8-.8" />
  ),
  check: <path d="m3.5 8.5 3 3 6-7" />,
  x: <path d="m4 4 8 8M12 4l-8 8" />,
  plus: <path d="M8 3v10M3 8h10" />,
  minus: <path d="M3 8h10" />,
  search: (
    <>
      <circle cx="7" cy="7" r="4" />
      <path d="m10 10 3.2 3.2" />
    </>
  ),
  chevronDown: <path d="m4 6 4 4 4-4" />,
  chevronRight: <path d="m6 4 4 4-4 4" />,
  chevronLeft: <path d="m10 4-4 4 4 4" />,
  arrowRight: <path d="M3 8h10M9 4l4 4-4 4" />,
  arrowLeft: <path d="M13 8H3M7 4 3 8l4 4" />,
  external: (
    <path d="M9.5 2.5h4v4M13.5 2.5 8 8M11.5 9.5v3a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3" />
  ),
  sliders: <path d="M3 4.5h6M12 4.5h1M3 11.5h1M7 11.5h6M10.5 3v3M5.5 10v3" />,
  info: (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 7.2v3.6M8 5.1v.1" />
    </>
  ),
  crosshair: (
    <>
      <circle cx="8" cy="8" r="4.5" />
      <path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3" />
    </>
  ),
  cube: <path d="M8 1.8 13.5 5v6L8 14.2 2.5 11V5zM2.5 5 8 8.2 13.5 5M8 8.2v6" />,
  layers: <path d="M8 2.5 14 6 8 9.5 2 6zM2 9.2 8 12.7l6-3.5" />,
  target: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <circle cx="8" cy="8" r="2" />
    </>
  ),
  book: (
    <path d="M2.5 3.5h4A1.5 1.5 0 0 1 8 5v8a1.5 1.5 0 0 0-1.5-1.5h-4zM13.5 3.5h-4A1.5 1.5 0 0 0 8 5v8a1.5 1.5 0 0 1 1.5-1.5h4z" />
  ),
  github: (
    <path
      d="M8 1.3a6.7 6.7 0 0 0-2.1 13c.3.1.5-.1.5-.3v-1.2c-1.9.4-2.3-.8-2.3-.8-.3-.8-.7-1-.7-1-.6-.4 0-.4 0-.4.7 0 1 .7 1 .7.6 1 1.6.7 2 .6 0-.4.2-.7.4-.9-1.5-.2-3-.7-3-3.3 0-.7.2-1.3.7-1.8-.1-.2-.3-.9.1-1.8 0 0 .6-.2 1.8.7a6.3 6.3 0 0 1 3.3 0c1.3-.9 1.8-.7 1.8-.7.4.9.1 1.6.1 1.8.4.5.7 1.1.7 1.8 0 2.6-1.6 3.1-3 3.3.2.2.4.6.4 1.2V14c0 .2.2.4.5.3A6.7 6.7 0 0 0 8 1.3z"
      fill="currentColor"
      stroke="none"
    />
  ),
  grid: <path d="M2.5 2.5h4.5v4.5H2.5zM9 2.5h4.5v4.5H9zM2.5 9h4.5v4.5H2.5zM9 9h4.5v4.5H9z" />,
  menu: <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />,
  speed: <path d="M2.5 11a5.5 5.5 0 1 1 11 0M8 11l2.8-3.3" />,
  more: (
    <>
      <circle cx="3.5" cy="8" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="8" cy="8" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="12.5" cy="8" r="1.1" fill="currentColor" stroke="none" />
    </>
  ),
  copy: (
    <path d="M5.5 5.5V3.2c0-.4.3-.7.7-.7h6.6c.4 0 .7.3.7.7v6.6c0 .4-.3.7-.7.7h-2.3M3.2 5.5h6.6c.4 0 .7.3.7.7v6.6c0 .4-.3.7-.7.7H3.2c-.4 0-.7-.3-.7-.7V6.2c0-.4.3-.7.7-.7z" />
  ),
  code: <path d="m5.5 4.5-3.5 3.5 3.5 3.5M10.5 4.5l3.5 3.5-3.5 3.5" />,
  quote: (
    <path d="M3 9.5c0-2.6 1.2-4.2 3-5M3 9.5h2.8v3H3zM9 9.5c0-2.6 1.2-4.2 3-5M9 9.5h2.8v3H9z" />
  ),
  bracket: <path d="M5 3H3.5v10H5M11 3h1.5v10H11M6.5 8h3" />,
  /** A die: a new random draw (the seed control), never "reset". */
  dice: (
    <>
      <rect x="2.25" y="2.25" width="11.5" height="11.5" rx="2.5" />
      <circle cx="5.6" cy="5.6" r="1.15" fill="currentColor" stroke="none" />
      <circle cx="8" cy="8" r="1.15" fill="currentColor" stroke="none" />
      <circle cx="10.4" cy="10.4" r="1.15" fill="currentColor" stroke="none" />
    </>
  ),
} as const;

export type IconName = keyof typeof PATHS;

export interface IconProps extends Omit<SVGProps<SVGSVGElement>, 'name'> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 16, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {PATHS[name]}
    </svg>
  );
}
