/**
 * Surface3D — f(x, y) as a lit 3-D surface with the iterate paths drawn on it.
 *
 * three.js is heavy, so never import this file directly: use `LazySurface3D` from `./index`
 * (React.lazy), which splits three into its own chunk.
 */
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { SEQUENTIAL } from '../ui/colors';
import { useChartColors } from '../ui/theme';
import { segmentAt, lerp } from '../play/timeline';
import type { Domain2D } from './Contour2D';
import { chooseTransform } from './contourField';
import styles from './viz.module.css';

/** A polyline parametrized by vertex index, so TubeGeometry segment i spans vertices i..i+1. */
class PolylineCurve extends THREE.Curve<THREE.Vector3> {
  readonly points: THREE.Vector3[];
  constructor(points: THREE.Vector3[]) {
    super();
    this.points = points;
  }
  override getPoint(u: number, target = new THREE.Vector3()): THREE.Vector3 {
    const n = this.points.length - 1;
    const x = Math.min(n, Math.max(0, u * n));
    const i = Math.min(n - 1, Math.floor(x));
    return target.lerpVectors(this.points[i], this.points[i + 1], x - i);
  }
  override getTangent(u: number, target = new THREE.Vector3()): THREE.Vector3 {
    const n = this.points.length - 1;
    const i = Math.min(n - 1, Math.max(0, Math.floor(u * n - 1e-9)));
    target.subVectors(this.points[i + 1], this.points[i]);
    if (target.lengthSq() < 1e-18) target.set(1, 0, 0);
    return target.normalize();
  }
  // Index parametrization: no arc-length remapping.
  override getUtoTmapping(u: number): number {
    return u;
  }
}

export interface Surface3DProps {
  f: (x: number, y: number) => number;
  domain: Domain2D;
  paths?: readonly { points: readonly (readonly [number, number])[]; color: string }[];
  t?: number;
  /** Compress tall surfaces: 'log' maps z = log10(f − fmin + δ); 'auto' picks like the contours. */
  zScale?: 'auto' | 'linear' | 'log';
  resolution?: number;
  ariaLabel: string;
  className?: string;
}

export default function Surface3D({
  f,
  domain,
  paths = [],
  t = 0,
  zScale = 'auto',
  resolution = 120,
  ariaLabel,
  className,
}: Surface3DProps) {
  const host = useRef<HTMLDivElement>(null);
  const colors = useChartColors();
  const state = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    world: THREE.Group;
    pathGroup: THREE.Group;
    z: (x: number, y: number) => number;
    toWorld: (x: number, y: number, z: number) => THREE.Vector3;
    render: () => void;
  } | null>(null);

  // Build scene + surface (rebuilt when the function, domain or theme changes).
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.appendChild(renderer.domElement);
    renderer.domElement.className = styles.layer;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
    scene.add(
      new THREE.HemisphereLight(
        0xffffff,
        colors.mode === 'dark' ? 0x222233 : 0x8890a0,
        colors.mode === 'dark' ? 1.4 : 1.7,
      ),
    );
    const sun = new THREE.DirectionalLight(0xffffff, colors.mode === 'dark' ? 1.1 : 1.3);
    sun.position.set(2, 4, 3);
    scene.add(sun);

    // Sample f and its transformed height.
    const [[x0, x1], [y0, y1]] = domain;
    const n = resolution;
    const raw = new Float64Array((n + 1) * (n + 1));
    let lo = Infinity,
      hi = -Infinity;
    for (let j = 0; j <= n; j++)
      for (let i = 0; i <= n; i++) {
        const v = f(x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * j) / n);
        raw[j * (n + 1) + i] = v;
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
    const delta = (hi - lo) * 1e-4 || 1e-9;
    const useLog =
      zScale === 'log' || (zScale === 'auto' && chooseTransform(raw, 'auto').scale === 'log');
    const tr = (v: number) => (useLog ? Math.log10(Math.max(0, v - lo) + delta) : v);
    const zLo = tr(lo),
      zHi = tr(hi);
    const height = 0.9;
    const z = (x: number, y: number) => {
      const v = f(x, y);
      return Number.isFinite(v) ? ((tr(v) - zLo) / (zHi - zLo || 1)) * height : height;
    };
    const span = Math.max(x1 - x0, y1 - y0);
    const toWorld = (x: number, y: number, zz: number) =>
      new THREE.Vector3(
        ((x - x0) / span - (x1 - x0) / span / 2) * 2,
        zz,
        -((y - y0) / span - (y1 - y0) / span / 2) * 2,
      );

    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array((n + 1) * (n + 1) * 3);
    const col = new Float32Array((n + 1) * (n + 1) * 3);
    const map = SEQUENTIAL;
    for (let j = 0; j <= n; j++)
      for (let i = 0; i <= n; i++) {
        const k = j * (n + 1) + i;
        const v = raw[k];
        const zz = Number.isFinite(v) ? ((tr(v) - zLo) / (zHi - zLo || 1)) * height : height;
        const p = toWorld(x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * j) / n, zz);
        pos.set([p.x, p.y, p.z], k * 3);
        const u = Math.round(Math.min(1, Math.max(0, zz / height)) * 255) * 3;
        col.set([map.lut[u] / 255, map.lut[u + 1] / 255, map.lut[u + 2] / 255], k * 3);
      }
    const idx: number[] = [];
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const a = j * (n + 1) + i,
          b = a + 1,
          c = a + n + 1,
          d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.85,
      metalness: 0,
      side: THREE.DoubleSide,
    });
    const world = new THREE.Group();
    world.add(new THREE.Mesh(geo, mat));
    // Faint base frame.
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(2 * ((x1 - x0) / span), 0.001, 2 * ((y1 - y0) / span)),
      ),
      new THREE.LineBasicMaterial({
        color: new THREE.Color(colors.mode === 'dark' ? 0x55554f : 0xb8b6ae),
      }),
    );
    world.add(frame);
    const pathGroup = new THREE.Group();
    world.add(pathGroup);
    scene.add(world);

    // Minimal orbit: drag rotates, wheel zooms.
    let az = -0.6,
      el2 = 0.88,
      dist = 4.4;
    const place = () => {
      camera.position.set(
        dist * Math.cos(el2) * Math.sin(az),
        dist * Math.sin(el2) + 0.3,
        dist * Math.cos(el2) * Math.cos(az),
      );
      camera.lookAt(0, 0.3, 0);
    };
    const render = () => renderer.render(scene, camera);
    const resize = () => {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return;
      renderer.setSize(r.width, r.height, false);
      camera.aspect = r.width / r.height;
      camera.updateProjectionMatrix();
      render();
    };
    place();
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    let drag: { x: number; y: number } | null = null;
    const down = (e: PointerEvent) => {
      drag = { x: e.clientX, y: e.clientY };
      renderer.domElement.setPointerCapture(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      az -= (e.clientX - drag.x) * 0.008;
      el2 = Math.max(0.08, Math.min(1.45, el2 + (e.clientY - drag.y) * 0.006));
      drag = { x: e.clientX, y: e.clientY };
      place();
      render();
    };
    const up = () => (drag = null);
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      dist = Math.max(2.2, Math.min(9, dist * Math.exp(e.deltaY * 0.001)));
      place();
      render();
    };
    const c = renderer.domElement;
    // One finger scrolls the page vertically (and orbits horizontally); see viz.module.css.
    c.style.touchAction = 'pan-y';
    c.style.cursor = 'grab';
    c.addEventListener('pointerdown', down);
    c.addEventListener('pointermove', move);
    c.addEventListener('pointerup', up);
    c.addEventListener('wheel', wheel, { passive: false });

    state.current = { renderer, scene, camera, world, pathGroup, z, toWorld, render };
    resize();
    return () => {
      ro.disconnect();
      c.removeEventListener('pointerdown', down);
      c.removeEventListener('pointermove', move);
      c.removeEventListener('pointerup', up);
      c.removeEventListener('wheel', wheel);
      geo.dispose();
      mat.dispose();
      renderer.dispose();
      c.remove();
      state.current = null;
    };
  }, [f, domain, zScale, resolution, colors.mode]);

  // Paths: one tube per trace, built once per trace/theme change. Each frame only reveals a
  // prefix of it (geometry.setDrawRange) and moves the head sphere.
  const tubes = useRef<
    {
      mesh: THREE.Mesh;
      head: THREE.Mesh;
      sub: number;
      n: number;
      radial: number;
      points: readonly (readonly [number, number])[];
    }[]
  >([]);
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    const radial = 6;
    const built: typeof tubes.current = [];
    for (const p of paths) {
      const n = p.points.length;
      if (n === 0) continue;
      // Subdivide each step so the path hugs the surface (fewer subdivisions for long traces).
      const sub = Math.max(1, Math.min(8, Math.floor(2400 / Math.max(1, n - 1))));
      const lift = 0.018;
      const pts: THREE.Vector3[] = [];
      let last = s.toWorld(
        p.points[0][0],
        p.points[0][1],
        s.z(p.points[0][0], p.points[0][1]) + lift,
      );
      for (let k = 0; k < Math.max(1, n - 1); k++) {
        const a = p.points[k],
          b = p.points[Math.min(k + 1, n - 1)];
        for (let q = 0; q < sub; q++) {
          const x = lerp(a[0], b[0], q / sub),
            y = lerp(a[1], b[1], q / sub);
          // Non-finite iterates (divergence) repeat the last finite point.
          if (Number.isFinite(x) && Number.isFinite(y)) last = s.toWorld(x, y, s.z(x, y) + lift);
          pts.push(last.clone());
        }
      }
      const end = p.points[n - 1];
      if (Number.isFinite(end[0]) && Number.isFinite(end[1]))
        last = s.toWorld(end[0], end[1], s.z(end[0], end[1]) + lift);
      pts.push(last.clone());
      const material = new THREE.MeshBasicMaterial({ color: new THREE.Color(p.color) });
      // WebGL ignores line widths, so the path is a thin tube. The curve is parametrized by
      // vertex index (not arc length), so tube segment i ↔ polyline segment i.
      const curve = new PolylineCurve(pts.length > 1 ? pts : [pts[0], pts[0].clone()]);
      const geometry = new THREE.TubeGeometry(curve, curve.points.length - 1, 0.009, radial, false);
      geometry.setDrawRange(0, 0);
      const mesh = new THREE.Mesh(geometry, material);
      const head = new THREE.Mesh(
        new THREE.SphereGeometry(0.032, 20, 14),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(p.color), roughness: 0.4 }),
      );
      head.position.copy(pts[0]);
      s.pathGroup.add(mesh, head);
      built.push({ mesh, head, sub, n, radial, points: p.points });
    }
    tubes.current = built;
    return () => {
      for (const t of built) {
        s.pathGroup.remove(t.mesh, t.head);
        t.mesh.geometry.dispose();
        (t.mesh.material as THREE.Material).dispose();
        t.head.geometry.dispose();
        (t.head.material as THREE.Material).dispose();
      }
      tubes.current = [];
    };
  }, [paths, f, domain, colors.mode, zScale, resolution]);

  // Per frame: reveal each tube up to the playhead and move its head (one f evaluation).
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    for (const tube of tubes.current) {
      const { i, u } = segmentAt(t, tube.n);
      const segs = Math.min((tube.n - 1) * tube.sub, Math.floor((i + u) * tube.sub));
      tube.mesh.geometry.setDrawRange(0, segs * tube.radial * 6);
      const a = tube.points[i],
        b = tube.points[Math.min(i + 1, tube.n - 1)];
      const x = lerp(a[0], b[0], u),
        y = lerp(a[1], b[1], u);
      if (Number.isFinite(x) && Number.isFinite(y))
        tube.head.position.copy(s.toWorld(x, y, s.z(x, y) + 0.018));
    }
    s.render();
  }, [t, paths, f, domain, colors.mode, zScale, resolution]);

  return (
    <div
      ref={host}
      className={`${styles.stack} ${className ?? ''}`}
      role="img"
      aria-label={ariaLabel}
    />
  );
}
