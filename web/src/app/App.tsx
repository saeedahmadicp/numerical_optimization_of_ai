import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  type ComponentType,
  type LazyExoticComponent,
  type ReactNode,
} from 'react';
import { ToastProvider } from '../ui/components/Toast';
import { withKatex } from '../ui/katex';
import { getLab, LABS } from '../labs';
import { useRoute } from './router';
import { Home } from './Home';
import type { Crumb } from './AppHeader';
import { LabsPage, NotFound, PageLoading, PlannedLab } from './Pages';
import { RouteBoundary } from './RouteBoundary';

// The component catalog (#/dev) is for lab authors: dev server only, not in the production build.
const DevPlayground = import.meta.env.DEV ? lazy(() => import('./DevPlayground')) : null;
// Pages full of math wait for KaTeX and its fonts inside their Suspense fallback (withKatex), so
// every formula paints once, typeset, and nothing shifts when KaTeX arrives.
const MethodsPage = lazy(withKatex(() => import('../site/MethodsPage')));
const MethodPage = lazy(withKatex(() => import('../site/MethodPage')));
const ResearchPage = lazy(() => import('../site/ResearchPage'));
const StudyPage = lazy(withKatex(() => import('../site/StudyPage')));

/**
 * Each lab's page: its lazy chunk plus KaTeX (withKatex), made once at startup. The registry's
 * own `lab.component` does not wait for KaTeX; the lab route uses this one.
 */
const LAB_PAGES: Readonly<Record<string, LazyExoticComponent<ComponentType>>> = Object.fromEntries(
  LABS.flatMap((lab) => {
    const load = lab.preload as (() => Promise<{ default: ComponentType }>) | undefined;
    return lab.component && load ? [[lab.id, lazy(withKatex(load))] as const] : [];
  }),
);
const PythonPage = lazy(() => import('../site/PythonPage'));

const TITLES: Record<string, string> = {
  labs: 'Labs',
  methods: 'Methods',
  research: 'Research',
  python: 'Python',
  dev: 'Design system',
  notFound: 'Not found',
};
const SITE = 'numopt — numerical optimization, iterate by iterate';

/**
 * Move keyboard focus to the new page's heading after a client-side navigation, so the next Tab
 * starts in the page and screen readers announce it. Lab pages load lazily, so this waits (up to
 * ~2 s) for a rendered `<h1>`; a heading hidden with `display: none` is skipped.
 */
function focusPageHeading(): () => void {
  let raf = 0;
  const started = performance.now();
  const tryFocus = () => {
    const h1 = [...document.querySelectorAll<HTMLElement>('h1')].find(
      (h) => h.getClientRects().length > 0,
    );
    if (h1) {
      if (!h1.hasAttribute('tabindex')) h1.setAttribute('tabindex', '-1');
      h1.focus({ preventScroll: true });
      return;
    }
    if (performance.now() - started < 2000) raf = requestAnimationFrame(tryFocus);
  };
  raf = requestAnimationFrame(tryFocus);
  return () => cancelAnimationFrame(raf);
}

export function App() {
  const route = useRoute();
  const routeKey =
    route.name === 'lab' || route.name === 'method' || route.name === 'study'
      ? `${route.name}:${route.id}`
      : route.name;
  const first = useRef(true);

  useEffect(() => {
    const lab = route.name === 'lab' ? getLab(route.id) : undefined;
    // A dead link (unknown lab, unknown path) says so in the tab. A method page sets its own
    // title ("BFGS · Methods · numopt", or "Not found · numopt" for an unknown id): its effect
    // runs before this one when it is already loaded, so this one must not overwrite it.
    if (route.name !== 'method') {
      const name =
        route.name === 'lab'
          ? (lab?.title ?? TITLES.notFound)
          : route.name === 'study'
            ? 'Research'
            : TITLES[route.name];
      document.title = name ? `${name} · numopt` : SITE;
    }
    window.scrollTo(0, 0);
    if (first.current) {
      first.current = false;
      return;
    }
    return focusPageHeading();
  }, [routeKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Every lazy page has its own error boundary (keyed by route, so navigating resets it): a page
  // that throws shows a recoverable error panel, never a blank app.
  const lazyPage = (title: string, node: ReactNode, section?: Crumb, site = true) => (
    <RouteBoundary key={routeKey} title={title} section={section}>
      <Suspense fallback={<PageLoading title={title} section={section} site={site} />}>
        {node}
      </Suspense>
    </RouteBoundary>
  );

  let page;
  switch (route.name) {
    case 'home':
      page = <Home />;
      break;
    case 'labs':
      page = <LabsPage />;
      break;
    case 'methods':
      page = lazyPage('Methods', <MethodsPage />);
      break;
    case 'method':
      page = lazyPage('Method', <MethodPage key={route.id} id={route.id} />, {
        label: 'Methods',
        href: '#/methods',
      });
      break;
    case 'research':
      page = lazyPage('Research', <ResearchPage />);
      break;
    case 'study':
      page = lazyPage('Study', <StudyPage key={route.id} id={route.id} />, {
        label: 'Research',
        href: '#/research',
      });
      break;
    case 'python':
      page = lazyPage('Python', <PythonPage />);
      break;
    case 'dev':
      page = DevPlayground ? lazyPage('Design system', <DevPlayground />) : <NotFound />;
      break;
    case 'lab': {
      const lab = getLab(route.id);
      if (!lab) page = <NotFound />;
      else if (!lab.component) page = <PlannedLab lab={lab} />;
      else {
        const Lab = LAB_PAGES[lab.id] ?? lab.component;
        page = lazyPage(lab.title, <Lab />, undefined, false);
      }
      break;
    }
    default:
      page = <NotFound />;
  }

  // No MotionConfig here: the motion library loads only with the components that animate (lab
  // shell, popovers, the first toast), and each one honors prefers-reduced-motion through
  // useMotionTransitions() (src/ui/motion.ts).
  return <ToastProvider>{page}</ToastProvider>;
}
