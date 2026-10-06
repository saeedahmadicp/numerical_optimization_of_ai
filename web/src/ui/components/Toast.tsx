import {
  createContext,
  lazy,
  Suspense,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import type { IconName } from './Icon';
import type { ToastItem } from './ToastItems';
import styles from './Toast.module.css';

// The list (and the motion library it animates with) loads with the first toast, so neither is
// on any page's critical path.
const ToastItems = lazy(() => import('./ToastItems'));

const ToastContext = createContext<(text: string, icon?: IconName) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const [used, setUsed] = useState(false);
  const next = useRef(1);
  const push = useCallback((text: string, icon?: IconName) => {
    const id = next.current++;
    setUsed(true);
    setItems((xs) => [...xs.slice(-2), { id, text, icon }]);
    // Long messages stay longer (~55 ms per character, 2.2–7 s).
    const ms = Math.max(2200, Math.min(7000, text.length * 55));
    window.setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), ms);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      {/* The live region exists from the start, so the first toast is announced. */}
      <div className={styles.region} role="status" aria-live="polite">
        {used && (
          <Suspense
            fallback={items.map((t) => (
              <div key={t.id} className={styles.toast}>
                {t.text}
              </div>
            ))}
          >
            <ToastItems items={items} />
          </Suspense>
        )}
      </div>
    </ToastContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export const useToast = () => useContext(ToastContext);
