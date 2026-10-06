/** Clipboard helpers with an inline confirmation state (no toasts). */
import { useEffect, useRef, useState } from 'react';

/** Write text to the clipboard; false when the browser refuses (insecure context, denied). */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/** "Copied" for 1.2 s after a successful copy (an inline swap, not a toast). */
export function useCopied(ms = 1200): [boolean | 'failed', (text: string) => Promise<void>] {
  const [state, setState] = useState<boolean | 'failed'>(false);
  const timer = useRef(0);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const copy = async (text: string) => {
    const ok = await copyText(text);
    setState(ok ? true : 'failed');
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState(false), ms);
  };
  return [state, copy];
}
