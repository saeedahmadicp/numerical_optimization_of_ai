import { useEffect, useState } from 'react';

/**
 * A counter that grows each time the document finishes loading fonts (KaTeX fonts load lazily,
 * on first use). Key a fitted formula on it, so it measures again with the real glyph widths.
 */
export function useFontsEpoch(): number {
  const [epoch, setEpoch] = useState(0);
  useEffect(() => {
    if (typeof document === 'undefined' || !document.fonts) return;
    let live = true;
    const bump = () => {
      if (live) setEpoch((e) => e + 1);
    };
    void document.fonts.ready.then(bump);
    document.fonts.addEventListener?.('loadingdone', bump);
    return () => {
      live = false;
      document.fonts.removeEventListener?.('loadingdone', bump);
    };
  }, []);
  return epoch;
}
