/** The animated toast list, in its own chunk with the motion library (see Toast.tsx). */
import { AnimatePresence, motion } from 'motion/react';
import { Icon, type IconName } from './Icon';
import { useMotionTransitions } from '../motion';
import styles from './Toast.module.css';

export interface ToastItem {
  id: number;
  text: string;
  icon?: IconName;
}

export default function ToastItems({ items }: { items: readonly ToastItem[] }) {
  const t = useMotionTransitions();
  return (
    <AnimatePresence>
      {items.map((it) => (
        <motion.div
          key={it.id}
          className={styles.toast}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, transition: t.exit }}
          transition={t.base}
        >
          {it.icon && <Icon name={it.icon} />}
          {it.text}
        </motion.div>
      ))}
    </AnimatePresence>
  );
}
