import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { Icon, type IconName } from './Icon';
import { Tooltip } from './Tooltip';
import styles from './Button.module.css';

type Variant = 'primary' | 'secondary' | 'ghost';
type Size = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: IconName;
  iconRight?: IconName;
  children?: ReactNode;
}

const cx = (...c: (string | false | undefined)[]) => c.filter(Boolean).join(' ');

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = 'secondary',
    size = 'md',
    icon,
    iconRight,
    className,
    children,
    type = 'button',
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx(styles.button, styles[variant], size !== 'md' && styles[size], className)}
      {...rest}
    >
      {icon && <Icon name={icon} size={size === 'sm' ? 14 : 16} />}
      {children}
      {iconRight && <Icon name={iconRight} size={size === 'sm' ? 14 : 16} />}
    </button>
  );
});

export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  icon: IconName;
  /** Accessible name; also shown as the tooltip. */
  label: string;
  shortcut?: string;
  variant?: Variant;
  size?: Size;
  tooltip?: boolean;
  /** Where the tooltip opens (it still flips when it would leave the viewport). */
  tooltipSide?: 'top' | 'bottom';
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  {
    icon,
    label,
    shortcut,
    variant = 'ghost',
    size = 'md',
    className,
    tooltip = true,
    tooltipSide = 'top',
    type = 'button',
    ...rest
  },
  ref,
) {
  const btn = (
    <button
      ref={ref}
      type={type}
      aria-label={label}
      aria-keyshortcuts={shortcut}
      className={cx(
        styles.button,
        styles[variant],
        size !== 'md' && styles[size],
        styles.iconOnly,
        className,
      )}
      {...rest}
    >
      <Icon name={icon} size={size === 'sm' ? 14 : 16} />
    </button>
  );
  if (!tooltip) return btn;
  return (
    // The tip repeats the accessible name (and the shortcut, already in aria-keyshortcuts), so it
    // is not linked as a description.
    <Tooltip content={label} shortcut={shortcut} side={tooltipSide} describe={false}>
      {btn}
    </Tooltip>
  );
});
