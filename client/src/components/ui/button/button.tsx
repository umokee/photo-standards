import clsx from "clsx";
import { LucideIcon } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import styles from "./button.module.scss";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  children?: ReactNode;
  icon?: LucideIcon;
  iconPosition?: "left" | "right";
  size?: "sm" | "md" | "lg" | "icon";
  variant?: "primary" | "ghost" | "warning" | "danger" | "ml";
  full?: boolean;
}

export default function Button({
  children,
  icon: Icon,
  iconPosition = "left",
  size = "md",
  variant = "primary",
  full = false,
  disabled = false,
  type = "button",
  onClick,
  className,
  ...buttonProps
}: Props) {
  return (
    <button
      {...buttonProps}
      type={type}
      className={clsx(styles.button, styles[size], styles[variant], full && styles.full, className)}
      disabled={disabled}
      onClick={onClick}
    >
      {Icon && iconPosition === "left" && <Icon />}
      {children && <span>{children}</span>}
      {Icon && iconPosition === "right" && <Icon />}
    </button>
  );
}
