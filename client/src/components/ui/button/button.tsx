import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import styles from "./button.module.scss";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  children?: ReactNode;
  icon?: LucideIcon;
  iconPosition?: "left" | "right";
  size?: "xs" | "sm" | "md" | "lg" | "icon";
  variant?: "primary" | "ghost" | "plain" | "success" | "warning" | "danger" | "ml";
  full?: boolean;
  unstyled?: boolean;
}

export default function Button({
  children,
  icon: Icon,
  iconPosition = "left",
  size = "md",
  variant = "primary",
  full = false,
  unstyled = false,
  disabled = false,
  type = "button",
  onClick,
  className,
  ...buttonProps
}: Props) {
  const shouldWrapChildren = typeof children === "string" || typeof children === "number";

  return (
    <button
      {...buttonProps}
      type={type}
      className={clsx(
        styles.button,
        styles[size],
        styles[variant],
        full && styles.full,
        unstyled && styles.unstyled,
        className,
      )}
      disabled={disabled}
      onClick={onClick}
    >
      {Icon && iconPosition === "left" && <Icon aria-hidden="true" />}
      {shouldWrapChildren ? <span>{children}</span> : children}
      {Icon && iconPosition === "right" && <Icon aria-hidden="true" />}
    </button>
  );
}
