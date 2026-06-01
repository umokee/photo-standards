import clsx from "clsx";
import { LucideIcon } from "lucide-react";
import styles from "./button.module.scss";

interface Props {
  children?: React.ReactNode;
  icon?: LucideIcon;
  iconPosition?: "left" | "right";
  size?: "sm" | "md" | "lg" | "icon";
  variant?: "primary" | "ghost" | "warning" | "danger" | "ml";
  full?: boolean;
  disabled?: boolean;
  type?: "button" | "submit" | "reset";
  onClick?: () => void;
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
}: Props) {
  return (
    <button
      type={type}
      className={clsx(styles.button, styles[size], styles[variant], full && styles.full)}
      disabled={disabled}
      onClick={onClick}
    >
      {Icon && iconPosition === "left" && <Icon />}
      {children && <span>{children}</span>}
      {Icon && iconPosition === "right" && <Icon />}
    </button>
  );
}
