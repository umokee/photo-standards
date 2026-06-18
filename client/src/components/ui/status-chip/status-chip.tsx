import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import s from "./status-chip.module.scss";

export type StatusChipTone = "neutral" | "accent" | "success" | "warning" | "danger";

type Props = {
  children: ReactNode;
  tone?: StatusChipTone;
  icon?: LucideIcon;
  uppercase?: boolean;
  className?: string;
  title?: string;
};

export const StatusChip = ({
  children,
  tone = "neutral",
  icon: Icon,
  uppercase = true,
  className,
  title,
}: Props) => {
  return (
    <span className={clsx(s.root, s[tone], uppercase && s.uppercase, className)} title={title}>
      {Icon ? <Icon className={s.icon} /> : null}
      {children}
    </span>
  );
};
