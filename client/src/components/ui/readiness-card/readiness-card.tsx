import clsx from "clsx";
import { AlertTriangle, CheckCircle2, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import s from "./readiness-card.module.scss";

type Props = {
  ready: boolean;
  title: ReactNode;
  value: ReactNode;
  hint: ReactNode;
  className?: string;
  topClassName?: string;
  iconWrapClassName?: string;
  titleClassName?: string;
  valueClassName?: string;
  hintClassName?: string;
  readyIcon?: LucideIcon;
  blockedIcon?: LucideIcon;
};

export const ReadinessCard = ({
  ready,
  title,
  value,
  hint,
  className,
  topClassName,
  iconWrapClassName,
  titleClassName,
  valueClassName,
  hintClassName,
  readyIcon: ReadyIcon = CheckCircle2,
  blockedIcon: BlockedIcon = AlertTriangle,
}: Props) => {
  const Icon = ready ? ReadyIcon : BlockedIcon;

  return (
    <div className={clsx(s.root, ready ? s.ready : s.blocked, className)}>
      <div className={clsx(s.top, topClassName)}>
        <div className={clsx(s.iconWrap, iconWrapClassName)} data-state={ready ? "ready" : "blocked"}>
          <Icon className={s.icon} />
        </div>
        <span className={clsx(s.title, titleClassName)}>{title}</span>
      </div>
      <strong className={clsx(s.value, valueClassName)}>{value}</strong>
      <p className={clsx(s.hint, hintClassName)}>{hint}</p>
    </div>
  );
};
