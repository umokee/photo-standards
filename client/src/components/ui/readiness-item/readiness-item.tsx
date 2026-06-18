import clsx from "clsx";
import { CheckCircle2, CircleDashed, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import s from "./readiness-item.module.scss";

type Variant = "row" | "compact" | "valueRow";

type Props = {
  done: boolean;
  title: ReactNode;
  description?: ReactNode;
  value?: ReactNode;
  to?: string;
  className?: string;
  variant?: Variant;
  doneIcon?: LucideIcon;
  pendingIcon?: LucideIcon;
  trailing?: ReactNode;
};

export const ReadinessItem = ({
  done,
  title,
  description,
  value,
  to,
  className,
  variant = "row",
  doneIcon: DoneIcon = CheckCircle2,
  pendingIcon: PendingIcon = CircleDashed,
  trailing,
}: Props) => {
  const Icon = done ? DoneIcon : PendingIcon;
  const content = (
    <>
      <Icon className={s.icon} />
      <span className={s.main}>
        <strong className={s.title}>{title}</strong>
        {description ? <small className={s.description}>{description}</small> : null}
      </span>
      {value !== undefined ? <strong className={s.value}>{value}</strong> : null}
      {trailing ? <span className={s.trailing}>{trailing}</span> : null}
    </>
  );

  const classes = clsx(s.root, s[variant], done ? s.done : s.pending, to && s.link, className);

  if (to) {
    return <Link to={to} className={classes}>{content}</Link>;
  }

  return <div className={classes}>{content}</div>;
};
