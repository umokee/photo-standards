import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import s from "./empty-state-card.module.scss";

type Props = {
  icon?: LucideIcon;
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  iconClassName?: string;
  eyebrowClassName?: string;
  titleClassName?: string;
  descriptionClassName?: string;
  actionsClassName?: string;
};

export const EmptyStateCard = ({
  icon: Icon,
  eyebrow,
  title,
  description,
  actions,
  children,
  className,
  iconClassName,
  eyebrowClassName,
  titleClassName,
  descriptionClassName,
  actionsClassName,
}: Props) => {
  return (
    <section className={clsx(s.root, className)}>
      {Icon ? <div className={clsx(s.icon, iconClassName)}><Icon /></div> : null}
      {eyebrow ? <span className={clsx(s.eyebrow, eyebrowClassName)}>{eyebrow}</span> : null}
      <h3 className={clsx(s.title, titleClassName)}>{title}</h3>
      {description ? <p className={clsx(s.description, descriptionClassName)}>{description}</p> : null}
      {actions ? <div className={clsx(s.actions, actionsClassName)}>{actions}</div> : null}
      {children}
    </section>
  );
};
