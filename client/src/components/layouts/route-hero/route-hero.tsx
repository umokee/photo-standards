import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import s from "./route-hero.module.scss";

type Props = {
  eyebrow: string;
  title: string;
  description: string;
  icon?: LucideIcon;
  actions?: ReactNode;
  className?: string;
  actionsClassName?: string;
};

export const RouteHero = ({ eyebrow, title, description, icon: Icon, actions, className, actionsClassName }: Props) => {
  return (
    <header className={clsx(s.root, className)}>
      <div className={s.copy}>
        <span className={s.eyebrow}>
          {Icon ? <Icon /> : null}
          {eyebrow}
        </span>
        <h1 className={s.title}>{title}</h1>
        <p className={s.description}>{description}</p>
      </div>
      {actions ? <div className={clsx(s.actions, actionsClassName)}>{actions}</div> : null}
    </header>
  );
};
