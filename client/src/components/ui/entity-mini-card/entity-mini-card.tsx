import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import s from "./entity-mini-card.module.scss";

type Props = {
  icon: LucideIcon;
  title: ReactNode;
  meta?: ReactNode;
  trailing?: ReactNode;
  to?: string;
  className?: string;
  iconClassName?: string;
  bodyClassName?: string;
  titleClassName?: string;
  metaClassName?: string;
  trailingClassName?: string;
};

export const EntityMiniCard = ({
  icon: Icon,
  title,
  meta,
  trailing,
  to,
  className,
  iconClassName,
  bodyClassName,
  titleClassName,
  metaClassName,
  trailingClassName,
}: Props) => {
  const content = (
    <>
      <Icon className={clsx(s.icon, iconClassName)} />
      <span className={clsx(s.body, bodyClassName)}>
        <strong className={clsx(s.title, titleClassName)}>{title}</strong>
        {meta ? <span className={clsx(s.meta, metaClassName)}>{meta}</span> : null}
      </span>
      {trailing ? <span className={clsx(s.trailing, trailingClassName)}>{trailing}</span> : null}
    </>
  );

  if (to) {
    return <Link to={to} className={clsx(s.root, s.link, className)}>{content}</Link>;
  }

  return <div className={clsx(s.root, className)}>{content}</div>;
};
