import clsx from "clsx";
import type { ReactNode } from "react";
import s from "./route-panel.module.scss";

type Props = {
  kicker?: ReactNode;
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  headerClassName?: string;
  headingClassName?: string;
  kickerClassName?: string;
  titleClassName?: string;
  actionsClassName?: string;
  bodyClassName?: string;
};

export const RoutePanel = ({
  kicker,
  title,
  actions,
  children,
  className,
  headerClassName,
  headingClassName,
  kickerClassName,
  titleClassName,
  actionsClassName,
  bodyClassName,
}: Props) => {
  return (
    <section className={clsx(s.root, className)}>
      {(kicker || title || actions) ? (
        <div className={clsx(s.header, headerClassName)}>
          {(kicker || title) ? (
            <div className={clsx(s.heading, headingClassName)}>
              {kicker ? <span className={clsx(s.kicker, kickerClassName)}>{kicker}</span> : null}
              {title ? <h3 className={clsx(s.title, titleClassName)}>{title}</h3> : null}
            </div>
          ) : <span />}
          {actions ? <div className={clsx(s.actions, actionsClassName)}>{actions}</div> : null}
        </div>
      ) : null}
      <div className={clsx(s.body, bodyClassName)}>{children}</div>
    </section>
  );
};
