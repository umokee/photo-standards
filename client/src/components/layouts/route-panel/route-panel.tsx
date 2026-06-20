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
  clipped?: boolean;
  bodyScroll?: boolean;
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
  clipped = false,
  bodyScroll = false,
}: Props) => {
  return (
    <section className={clsx(s.root, clipped && s.rootClipped, className)}>
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
      <div className={clsx(s.body, bodyScroll && s.bodyScroll, bodyClassName)}>{children}</div>
    </section>
  );
};
