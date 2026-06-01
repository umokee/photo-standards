import clsx from "clsx";
import { ReactNode } from "react";
import s from "./section.module.scss";

interface Props {
  children: ReactNode;
  bordered?: boolean;
  title?: string;
  side?: ReactNode;
  scrollable?: boolean;
  maxContentHeight?: number | string;
}

export const Section = ({
  children,
  bordered,
  title,
  side,
  scrollable,
  maxContentHeight,
}: Props) => {
  return (
    <section className={clsx(s.root, bordered && s.bordered)}>
      <div className={s.head}>
        {title && <span className={s.title}>{title}</span>}
        {side && <div className={s.side}>{side}</div>}
      </div>
      <div
        className={clsx(s.content, scrollable && s.scrollable)}
        style={maxContentHeight ? { maxHeight: maxContentHeight } : undefined}
      >
        {children}
      </div>
    </section>
  );
};
