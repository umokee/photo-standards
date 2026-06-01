import clsx from "clsx";
import { ReactNode } from "react";
import s from "./surface-section.module.scss";

interface Props {
  title?: ReactNode;
  hint?: ReactNode;
  aside?: ReactNode;
  className?: string;
  children: ReactNode;
  transparent?: boolean;
  direction?: "column" | "row";
}

export default function SurfaceSection({
  title,
  hint,
  aside,
  className,
  children,
  transparent,
  direction = "column",
}: Props) {
  const hasHeader = !!title || !!hint || !!aside;

  return (
    <section className={clsx(s.root, transparent && s.transparent, className)}>
      {hasHeader && (
        <div className={clsx(s.header, aside && s.headerWithAside)}>
          <div className={s.headerContent}>
            {title ? <span className={s.title}>{title}</span> : null}
            {hint ? <span className={s.hint}>{hint}</span> : null}
          </div>

          {aside ? <div className={s.aside}>{aside}</div> : null}
        </div>
      )}
      <div className={clsx(s.content, s[direction])}>{children}</div>
    </section>
  );
}
