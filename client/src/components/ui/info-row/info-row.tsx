import clsx from "clsx";
import type { ReactNode } from "react";
import s from "./info-row.module.scss";

type Layout = "row" | "column";
type Variant = "default" | "card";
type ValueSize = "default" | "stat";
type ValueWrap = "truncate" | "wrap";

interface Props {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  layout?: Layout;
  variant?: Variant;
  valueSize?: ValueSize;
  valueWrap?: ValueWrap;
  valueTitle?: string;
}

export const InfoRow = ({
  label,
  value,
  hint,
  layout = "row",
  variant = "default",
  valueSize = "default",
  valueWrap,
  valueTitle,
}: Props) => {
  const resolvedValueWrap = valueWrap ?? (layout === "column" ? "wrap" : "truncate");

  return (
    <div
      className={clsx(
        s.root,
        s[layout],
        s[variant],
        valueSize === "stat" && s.valueSizeStat,
        resolvedValueWrap === "wrap" ? s.valueWrapWrap : s.valueWrapTruncate
      )}
    >
      <div className={s.labelBlock}>
        <span className={s.label}>{label}</span>
        {hint ? <span className={s.hint}>{hint}</span> : null}
      </div>

      <strong className={s.value} title={valueTitle}>
        {value}
      </strong>
    </div>
  );
};
