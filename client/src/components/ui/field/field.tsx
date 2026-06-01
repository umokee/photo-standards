import clsx from "clsx";
import { ReactNode } from "react";
import s from "./field.module.scss";

interface Props {
  label?: string;
  error?: string;
  noMargin?: boolean;
  children: ReactNode;
}

export function Field({ label, error, noMargin, children }: Props) {
  return (
    <div className={clsx(s.root, noMargin && s.noMargin)}>
      {label && <label className={s.label}>{label}</label>}
      {children}
      {error && <span className={s.errorMsg}>{error}</span>}
    </div>
  );
}
