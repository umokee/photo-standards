import clsx from "clsx";
import { ReactNode } from "react";
import Toggle from "../toggle/toggle";
import s from "./toggle-card.module.scss";

interface Props {
  title: ReactNode;
  hint?: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  className?: string;
}

export default function ToggleCard({ title, hint, checked, onChange, className }: Props) {
  return (
    <div className={clsx(s.root, className)}>
      <div className={s.content}>
        <span className={s.title}>{title}</span>
        {hint ? <span className={s.hint}>{hint}</span> : null}
      </div>

      <Toggle checked={checked} onChange={onChange} />
    </div>
  );
}
