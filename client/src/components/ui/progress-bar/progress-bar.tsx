import clsx from "clsx";
import { TriangleAlert } from "lucide-react";
import s from "./progress-bar.module.scss";

interface Props {
  value: number;
  max: number;
  warn?: boolean;
  showLabel?: boolean;
}

export default function ProgressBar({ value, max, warn, showLabel }: Props) {
  const pct = max > 0 ? (value / max) * 100 : 0;

  return (
    <div className={s.root}>
      <div className={s.track}>
        <div className={clsx(s.fill, warn && s.warn)} style={{ width: `${pct}%` }} />
      </div>

      {showLabel && (
        <span className={s.label}>
          {value}/{max}
        </span>
      )}

      <span className={s.iconSlot}>
        {warn ? <TriangleAlert size={14} className={s.warn} /> : null}
      </span>
    </div>
  );
}
