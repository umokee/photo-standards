import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import s from "./metric-card.module.scss";

type Props = {
  icon?: LucideIcon;
  label: string;
  value: string | number;
  hint?: string;
  className?: string;
  tone?: string;
  variant?: "default" | "stacked" | "centered" | "valueFirst";
};

export const MetricCard = ({ icon: Icon, label, value, hint, className, tone, variant = "default" }: Props) => {
  const labelNode = <span className={s.label}>{label}</span>;
  const valueNode = <strong className={s.value}>{value}</strong>;
  const hintNode = hint ? <small className={s.hint}>{hint}</small> : null;

  return (
    <article className={clsx(s.root, s[variant], className)} data-tone={tone}>
      {Icon ? <Icon className={s.icon} /> : null}
      <div className={s.content}>
        {variant === "valueFirst" ? (
          <>
            {valueNode}
            {labelNode}
            {hintNode}
          </>
        ) : (
          <>
            {labelNode}
            {valueNode}
            {variant === "default" ? null : hintNode}
          </>
        )}
      </div>
      {variant === "default" ? hintNode : null}
    </article>
  );
};
