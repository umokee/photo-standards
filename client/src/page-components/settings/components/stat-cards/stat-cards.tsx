import { ReactNode } from "react";
import s from "./stat-cards.module.scss";

interface StatItem {
  value: ReactNode;
  hint?: ReactNode;
}

type Props = {
  items: Record<string, StatItem>;
};

export const StatCards = ({ items }: Props) => {
  return (
    <div className={s.root}>
      {Object.entries(items).map(([title, config]) => (
        <StatCardItem key={title} title={title} {...config} />
      ))}
    </div>
  );
};

const StatCardItem = ({
  title,
  value,
  hint,
}: {
  title: string;
  value: ReactNode;
  hint?: ReactNode;
}) => {
  return (
    <article className={s.item}>
      <div className={s.head}>
        <span className={s.title}>{title}</span>
      </div>

      <div className={s.main}>
        <strong className={s.value}>{value}</strong>
        {hint && <span className={s.hint}>{hint}</span>}
      </div>
    </article>
  );
};
