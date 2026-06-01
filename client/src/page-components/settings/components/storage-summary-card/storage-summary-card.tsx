import { ReactNode } from "react";
import s from "./storage-summary-card.module.scss";

interface StorageSummaryItem {
  label: string;
  value: ReactNode;
}

interface Props {
  usedLabel?: string;
  value: ReactNode;
  items: StorageSummaryItem[];
}

export const StorageSummaryCard = ({ usedLabel = "Использовано", value, items }: Props) => {
  return (
    <article className={s.root}>
      <div className={s.summary}>
        <span className={s.label}>{usedLabel}</span>
        <div className={s.value}>
          <strong>{value}</strong>
        </div>
      </div>

      <div className={s.items}>
        {items.map((item) => (
          <div key={item.label} className={s.item}>
            <span className={s.label}>{item.label}</span>
            <strong className={s.value}>{item.value}</strong>
          </div>
        ))}
      </div>
    </article>
  );
};
