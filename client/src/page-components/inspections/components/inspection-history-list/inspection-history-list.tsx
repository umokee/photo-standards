import type { InspectionHistoryItem } from "@/types/contracts";
import type { ReactNode } from "react";
import { InspectionHistoryCard } from "../inspection-history-card/inspection-history-card";
import s from "./inspection-history-list.module.scss";

type Props = {
  items: InspectionHistoryItem[];
  selectedInspectionId: string | null;
  onSelect: (inspectionId: string) => void;
  renderDetail?: (item: InspectionHistoryItem) => ReactNode;
};

export const InspectionHistoryList = ({
  items,
  selectedInspectionId,
  onSelect,
  renderDetail,
}: Props) => {
  return (
    <div className={s.list}>
      {items.map((item) => {
        return (
          <InspectionHistoryCard
            key={item.id}
            item={item}
            isExpanded={item.id === selectedInspectionId}
            onSelect={onSelect}
            detail={renderDetail?.(item)}
          />
        );
      })}
    </div>
  );
};
