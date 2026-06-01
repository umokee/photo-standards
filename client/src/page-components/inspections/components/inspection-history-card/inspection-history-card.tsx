import { Badge } from "@/components/ui/badge/badge";
import type { InspectionHistoryItem } from "@/types/contracts";
import { inspectionModeLabel, inspectionStatusLabel } from "@/utils/labels";
import clsx from "clsx";
import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import {
  formatInspectionHistoryDateTime,
  getInspectionHistoryStatusBadgeType,
} from "../../lib/inspection-history";
import s from "./inspection-history-card.module.scss";

type Props = {
  item: InspectionHistoryItem;
  isExpanded: boolean;
  onSelect: (inspectionId: string) => void;
  detail?: ReactNode;
};

export const InspectionHistoryCard = ({ item, isExpanded, onSelect, detail }: Props) => {
  const badgeType = getInspectionHistoryStatusBadgeType(item.status);
  const title = item.standard_name || "Проверка без эталона";
  const contextMeta = [item.camera_name, item.model_name].filter((value): value is string =>
    Boolean(value)
  );
  const mismatchCount = Math.max(item.total_segments - item.matched_segments, 0);
  const noteLabel = item.notes?.trim() ? "Есть примечание" : null;
  const issueLabel =
    item.status === "passed"
      ? "Без замечаний"
      : mismatchCount > 0
        ? `Не совпало ${mismatchCount}`
        : "Есть замечания";
  const metaParts = [
    formatInspectionHistoryDateTime(item.inspected_at),
    inspectionModeLabel(item.mode),
    `Совпало ${item.matched_segments} из ${item.total_segments}`,
    ...contextMeta,
  ];

  return (
    <article className={clsx(s.card, isExpanded && s.cardExpanded)}>
      <button type="button" className={s.header} onClick={() => onSelect(item.id)}>
        <div className={s.headerTop}>
          <span className={s.title}>{title}</span>

          <div className={s.status}>
            <Badge type={badgeType}>{inspectionStatusLabel(item.status)}</Badge>
            <ChevronRight className={s.chevron} size={16} />
          </div>
        </div>

        <div className={s.metaLine}>
          {metaParts.map((part, index) => (
            <span key={`${part}-${index}`} className={s.metaItem}>
              {part}
            </span>
          ))}
        </div>
      </button>

      {isExpanded && detail ? <div className={s.body}>{detail}</div> : null}
    </article>
  );
};
