import type { CSSProperties } from "react";

import { Badge } from "@/components/ui/badge/badge";
import { getInspectionResultTone } from "@/page-components/inspections/lib/inspection-result-tone";
import type { InspectionTaskSegmentDetail } from "@/types/contracts";
import clsx from "clsx";
import s from "./inspection-result-details.module.scss";

type InspectionResultDetailItem = Pick<
  InspectionTaskSegmentDetail,
  "confidence" | "hue" | "iou" | "name" | "status"
>;

type Props = {
  details: InspectionResultDetailItem[];
  activeMatchKey?: string | null;
  onActiveMatchChange?: (matchKey: string | null) => void;
  variant?: "panel" | "history";
};

export const InspectionResultDetails = ({
  details,
  activeMatchKey = null,
  onActiveMatchChange,
  variant = "panel",
}: Props) => {
  if (details.length === 0) {
    return null;
  }

  return (
    <div className={clsx(s.root, variant === "history" && s.rootHistory)}>
      <div className={s.list}>
        {details.map((detail, index) => {
          const matchKey = String(index);
          const tone = getInspectionResultTone(detail.status);
          const isActive = activeMatchKey === matchKey;
          const rowStyle = {
            "--detail-stroke": tone.stroke,
          } as CSSProperties;

          const content = (
            <DetailRowContent
              detail={detail}
              badgeType={tone.badge}
              badgeLabel={tone.label}
              toneColor={tone.stroke}
              variant={variant}
            />
          );

          if (onActiveMatchChange) {
            return (
              <button
                key={matchKey}
                type="button"
                className={clsx(
                  s.row,
                  variant === "history" && s.rowHistory,
                  s.rowInteractive,
                  isActive && s.rowActive
                )}
                style={rowStyle}
                aria-pressed={isActive}
                onClick={() => onActiveMatchChange(isActive ? null : matchKey)}
                onContextMenu={(event) => {
                  event.preventDefault();
                  onActiveMatchChange(null);
                }}
              >
                {content}
              </button>
            );
          }

          return (
            <div
              key={matchKey}
              className={clsx(s.row, variant === "history" && s.rowHistory)}
              style={rowStyle}
            >
              {content}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const DetailRowContent = ({
  detail,
  badgeType,
  badgeLabel,
  toneColor,
  variant,
}: {
  detail: InspectionResultDetailItem;
  badgeType: "info" | "success" | "warning" | "danger";
  badgeLabel: string;
  toneColor: string;
  variant: "panel" | "history";
}) => {
  const showConfidence = detail.confidence !== null;
  const showIou = detail.iou !== null;
  const showHistoryMeta = variant === "history" && (showConfidence || showIou);

  return (
    <>
      <div className={s.rowHead}>
        <div className={s.rowTitle}>
          <span
            className={s.dot}
            style={{
              background: detail.hue !== null ? `hsl(${detail.hue}, 70%, 50%)` : toneColor,
            }}
          />
          <span className={s.name}>{detail.name}</span>
        </div>

        {variant === "panel" ? (
          <div className={s.rowSide}>
            {showConfidence && (
              <span className={s.confidence}>{Math.round(detail.confidence! * 100)}%</span>
            )}
            <Badge type={badgeType}>{badgeLabel}</Badge>
          </div>
        ) : null}
      </div>

      {variant === "panel" && showIou ? (
        <div className={s.rowMeta}>
          <span>Совпадение по площади: {Math.round(detail.iou! * 100)}%</span>
        </div>
      ) : null}

      {showHistoryMeta ? (
        <div className={s.rowMeta}>
          {showConfidence ? <span>Точность {Math.round(detail.confidence! * 100)}%</span> : null}
          {showIou ? <span>Совпадение {Math.round(detail.iou! * 100)}%</span> : null}
        </div>
      ) : null}
    </>
  );
};
