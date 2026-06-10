import type { CSSProperties } from "react";

import { Badge } from "@/components/ui/badge/badge";
import { getInspectionResultTone } from "@/page-components/inspections/lib/inspection-result-tone";
import type { InspectionTaskSegmentDetail } from "@/types/contracts";
import clsx from "clsx";
import s from "./inspection-result-details.module.scss";

type InspectionResultDetailItem = Pick<
  InspectionTaskSegmentDetail,
  "confidence" | "debug" | "hue" | "iou" | "name" | "status"
>;

type Props = {
  details: InspectionResultDetailItem[];
  activeMatchKey?: string | null;
  onActiveMatchChange?: (matchKey: string | null) => void;
  variant?: "panel" | "history";
};

type IndexedDetail = {
  detail: InspectionResultDetailItem;
  originalIndex: number;
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

  const indexedDetails = details.map((detail, originalIndex) => ({
    detail,
    originalIndex,
  }));

  const renderDetailRow = ({ detail, originalIndex }: IndexedDetail) => {
    const matchKey = String(originalIndex);
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
  };

  if (variant === "history") {
    return (
      <div className={clsx(s.root, s.rootHistory)}>
        <div className={s.list}>{indexedDetails.map(renderDetailRow)}</div>
      </div>
    );
  }

  const issueItems = indexedDetails.filter(({ detail }) => detail.status !== "ok");
  const checkedItems = indexedDetails.filter(({ detail }) => detail.status === "ok");

  return (
    <div className={s.root}>
      <div className={s.list}>
        <section className={s.reportSection}>
          <div className={s.reportHead}>
            <span className={s.reportTitle}>Требуют внимания</span>
            <span className={s.reportCount}>{issueItems.length}</span>
          </div>

          <div className={s.reportList}>
            {issueItems.length > 0 ? (
              issueItems.map(renderDetailRow)
            ) : (
              <div className={s.empty}>Замечаний по выбранным элементам нет.</div>
            )}
          </div>
        </section>

        {checkedItems.length > 0 ? (
          <details className={s.reportDisclosure}>
            <summary className={s.reportHead}>
              <span className={s.reportTitle}>Проверенные элементы</span>
              <span className={s.reportCount}>{checkedItems.length}</span>
            </summary>

            <div className={s.reportList}>{checkedItems.map(renderDetailRow)}</div>
          </details>
        ) : null}
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
  const showMeta = showConfidence || showIou;

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
            <Badge type={badgeType}>{badgeLabel}</Badge>
          </div>
        ) : null}
      </div>

      {variant === "panel" && detail.status !== "ok" ? (
        <div className={s.rowText}>{getDetailMessage(detail)}</div>
      ) : null}

      {showMeta ? (
        <div className={s.rowMeta}>
          {showConfidence ? <span>уверенность {formatMetricPercent(detail.confidence)}</span> : null}
          {showIou ? <span>зона {formatMetricPercent(detail.iou)}</span> : null}
        </div>
      ) : null}
    </>
  );
};

const formatMetricPercent = (value: number | null | undefined) =>
  `${Math.round((value ?? 0) * 100)}%`;

const getDetailMessage = (detail: InspectionResultDetailItem) => {
  const expectedName = getDebugString(detail.debug, "expected_name");

  switch (detail.status) {
    case "ok":
      return "Элемент найден в ожидаемой зоне.";
    case "missing":
      return "Элемент не найден в ожидаемой зоне.";
    case "extra":
      return expectedName
        ? `На фото найден другой элемент. Ожидалось: ${expectedName}.`
        : "На фото есть лишний элемент для выбранного эталона.";
    case "unmatched":
      return "Не удалось уверенно сопоставить элемент с эталоном.";
    default:
      return "Требуется ручная проверка.";
  }
};

const getDebugString = (debug: Record<string, unknown> | null | undefined, key: string) => {
  const value = debug?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
};
