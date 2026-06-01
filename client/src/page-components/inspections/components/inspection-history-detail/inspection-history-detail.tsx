import { Section } from "@/components/layouts/section/section";
import Button from "@/components/ui/button/button";
import { InfoRow } from "@/components/ui/info-row/info-row";
import QueryState from "@/components/ui/query-state/query-state";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import { InspectionResultDetails } from "@/page-components/inspections/components/inspection-result-details/inspection-result-details";
import type { InspectionResult } from "@/types/contracts";
import clsx from "clsx";
import { ChevronDown, ChevronUp, ImageIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  formatInspectionHistoryDateTime,
  getInspectionHistoryAlignmentLabel,
} from "../../lib/inspection-history";
import s from "./inspection-history-detail.module.scss";

type Props = {
  inspection: InspectionResult;
};

type SegmentResult = InspectionResult["segment_results"][number];

export const InspectionHistoryDetail = ({ inspection }: Props) => {
  const sourceImageUrl = toStorageUrl(inspection.image_path);
  const referenceImageUrl = inspection.standard_reference_path
    ? toStorageUrl(inspection.standard_reference_path)
    : null;
  const resultImageUrl = inspection.result_image_path
    ? toStorageUrl(inspection.result_image_path)
    : null;

  const sortedSegmentResults = useMemo(
    () => [...inspection.segment_results].sort(compareSegmentResults),
    [inspection.segment_results]
  );

  const missingSegmentResults = useMemo(
    () => sortedSegmentResults.filter((item) => item.status === "missing"),
    [sortedSegmentResults]
  );
  const extraSegmentResults = useMemo(
    () => sortedSegmentResults.filter((item) => item.status === "extra"),
    [sortedSegmentResults]
  );
  const okSegmentResults = useMemo(
    () => sortedSegmentResults.filter((item) => item.status === "ok"),
    [sortedSegmentResults]
  );
  const hasIssues = missingSegmentResults.length + extraSegmentResults.length > 0;

  const alignmentIssue =
    inspection.alignment_status && inspection.alignment_status !== "success"
      ? getInspectionHistoryAlignmentLabel(inspection.alignment_status)
      : null;
  const contextItems = [
    { label: "Дата проверки", value: formatInspectionHistoryDateTime(inspection.inspected_at) },
    { label: "Эталон", value: inspection.standard_name || "Не указан" },
    { label: "Камера", value: inspection.camera_name || "Не указана" },
    { label: "Модель", value: inspection.model_name || "Не указана" },
  ];

  const shouldOpenResolvedByDefault = !hasIssues;
  const [resolvedOpen, setResolvedOpen] = useState(shouldOpenResolvedByDefault);

  useEffect(() => {
    setResolvedOpen(shouldOpenResolvedByDefault);
  }, [inspection.id, shouldOpenResolvedByDefault]);

  return (
    <div className={s.root}>
      <Section title="Итог проверки" bordered>
        <div className={s.summaryPanel}>
          <div className={s.metricsGrid}>
            <InfoRow
              label="На месте"
              value={String(okSegmentResults.length)}
              layout="column"
              variant="card"
              valueTitle={String(okSegmentResults.length)}
            />
            <InfoRow
              label="Отсутствует"
              value={String(missingSegmentResults.length)}
              layout="column"
              variant="card"
              valueTitle={String(missingSegmentResults.length)}
            />
            <InfoRow
              label="Лишние"
              value={String(extraSegmentResults.length)}
              layout="column"
              variant="card"
              valueTitle={String(extraSegmentResults.length)}
            />
          </div>

          {alignmentIssue ? (
            <div className={s.summaryNote}>
              <strong>Проблема выравнивания:</strong> {alignmentIssue}
            </div>
          ) : null}
        </div>
      </Section>

      <Section
        title="Компоненты"
        side={
          okSegmentResults.length ? (
            <div className={s.sectionActions}>
              <Button
                size="sm"
                variant="ghost"
                icon={resolvedOpen ? ChevronUp : ChevronDown}
                onClick={() => setResolvedOpen((value) => !value)}
              >
                {resolvedOpen ? "Скрыть совпавшие" : "Показать совпавшие"}
              </Button>
            </div>
          ) : undefined
        }
        bordered
      >
        {!inspection.segment_results.length ? (
          <QueryState
            isEmpty
            size="block"
            emptyTitle="Нет детализации"
            emptyDescription="Для этой проверки не сохранены результаты по отдельным компонентам."
          />
        ) : (
          <div className={s.componentSections}>
            {missingSegmentResults.length ? (
              <SurfaceSection title="Отсутствуют" transparent>
                <InspectionResultDetails details={missingSegmentResults} variant="history" />
              </SurfaceSection>
            ) : null}

            {extraSegmentResults.length ? (
              <SurfaceSection title="Лишние" transparent>
                <InspectionResultDetails details={extraSegmentResults} variant="history" />
              </SurfaceSection>
            ) : null}

            {!hasIssues ? (
              <div className={s.componentSuccessState}>
                <span className={s.componentSuccessTitle}>Замечаний по компонентам нет</span>
                <span className={s.componentSuccessText}>
                  Все ожидаемые позиции найдены, лишние компоненты не обнаружены.
                </span>
              </div>
            ) : null}

            {okSegmentResults.length ? (
              <SurfaceSection title="Совпавшие" transparent>
                {resolvedOpen ? (
                  <InspectionResultDetails details={okSegmentResults} variant="history" />
                ) : (
                  <div className={s.collapsedSummary}>
                    <span>Совпавшие компоненты скрыты</span>
                  </div>
                )}
              </SurfaceSection>
            ) : null}
          </div>
        )}
      </Section>

      <Section title="Изображения" bordered>
        <div className={s.imagesGrid}>
          <ImageCard
            title="Результат контроля"
            imageUrl={resultImageUrl}
            emptyText="Нет изображения результата"
            large
          />

          <div className={s.sideImages}>
            <ImageCard
              title="Исходное фото"
              imageUrl={sourceImageUrl}
              emptyText="Нет исходного фото"
            />

            <ImageCard
              title="Эталон"
              imageUrl={referenceImageUrl}
              emptyText="Эталонное фото не сохранено"
            />
          </div>
        </div>
      </Section>

      <Section title="Контекст проверки">
        <div className={s.infoGrid}>
          {contextItems.map(({ label, value }) => (
            <InfoRow key={label} label={label} value={value} layout="column" variant="card" />
          ))}
        </div>

        {inspection.notes ? (
          <div className={s.notesRow}>
            <InfoRow label="Примечание" value={inspection.notes} layout="column" variant="card" />
          </div>
        ) : null}
      </Section>
    </div>
  );
};

const ImageCard = ({
  title,
  imageUrl,
  emptyText,
  large = false,
}: {
  title: string;
  imageUrl: string | null;
  emptyText: string;
  large?: boolean;
}) => (
  <div className={clsx(s.imageCard, large && s.imageCardLarge)}>
    <div className={s.imageHead}>
      <strong>{title}</strong>
    </div>

    {imageUrl ? (
      <a className={s.imageLink} href={imageUrl} target="_blank" rel="noreferrer">
        <img className={s.image} src={imageUrl} alt={title} />
      </a>
    ) : (
      <div className={s.imageEmpty}>
        <ImageIcon size={18} />
        <span>{emptyText}</span>
      </div>
    )}
  </div>
);

function toStorageUrl(path: string | null | undefined) {
  return path ? `/storage/${path}` : null;
}

function compareSegmentResults(a: SegmentResult, b: SegmentResult) {
  const statusDiff = getSegmentStatusOrder(a.status) - getSegmentStatusOrder(b.status);

  if (statusDiff !== 0) {
    return statusDiff;
  }

  return a.name.localeCompare(b.name, "ru");
}

function getSegmentStatusOrder(status: string) {
  if (status === "missing") {
    return 0;
  }

  if (status === "extra") {
    return 1;
  }

  if (status === "ok") {
    return 2;
  }

  return 3;
}
