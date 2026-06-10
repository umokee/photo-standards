import Button from "@/components/ui/button/button";
import QueryState from "@/components/ui/query-state/query-state";
import { InspectionResultDetails } from "@/page-components/inspections/components/inspection-result-details/inspection-result-details";
import type { InspectionResult } from "@/types/contracts";
import clsx from "clsx";
import { ChevronDown, ChevronUp, ImageIcon } from "lucide-react";
import type { ReactNode } from "react";
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
      <section className={s.detailSection}>
        <SectionHead title="Итог проверки" />

        <div className={s.summaryStrip}>
          <SummaryItem label="На месте" value={okSegmentResults.length} />
          <SummaryItem label="Отсутствует" value={missingSegmentResults.length} />
          <SummaryItem label="Лишние" value={extraSegmentResults.length} />
        </div>

        {alignmentIssue ? (
          <div className={s.summaryNote}>
            <strong>Проблема выравнивания:</strong> {alignmentIssue}
          </div>
        ) : null}
      </section>

      <section className={s.detailSection}>
        <SectionHead
          title="Компоненты"
          side={
            okSegmentResults.length ? (
              <Button
                size="sm"
                variant="ghost"
                icon={resolvedOpen ? ChevronUp : ChevronDown}
                onClick={() => setResolvedOpen((value) => !value)}
              >
                {resolvedOpen ? "Скрыть совпавшие" : "Показать совпавшие"}
              </Button>
            ) : undefined
          }
        />

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
              <ComponentGroup title="Отсутствуют" count={missingSegmentResults.length}>
                <InspectionResultDetails details={missingSegmentResults} variant="history" />
              </ComponentGroup>
            ) : null}

            {extraSegmentResults.length ? (
              <ComponentGroup title="Лишние" count={extraSegmentResults.length}>
                <InspectionResultDetails details={extraSegmentResults} variant="history" />
              </ComponentGroup>
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
              <ComponentGroup title="Совпавшие" count={okSegmentResults.length}>
                {resolvedOpen ? (
                  <InspectionResultDetails details={okSegmentResults} variant="history" />
                ) : (
                  <div className={s.collapsedSummary}>
                    Совпавшие компоненты скрыты · {okSegmentResults.length}
                  </div>
                )}
              </ComponentGroup>
            ) : null}
          </div>
        )}
      </section>

      <section className={s.detailSection}>
        <SectionHead title="Изображения" />

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
      </section>

      <section className={s.detailSection}>
        <SectionHead title="Контекст проверки" />

        <div className={s.factGrid}>
          {contextItems.map(({ label, value }) => (
            <FactItem key={label} label={label} value={value} />
          ))}

          {inspection.notes ? (
            <FactItem label="Примечание" value={inspection.notes} wide />
          ) : null}
        </div>
      </section>
    </div>
  );
};

const SectionHead = ({ title, side }: { title: string; side?: ReactNode }) => (
  <div className={s.sectionHead}>
    <span className={s.sectionTitle}>{title}</span>
    {side ? <div className={s.sectionSide}>{side}</div> : null}
  </div>
);

const SummaryItem = ({ label, value }: { label: string; value: number }) => (
  <div className={s.summaryItem}>
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
);

const ComponentGroup = ({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: ReactNode;
}) => (
  <div className={s.componentGroup}>
    <div className={s.componentGroupHead}>
      <span>{title}</span>
      <span>{count}</span>
    </div>
    <div className={s.componentGroupBody}>{children}</div>
  </div>
);

const FactItem = ({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) => (
  <div className={clsx(s.factItem, wide && s.factItemWide)}>
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
);

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
