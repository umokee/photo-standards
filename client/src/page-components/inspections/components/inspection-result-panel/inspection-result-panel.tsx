import Button from "@/components/ui/button/button";
import Input from "@/components/ui/input/input";
import { InspectionResultDetails } from "@/page-components/inspections/components/inspection-result-details/inspection-result-details";
import type { InspectionRealtimeStatus, InspectionTaskResult } from "@/types/contracts";
import { useState } from "react";
import { useDiscardInspection } from "../../api/discard-inspection";
import { buildSaveInspectionPayload, useSaveInspection } from "../../api/save-inspection";
import {
  buildSaveRealtimeSnapshotPayload,
  useSaveRealtimeSnapshot,
} from "../../api/save-realtime-snapshot";
import s from "./inspection-result-panel.module.scss";

type BaseProps = {
  activeMatchKey: string | null;
  setActiveMatchKey: (value: string | null) => void;
};

type TaskProps = BaseProps & {
  kind?: "task";
  result: InspectionTaskResult;
  savedInspectionId: string | null;
  onSaved: (inspectionId: string) => void;
  onDiscarded: () => void;
};

type RealtimeProps = BaseProps & {
  kind: "realtime";
  result: InspectionRealtimeStatus | null;
  sessionId: string;
};

type Props = TaskProps | RealtimeProps;

type PanelResult = {
  matched: number;
  total: number;
  status: string;
  alignmentStatus: string | null;
  details: InspectionTaskResult["details"];
  modelName: string | null;
  isWarmingUp: boolean;
};

const isAlignmentFailed = (result: PanelResult) =>
  !result.isWarmingUp &&
  result.alignmentStatus !== null &&
  result.alignmentStatus !== "pending" &&
  result.alignmentStatus !== "success";

const getResultMeta = (result: PanelResult) => {
  if (result.isWarmingUp) {
    return {
      badgeType: "warning" as const,
      badgeLabel: "Подготовка",
      title: "Ожидаем первый кадр",
      description: "Проверка начнётся автоматически после подготовки модели и видеопотока.",
      scoreLabel: "Подготовка",
    };
  }

  if (isAlignmentFailed(result)) {
    return {
      badgeType: "danger" as const,
      badgeLabel: "Не пройдено",
      title: "Фото не сопоставилось с эталоном",
      description: "Ракурс или геометрия слишком отличаются. Переснимите ближе к эталонному виду.",
      scoreLabel: "Требуется новый кадр",
    };
  }

  const issueCount = Math.max(result.total - result.matched, 0);

  if (result.status === "passed") {
    return {
      badgeType: "success" as const,
      badgeLabel: "Пройдено",
      title: "Все элементы на месте",
      description: "Замечаний по выбранным элементам контроля нет.",
      scoreLabel: `${formatElementCount(result.total)} проверено`,
    };
  }

  return {
    badgeType: "danger" as const,
    badgeLabel: "Не пройдено",
    title: `${formatElementCount(issueCount)} ${issueCount === 1 ? "требует" : "требуют"} внимания`,
    description: "Проверьте отмеченные элементы и соответствующие зоны на фото.",
    scoreLabel: `${formatElementCount(result.matched)} на месте`,
  };
};

const formatElementCount = (count: number) => `${count} ${getElementWord(count)}`;

const getElementWord = (count: number) => {
  const mod10 = Math.abs(count) % 10;
  const mod100 = Math.abs(count) % 100;

  if (mod10 === 1 && mod100 !== 11) {
    return "элемент";
  }

  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    return "элемента";
  }

  return "элементов";
};

export const InspectionResultPanel = ({ ...props }: Props) => {
  const [notes, setNotes] = useState("");
  const isRealtime = props.kind === "realtime";
  const normalizedResult = normalizePanelResult(props);

  const saveMutation = useSaveInspection({
    mutationConfig: {
      onSuccess: (data) => {
        if (!isRealtime) {
          props.onSaved(data.inspection_id);
        }
      },
    },
  });
  const saveRealtimeMutation = useSaveRealtimeSnapshot();

  const effectiveInspectionId = isRealtime
    ? null
    : (props.savedInspectionId ?? props.result.inspection_id ?? null);
  const isSavePending = isRealtime ? saveRealtimeMutation.isPending : saveMutation.isPending;
  const canSave = isRealtime
    ? props.result?.state === "online" && !isSavePending
    : !effectiveInspectionId && !isSavePending;
  const meta = getResultMeta(normalizedResult);
  const shouldShowSummaryCopy =
    normalizedResult.isWarmingUp ||
    isAlignmentFailed(normalizedResult) ||
    normalizedResult.status === "passed";

  const handleSave = () => {
    if (!canSave) {
      return;
    }

    if (isRealtime) {
      const payloadResult = buildSaveRealtimeSnapshotPayload({
        session_id: props.sessionId,
        notes,
      });

      if (!payloadResult.ok) {
        return;
      }

      saveRealtimeMutation.mutate(payloadResult.data);
      return;
    }

    const payloadResult = buildSaveInspectionPayload({
      task_id: props.result.task_id,
      notes,
    });

    if (!payloadResult.ok) {
      return;
    }

    saveMutation.mutate(payloadResult.data);
  };

  const discardMutation = useDiscardInspection({
    mutationConfig: {
      onSuccess: () => {
        if (!isRealtime) {
          props.onDiscarded();
        }
      },
    },
  });

  const canDiscard =
    !isRealtime && !effectiveInspectionId && !discardMutation.isPending && !saveMutation.isPending;

  const handleDiscard = () => {
    if (isRealtime || !canDiscard) {
      return;
    }

    discardMutation.mutate({ taskId: props.result.task_id });
  };

  return (
    <div className={s.root}>
      <section className={s.summary} aria-label="Итог проверки">
        <div className={s.summaryTop}>
          <div className={s.summaryStatus}>
            <span className={s.eyebrow}>Результат</span>
            <span className={s.resultStatus} data-tone={meta.badgeType}>{meta.badgeLabel}</span>
          </div>

          <div
            className={s.summaryRatio}
            aria-label={`${normalizedResult.matched} из ${normalizedResult.total}`}
          >
            {normalizedResult.matched}
            <span className={s.summaryRatioMuted}>/{normalizedResult.total}</span>
          </div>
        </div>

        {shouldShowSummaryCopy ? (
          <div className={s.summaryContent}>
            <div className={s.summaryTitle}>{meta.title}</div>
            <div className={s.summaryText}>{meta.description}</div>
          </div>
        ) : null}
      </section>

      <InspectionResultDetails
        details={normalizedResult.details}
        activeMatchKey={props.activeMatchKey}
        onActiveMatchChange={props.setActiveMatchKey}
      />

      <div className={s.saveDock}>
        <div className={s.saveHead}>
          <span className={s.eyebrow}>{isRealtime ? "Кадр отчёта" : "Отчёт"}</span>
          <span className={s.saveMeta}>{effectiveInspectionId ? "Сохранён" : "Не сохранён"}</span>
        </div>

        {effectiveInspectionId ? (
          <div className={s.saveState}>
            <div className={s.saveStateTitle}>Результат сохранён</div>
            <div className={s.saveStateText}>ID проверки: {effectiveInspectionId}</div>
          </div>
        ) : (
          <>
            <details className={s.noteDetails}>
              <summary className={s.noteSummary}>Комментарий к отчёту</summary>
              <div className={s.noteBody}>
                <Input
                  label="Комментарий к отчёту"
                  placeholder="Причина отклонения или решение оператора"
                  value={notes}
                  onChange={setNotes}
                />
              </div>
            </details>

            {isRealtime ? (
              <div className={s.saveActionsSingle}>
                <Button full disabled={!canSave} onClick={handleSave}>
                  {saveRealtimeMutation.isPending ? "Сохранение..." : "Сохранить кадр"}
                </Button>
              </div>
            ) : (
              <div className={s.saveActions}>
                <Button variant="ghost" disabled={!canDiscard} onClick={handleDiscard}>
                  {discardMutation.isPending ? "Сброс..." : "Сбросить"}
                </Button>
                <Button disabled={!canSave} onClick={handleSave}>
                  {saveMutation.isPending ? "Сохранение..." : "Сохранить"}
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

function normalizePanelResult(props: Props): PanelResult {
  if (props.kind === "realtime") {
    const realtimeResult = props.result;

    return {
      matched: realtimeResult?.matched ?? 0,
      total: realtimeResult?.total ?? 0,
      status: realtimeResult?.status ?? "failed",
      alignmentStatus: realtimeResult?.alignment_status ?? "pending",
      details: realtimeResult?.details ?? [],
      modelName: null,
      isWarmingUp: !realtimeResult || realtimeResult.state === "warming_up",
    };
  }

  return {
    matched: props.result.matched,
    total: props.result.total,
    status: props.result.status,
    alignmentStatus: props.result.alignment_status,
    details: props.result.details,
    modelName: props.result.model_name ?? null,
    isWarmingUp: false,
  };
}

// UI clean v16: keep repeated failed-result copy out of the main summary.
