import { Badge } from "@/components/ui/badge/badge";
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
      title: "Проверка запускается",
      text: "Как только придёт первый обработанный кадр, список классов начнёт обновляться в реальном времени.",
    };
  }

  if (isAlignmentFailed(result)) {
    return {
      badgeType: "danger" as const,
      badgeLabel: "Не пройдено",
      title: "Не удалось сопоставить кадр",
      text: "Кадр слишком отличается по геометрии от эталона. Переснимите ближе к ракурсу эталонного снимка.",
    };
  }

  const issueCount = Math.max(result.total - result.matched, 0);

  if (result.status === "passed") {
    return {
      badgeType: "success" as const,
      badgeLabel: "Пройдено",
      title: "Проверка пройдена",
      text: `На месте все ожидаемые сегменты: ${result.total} из ${result.total}.`,
    };
  }

  return {
    badgeType: "danger" as const,
    badgeLabel: "Не пройдено",
    title: "Есть расхождения",
    text: `На месте ${result.matched} из ${result.total}. Требуют внимания: ${issueCount}.`,
  };
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
      <div className={s.summary}>
        <div className={s.summaryHead}>
          <span className={s.eyebrow}>Результат проверки</span>
          <Badge type={meta.badgeType}>{meta.badgeLabel}</Badge>
        </div>

        <div className={s.summaryBody}>
          <div className={s.summaryValue}>
            {normalizedResult.matched}
            <span className={s.summaryValueMuted}> / {normalizedResult.total}</span>
          </div>

          <div className={s.summaryContent}>
            <div className={s.summaryTitle}>{meta.title}</div>
            <div className={s.summaryText}>{meta.text}</div>
            {!!normalizedResult.modelName && (
              <div className={s.summaryHint}>Модель: {normalizedResult.modelName}</div>
            )}
          </div>
        </div>
      </div>

      <InspectionResultDetails
        details={normalizedResult.details}
        activeMatchKey={props.activeMatchKey}
        onActiveMatchChange={props.setActiveMatchKey}
      />

      <div className={s.saveDock}>
        <div className={s.saveHead}>
          <span className={s.eyebrow}>
            {isRealtime ? "Сохранение текущего кадра" : "Сохранение результата"}
          </span>
        </div>

        <div className={s.saveBody}>
          {effectiveInspectionId ? (
            <div className={s.saveState}>
              <div className={s.saveStateTitle}>Результат уже сохранён</div>
              <div className={s.saveStateText}>ID проверки: {effectiveInspectionId}</div>
            </div>
          ) : (
            <>
              <Input
                label="Комментарий"
                placeholder="Дополнительные заметки"
                value={notes}
                onChange={setNotes}
              />
              {isRealtime ? (
                <div className={s.saveActionsSingle}>
                  <Button full disabled={!canSave} onClick={handleSave}>
                    {saveRealtimeMutation.isPending ? "Сохранение..." : "Сохранить"}
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
