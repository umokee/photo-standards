import { Badge } from "@/components/ui/badge/badge";
import Button from "@/components/ui/button/button";
import { InfoRow } from "@/components/ui/info-row/info-row";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import type { MlModel, TaskResponse, TrainingStatus } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { architectureLabel, metricLabel, trainingStatusLabel } from "@/utils/labels";
import clsx from "clsx";
import { ChevronRight } from "lucide-react";
import { useEffect, useRef } from "react";
import {
  formatModelMetric,
  getDisplayMetrics,
  getModelClassLabels,
  getModelVersionLabel,
  getTrainingPercent,
  getTrainingStageLabel,
  getTrainingStatus,
  MODEL_METRIC_KEYS,
} from "../../lib/model-helpers";
import s from "./model-card.module.scss";

type Props = {
  model: MlModel;
  task?: TaskResponse | null;
  expanded: boolean;
  onToggle: () => void;
  onActivate?: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  onPause?: (taskId: string) => void;
  onResume?: (taskId: string) => void;
  onCancel?: (taskId: string) => void;
  isActivating?: boolean;
  isDeleting?: boolean;
  isPausing?: boolean;
  isResuming?: boolean;
  isCancelling?: boolean;
};

const statusBadgeTypeMap: Record<TrainingStatus, "info" | "success" | "warning" | "danger"> = {
  pending: "warning",
  preparing: "info",
  training: "info",
  saving: "info",
  paused: "warning",
  done: "success",
  failed: "danger",
};

export const ModelCard = ({
  model,
  task = null,
  expanded,
  onToggle,
  onActivate,
  onDelete,
  onPause,
  onResume,
  onCancel,
  isActivating = false,
  isDeleting = false,
  isPausing = false,
  isResuming = false,
  isCancelling = false,
}: Props) => {
  const status = getTrainingStatus(model, task);
  const progress = getTrainingPercent(model, task);
  const displayMetrics = getDisplayMetrics(model, task);
  const versionLabel = getModelVersionLabel(model);
  const classLabels = getModelClassLabels(model);

  const statusBadge = model.is_active
    ? { type: "success" as const, label: "Активна" }
    : { type: statusBadgeTypeMap[status] ?? "info", label: trainingStatusLabel(status) };

  const trainingStageLabel = getTrainingStageLabel(model, task, trainingStatusLabel(status));

  const isRunningTraining =
    status === "pending" || status === "preparing" || status === "training" || status === "saving";

  const hasRealProgress = status === "training" || status === "saving";
  const trainingProgressLabel = hasRealProgress ? `${progress}%` : trainingStatusLabel(status);

  const metaParameters = [
    ...(model.trained_at ? [{ label: "Обучена", value: formatDate(model.trained_at) }] : []),
    { label: "Классы", value: String(model.num_classes ?? classLabels.length) },
    { label: "mAP50", value: formatModelMetric(displayMetrics?.mAP50) },
    { label: "Recall", value: formatModelMetric(displayMetrics?.recall) },
  ];

  const statusMessage =
    status === "failed"
      ? (task?.error ?? "Ошибка обучения")
      : status === "paused" || isRunningTraining
        ? trainingStageLabel
        : null;

  const statusValue = isRunningTraining ? trainingProgressLabel : null;

  const rootRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!expanded) return;

    requestAnimationFrame(() => {
      rootRef.current?.scrollIntoView({
        behavior: "instant",
        block: "nearest",
      });
    });
  }, [expanded]);

  return (
    <article ref={rootRef} className={clsx(s.root, expanded && s.expanded)}>
      <div className={s.header} onClick={onToggle}>
        <div className={s.titleRow}>
          <div className={s.info}>
            <span className={s.title}>
              {architectureLabel(model.architecture)} &middot; {versionLabel}
            </span>
            <div className={s.meta}>
              {metaParameters.map(({ label, value }) => (
                <span key={label}>
                  {label} <strong>{value}</strong>
                </span>
              ))}
            </div>
          </div>
          <div className={s.side}>
            <Badge type={statusBadge.type}>{statusBadge.label}</Badge>
            <ChevronRight className={s.chevron} size={16} />
          </div>
        </div>

        {statusMessage && (
          <div className={clsx(s.headerStatus, status === "failed" && s.error)}>
            <div className={s.meta}>
              <span>{statusMessage}</span>
              {statusValue && <strong>{statusValue}</strong>}
            </div>

            {isRunningTraining && (
              <div className={s.progress}>
                <div className={s.fill} style={{ width: `${progress}%` }} />
              </div>
            )}
          </div>
        )}
      </div>

      {expanded && (
        <div className={s.body}>
          <ModelCardDetail
            model={model}
            task={task}
            onActivate={onActivate}
            onDelete={onDelete}
            onPause={onPause}
            onResume={onResume}
            onCancel={onCancel}
            isActivating={isActivating}
            isDeleting={isDeleting}
            isPausing={isPausing}
            isResuming={isResuming}
            isCancelling={isCancelling}
          />
        </div>
      )}
    </article>
  );
};

interface DetailProps {
  model: MlModel;
  task?: TaskResponse | null;
  onActivate?: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  onPause?: (taskId: string) => void;
  onResume?: (taskId: string) => void;
  onCancel?: (taskId: string) => void;
  isActivating?: boolean;
  isDeleting?: boolean;
  isPausing?: boolean;
  isResuming?: boolean;
  isCancelling?: boolean;
}

const ModelCardDetail = ({
  model,
  task,
  onActivate,
  onDelete,
  onPause,
  onResume,
  onCancel,
  isActivating = false,
  isDeleting = false,
  isPausing = false,
  isResuming = false,
  isCancelling = false,
}: DetailProps) => {
  const displayMetrics = getDisplayMetrics(model, task ?? null);
  const taskStatus = task?.status ?? null;

  const hasOpenTrainingTask =
    taskStatus !== null &&
    taskStatus !== "succeeded" &&
    taskStatus !== "failed" &&
    taskStatus !== "cancelled";

  const canActivate =
    !!onActivate &&
    !model.is_active &&
    !!model.trained_at &&
    model.version !== null &&
    !hasOpenTrainingTask;

  const canDelete = !!onDelete && !model.is_active && !hasOpenTrainingTask;

  const canPause =
    !!onPause &&
    !!task?.id &&
    (taskStatus === "pending" ||
      taskStatus === "queued" ||
      taskStatus === "running" ||
      taskStatus === "resuming");

  const canResume = !!onResume && !!task?.id && taskStatus === "paused";

  const canCancel = !!onCancel && !!task?.id && hasOpenTrainingTask;

  const hasActions = canActivate || canDelete || canPause || canResume || canCancel;

  const modelClassLabels = getModelClassLabels(model);

  const parameterRows = [
    { label: "Архитектура", value: architectureLabel(model.architecture) },
    { label: "Эпох", value: model.epochs ?? "n/a" },
    { label: "Размер изображения", value: model.imgsz },
    { label: "Batch size", value: model.batch_size ?? "n/a" },
  ];

  const datasetRows = [
    { label: "Всего изображений", value: model.total_images ?? "n/a" },
    { label: "Train", value: formatDatasetSplit(model.train_ratio, model.train_count) },
    { label: "Val", value: formatDatasetSplit(model.val_ratio, model.val_count) },
    { label: "Test", value: formatDatasetSplit(model.test_ratio, model.test_count) },
  ];

  return (
    <div className={s.detail}>
      <div className={s.detailGrid}>
        <SurfaceSection title="Датасет">
          {datasetRows.map((item) => (
            <InfoRow key={item.label} label={item.label} value={item.value} />
          ))}
        </SurfaceSection>

        <SurfaceSection title="Параметры">
          {parameterRows.map((item) => (
            <InfoRow key={item.label} label={item.label} value={item.value} />
          ))}
        </SurfaceSection>

        <SurfaceSection title="Метрики">
          {MODEL_METRIC_KEYS.map((key) => (
            <InfoRow
              key={key}
              label={metricLabel(key)}
              value={formatModelMetric(displayMetrics?.[key])}
            />
          ))}
        </SurfaceSection>
      </div>

      {taskStatus === "failed" && task?.error && (
        <SurfaceSection title="Ошибка обучения">
          <span className={s.emptyText}>{task.error}</span>
        </SurfaceSection>
      )}

      {!!modelClassLabels.length && (
        <SurfaceSection title="Классы модели" direction="row" transparent>
          {modelClassLabels.map((className) => (
            <Badge key={className}>{className}</Badge>
          ))}
        </SurfaceSection>
      )}

      {hasActions && (
        <SurfaceSection title="Управление" transparent>
          <div className={s.actions}>
            {canResume && task?.id && (
              <Button
                full
                size="sm"
                variant="ml"
                disabled={isResuming}
                onClick={() => onResume?.(task.id)}
              >
                {isResuming ? "Продолжаем..." : "Продолжить"}
              </Button>
            )}

            {canPause && task?.id && (
              <Button
                full
                size="sm"
                variant="ghost"
                disabled={isPausing}
                onClick={() => onPause?.(task.id)}
              >
                {isPausing ? "Пауза..." : "Пауза"}
              </Button>
            )}

            {canCancel && task?.id && (
              <Button
                full
                size="sm"
                variant="ghost"
                disabled={isCancelling}
                onClick={() => onCancel?.(task.id)}
              >
                {isCancelling ? "Отменяем..." : "Отменить"}
              </Button>
            )}

            {canActivate && (
              <Button
                full
                size="sm"
                disabled={isActivating}
                onClick={() => onActivate?.(model.id)}
              >
                {isActivating ? "Активация..." : "Сделать активной"}
              </Button>
            )}

            {canDelete && (
              <Button
                full
                size="sm"
                variant="danger"
                disabled={isDeleting}
                onClick={() => onDelete?.(model.id)}
              >
                {isDeleting ? "Удаление..." : "Удалить"}
              </Button>
            )}
          </div>
        </SurfaceSection>
      )}
    </div>
  );
};

const formatDatasetSplit = (ratio: number | null, count: number | null) => {
  if (ratio === null || ratio === undefined) {
    return "n/a";
  }

  if (count === null || count === undefined) {
    return `${ratio}%`;
  }

  return `${ratio}% (${count})`;
};
