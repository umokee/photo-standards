import { getTaskLiveMetrics, isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import {
  type Metric,
  type MlModel,
  type TaskResponse,
  type TrainingStatus,
} from "@/types/contracts";

export type DisplayMetrics = Partial<Record<Metric, number | null>>;
export const MODEL_METRIC_KEYS: Metric[] = ["mAP50", "mAP50_95", "precision", "recall"];

const ACTIVE_TRAINING_STATUSES = new Set<TrainingStatus>([
  "pending",
  "preparing",
  "training",
  "saving",
]);

const isMetricValue = (value: unknown): value is number | null => {
  return typeof value === "number" || value === null;
};

export const getModelTask = (model: MlModel, tasks: TaskResponse[]): TaskResponse | null => {
  return (
    tasks.find((task) => task.entity_type === "ml_model" && task.entity_id === model.id) ?? null
  );
};

export const getTrainingStatus = (model: MlModel, task: TaskResponse | null): TrainingStatus => {
  if (task) {
    if (task.status === "pending" || task.status === "queued") return "pending";
    if (task.status === "paused") return "paused";

    if (isActiveTaskStatus(task.status)) {
      const stage = task.stage?.toLowerCase() ?? "";

      if (stage.includes("подготов")) return "preparing";
      if (stage.includes("сохран")) return "saving";

      return "training";
    }

    if (task.status === "failed" || task.status === "cancelled") {
      return "failed";
    }

    if (task.status === "succeeded") return "done";
  }

  return model.trained_at ? "done" : "pending";
};

export const getTrainingPercent = (model: MlModel, task: TaskResponse | null): number => {
  if (!task) return model.trained_at ? 100 : 0;
  if (task.progress_percent != null) return task.progress_percent;

  if (task.progress_current != null && model.epochs && model.epochs > 0) {
    return Math.min(100, Math.round((task.progress_current / model.epochs) * 100));
  }

  if (task.status === "succeeded") return 100;

  return 0;
};

export const isTrainingModel = (model: MlModel, task: TaskResponse | null): boolean => {
  return ACTIVE_TRAINING_STATUSES.has(getTrainingStatus(model, task));
};

export const isTrainingFailedModel = (model: MlModel, task: TaskResponse | null): boolean => {
  return getTrainingStatus(model, task) === "failed";
};

export const getModelVersionLabel = (model: MlModel): string => {
  return model.version !== null ? `v${model.version}` : "без версии";
};

export const getModelClassLabels = (model: MlModel): string[] => {
  return model.class_meta?.map((item) => item.name) ?? model.class_keys ?? [];
};

export const formatModelMetric = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "n/a";
  return value.toFixed(3);
};

export const getDisplayMetrics = (
  model: MlModel,
  task: TaskResponse | null
): DisplayMetrics | null => {
  const liveMetrics = isTrainingModel(model, task) ? getTaskLiveMetrics(task) : null;
  if (liveMetrics) {
    return liveMetrics;
  }

  if (!model.metrics) {
    return null;
  }

  const source = model.metrics as Record<string, unknown>;
  const metrics: DisplayMetrics = {};

  for (const key of MODEL_METRIC_KEYS) {
    const value = source[key];
    if (isMetricValue(value)) {
      metrics[key] = value;
    }
  }

  return Object.keys(metrics).length > 0 ? metrics : null;
};

export const getTrainingStageLabel = (
  model: MlModel,
  task: TaskResponse | null,
  fallback: string
): string => {
  const status = getTrainingStatus(model, task);
  return status === "pending" ? "Ожидает запуска" : (task?.stage ?? fallback);
};
