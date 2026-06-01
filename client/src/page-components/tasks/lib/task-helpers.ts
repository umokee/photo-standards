import { Metric, TaskResponse } from "@/types/contracts";

export type LiveTrainingMetrics = Partial<Record<Metric, number | null>>;

export const ACTIVE_TASK_STATUSES = new Set<string>([
  "pending",
  "queued",
  "running",
  "resuming",
  "pausing",
]);

export const TERMINAL_TASK_STATUSES = new Set<string>(["succeeded", "failed", "cancelled"]);

const TRAINING_METRIC_KEYS: Metric[] = ["mAP50", "mAP50_95", "precision", "recall"];

const isMetricValue = (value: unknown): value is number | null => {
  return typeof value === "number" || value === null;
};

export const isActiveTaskStatus = (status?: string | null) =>
  status != null && ACTIVE_TASK_STATUSES.has(status);

export const isTerminalTaskStatus = (status?: string | null) =>
  status != null && TERMINAL_TASK_STATUSES.has(status);

export const isTaskCancelled = (task?: Pick<TaskResponse, "status"> | null) =>
  task?.status === "cancelled";

export const isTaskPaused = (task?: Pick<TaskResponse, "status"> | null) =>
  task?.status === "paused";

export const getTaskProgressPercent = (task?: Pick<TaskResponse, "progress_percent"> | null) =>
  task?.progress_percent ?? 0;

export const getTaskStageLabel = (
  task?: Pick<TaskResponse, "stage" | "message"> | null,
  fallback = ""
) => {
  return task?.stage ?? task?.message ?? fallback;
};

export const getTaskLiveMetrics = (task?: TaskResponse | null): LiveTrainingMetrics | null => {
  const raw = task?.result?.live_metrics;

  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }

  const source = raw as Record<string, unknown>;
  const metrics: LiveTrainingMetrics = {};

  for (const key of TRAINING_METRIC_KEYS) {
    const value = source[key];
    if (isMetricValue(value)) {
      metrics[key] = value;
    }
  }

  return Object.keys(metrics).length > 0 ? metrics : null;
};
