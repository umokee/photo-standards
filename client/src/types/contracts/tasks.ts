export interface TaskResponse {
  id: string;
  type: string;
  status: string;

  queue: string | null;
  priority: number;

  progress_current: number | null;
  progress_total: number | null;
  progress_percent: number | null;

  stage: string | null;
  message: string | null;
  error: string | null;

  payload: Record<string, unknown> | null;
  result: Record<string, unknown> | null;

  entity_type: string | null;
  entity_id: string | null;
  group_id: string | null;

  external_job_id: string | null;

  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  cancelled_at: string | null;
}

export interface TrainingStartResponse {
  task_id: string;
  model_id: string;
}
