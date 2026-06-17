import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useCancelTask } from "@/page-components/tasks/api/cancel-task";
import { usePauseTask } from "@/page-components/tasks/api/pause-task";
import { useResumeTask } from "@/page-components/tasks/api/resume-task";
import { isActiveTaskStatus, isTerminalTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import type { MlModel, TaskResponse } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import {
  Activity,
  AlertTriangle,
  ArrowUpDown,
  BadgeCheck,
  Brain,
  CalendarClock,
  CheckCircle2,
  CircleDashed,
  Clock3,
  Database,
  FileJson,
  Filter,
  Link2,
  ListChecks,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  ServerCog,
  SlidersHorizontal,
  Trash2,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import s from "./_training-runs-strict.module.scss";
import { useTrainingModelOutletContext } from "./_training-detail";

type TaskFilter = "all" | "active" | "failed" | "finished" | "cancelled";
type SortMode = "latest" | "oldest" | "progress" | "priority";
type JobBucket = "live" | "failed" | "history";

type StatusTone = "live" | "paused" | "success" | "failed" | "cancelled" | "neutral";

type StatusMeta = {
  label: string;
  tone: StatusTone;
  icon: LucideIcon;
};

const filterLabels: Record<TaskFilter, string> = {
  all: "All",
  active: "Live",
  failed: "Failed",
  finished: "Finished",
  cancelled: "Cancelled",
};

const sortLabels: Record<SortMode, string> = {
  latest: "Latest first",
  oldest: "Oldest first",
  progress: "Progress",
  priority: "Priority",
};

function isLiveTask(task: Pick<TaskResponse, "status">) {
  return isActiveTaskStatus(task.status) || task.status === "paused";
}

function isFinishedTask(task: Pick<TaskResponse, "status">) {
  return isTerminalTaskStatus(task.status) && task.status !== "failed" && task.status !== "cancelled";
}

function statusMeta(status: string): StatusMeta {
  if (status === "succeeded") return { label: "Succeeded", tone: "success", icon: CheckCircle2 };
  if (status === "failed") return { label: "Failed", tone: "failed", icon: XCircle };
  if (status === "cancelled") return { label: "Cancelled", tone: "cancelled", icon: Trash2 };
  if (status === "paused") return { label: "Paused", tone: "paused", icon: Pause };
  if (isActiveTaskStatus(status)) return { label: status, tone: "live", icon: RotateCcw };
  return { label: status || "unknown", tone: "neutral", icon: CircleDashed };
}

function clampProgress(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function taskTitle(task: TaskResponse, model: MlModel | null) {
  if (model) {
    const version = model.version ? ` v${model.version}` : "";
    return `${model.architecture}${version}`;
  }
  return task.type.replaceAll("_", " ");
}

function taskSubtitle(task: TaskResponse, model: MlModel | null) {
  const scope = task.entity_type === "ml_model" ? "model job" : task.entity_type ?? "training job";
  const modelPart = model ? `${model.imgsz}px · ${model.epochs ?? "—"} epochs` : task.queue ?? "queue";
  return `${scope} · ${modelPart}`;
}

function taskMessage(task: TaskResponse) {
  return task.error ?? task.message ?? task.stage ?? task.queue ?? "No job message";
}

function taskBucket(task: TaskResponse): JobBucket {
  if (isLiveTask(task)) return "live";
  if (task.status === "failed") return "failed";
  return "history";
}

function taskModel(task: TaskResponse, models: MlModel[]) {
  if (task.entity_type !== "ml_model" || !task.entity_id) return null;
  return models.find((model) => model.id === task.entity_id) ?? null;
}

function modelScore(model: MlModel | null) {
  const raw = model?.metrics?.mAP50_95 ?? model?.metrics?.["mAP50-95"] ?? model?.metrics?.mAP50;
  if (typeof raw !== "number" || Number.isNaN(raw)) return "—";
  const normalized = raw <= 1 ? raw * 100 : raw;
  return `${Math.round(normalized * 10) / 10}%`;
}

function formatDuration(start: string | null, end: string | null) {
  if (!start) return "—";
  const startMs = Date.parse(start);
  const endMs = end ? Date.parse(end) : Date.now();
  if (Number.isNaN(startMs) || Number.isNaN(endMs) || endMs < startMs) return "—";
  const seconds = Math.round((endMs - startMs) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  const min = minutes % 60;
  return `${hours}h ${min}m`;
}

function compactValue(value: unknown) {
  if (value == null) return "—";
  if (typeof value === "string") return value.length > 96 ? `${value.slice(0, 96)}…` : value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>);
    return `${keys.length} fields`;
  }
  return String(value);
}

function objectPreview(source: Record<string, unknown> | null, limit = 8) {
  if (!source) return [];
  return Object.entries(source).slice(0, limit).map(([key, value]) => ({ key, value: compactValue(value) }));
}

export function Component() {
  const { group, models, tasks } = useTrainingModelOutletContext();
  const cancelMutation = useCancelTask({ groupId: group.id });
  const pauseMutation = usePauseTask({ groupId: group.id });
  const resumeMutation = useResumeTask({ groupId: group.id });

  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<TaskFilter>("all");
  const [sortMode, setSortMode] = useState<SortMode>("latest");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(tasks[0]?.id ?? null);

  const enrichedTasks = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = tasks
      .map((task) => ({ task, model: taskModel(task, models) }))
      .filter(({ task, model }) => {
        const live = isLiveTask(task);
        const matchesFilter =
          filter === "all" ? true :
          filter === "active" ? live :
          filter === "failed" ? task.status === "failed" :
          filter === "cancelled" ? task.status === "cancelled" :
          isFinishedTask(task);

        const haystack = [
          task.id,
          task.type,
          task.status,
          task.stage ?? "",
          task.message ?? "",
          task.error ?? "",
          task.queue ?? "",
          task.entity_type ?? "",
          task.entity_id ?? "",
          task.external_job_id ?? "",
          model?.architecture ?? "",
          model?.version ? `v${model.version}` : "",
        ].join(" ").toLowerCase();

        return matchesFilter && (!needle || haystack.includes(needle));
      });

    return filtered.sort((a, b) => {
      if (sortMode === "oldest") return Date.parse(a.task.created_at) - Date.parse(b.task.created_at);
      if (sortMode === "progress") return clampProgress(b.task.progress_percent) - clampProgress(a.task.progress_percent);
      if (sortMode === "priority") return (b.task.priority ?? 0) - (a.task.priority ?? 0);
      return Date.parse(b.task.created_at) - Date.parse(a.task.created_at);
    });
  }, [filter, models, query, sortMode, tasks]);

  const visibleTasks = enrichedTasks.map(({ task }) => task);
  const selectedTask = visibleTasks.find((task) => task.id === selectedTaskId) ?? visibleTasks[0] ?? tasks.find((task) => task.id === selectedTaskId) ?? null;
  const selectedModel = selectedTask ? taskModel(selectedTask, models) : null;

  const buckets = useMemo(() => {
    const next: Record<JobBucket, Array<{ task: TaskResponse; model: MlModel | null }>> = {
      live: [],
      failed: [],
      history: [],
    };
    for (const item of enrichedTasks) {
      next[taskBucket(item.task)].push(item);
    }
    return next;
  }, [enrichedTasks]);

  const totals = useMemo(() => {
    const live = tasks.filter(isLiveTask).length;
    const failed = tasks.filter((task) => task.status === "failed").length;
    const succeeded = tasks.filter((task) => task.status === "succeeded").length;
    const cancelled = tasks.filter((task) => task.status === "cancelled").length;
    const withModel = tasks.filter((task) => task.entity_type === "ml_model" && task.entity_id).length;
    return { live, failed, succeeded, cancelled, withModel };
  }, [tasks]);

  const renderTask = ({ task, model }: { task: TaskResponse; model: MlModel | null }) => (
    <TaskCard
      key={task.id}
      task={task}
      model={model}
      isSelected={selectedTask?.id === task.id}
      onSelect={() => setSelectedTaskId(task.id)}
      onPause={() => pauseMutation.mutate(task.id)}
      onResume={() => resumeMutation.mutate(task.id)}
      onCancel={() => cancelMutation.mutate(task.id)}
      isPausing={pauseMutation.isPending && pauseMutation.variables === task.id}
      isResuming={resumeMutation.isPending && resumeMutation.variables === task.id}
      isCancelling={cancelMutation.isPending && cancelMutation.variables === task.id}
      groupId={group.id}
    />
  );

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.headerCopy}>
          <span className={s.eyebrow}><Activity /> Train / Runs</span>
          <h1>Jobs registry</h1>
          <p>Очередь обучения, импорта и служебных задач проекта. Метрики модели берутся из backend/model endpoint, а Runs отвечает только за состояние jobs.</p>
        </div>
        <div className={s.headerActions}>
          <Link to={paths.trainingModels(group.id)} className={s.secondaryAction}><Brain /> Models</Link>
          <Link to={paths.trainingGroup(group.id)} className={s.secondaryAction}><ListChecks /> Overview</Link>
        </div>
      </header>

      <section className={s.metricsGrid} aria-label="Training jobs summary">
        <MetricCard icon={Activity} label="Total jobs" value={tasks.length} hint={`${totals.withModel} linked to models`} />
        <MetricCard icon={RotateCcw} label="Live" value={totals.live} hint="running / queued / paused" tone="live" />
        <MetricCard icon={CheckCircle2} label="Succeeded" value={totals.succeeded} hint="completed jobs" tone="success" />
        <MetricCard icon={XCircle} label="Failed" value={totals.failed} hint={`${totals.cancelled} cancelled`} tone="failed" />
      </section>

      <section className={s.workbench}>
        <aside className={s.filterPanel}>
          <div className={s.panelHeader}>
            <span className={s.eyebrow}><Filter /> Filters</span>
            <strong>{enrichedTasks.length} visible</strong>
          </div>

          <label className={s.searchBox}>
            <Search />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by status, model, error" />
          </label>

          <div className={s.filterGroup}>
            <span><SlidersHorizontal /> Status</span>
            {Object.keys(filterLabels).map((key) => {
              const value = key as TaskFilter;
              return (
                <button
                  key={value}
                  type="button"
                  className={filter === value ? s.filterActive : ""}
                  onClick={() => setFilter(value)}
                >
                  {filterLabels[value]}
                </button>
              );
            })}
          </div>

          <div className={s.filterGroup}>
            <span><ArrowUpDown /> Sort</span>
            {Object.keys(sortLabels).map((key) => {
              const value = key as SortMode;
              return (
                <button
                  key={value}
                  type="button"
                  className={sortMode === value ? s.filterActive : ""}
                  onClick={() => setSortMode(value)}
                >
                  {sortLabels[value]}
                </button>
              );
            })}
          </div>
        </aside>

        <main className={s.listPanel}>
          <div className={s.panelHeader}>
            <div>
              <span className={s.eyebrow}><ServerCog /> Job queue</span>
              <h2>{group.name}</h2>
            </div>
            <span className={s.queueBadge}>{totals.live ? `${totals.live} live` : "idle"}</span>
          </div>

          <QueryState
            size="block"
            isEmpty={!enrichedTasks.length}
            emptyTitle="No matching jobs"
            emptyDescription="Измени фильтр или запусти обучение/импорт модели — задачи появятся здесь."
          >
            <div className={s.jobsScroller}>
              <TaskGroup title="Live / paused" icon={RefreshCw} count={buckets.live.length} tone="live">
                {buckets.live.length ? buckets.live.map(renderTask) : <EmptyLine text="Активных задач нет." />}
              </TaskGroup>

              <TaskGroup title="Failed" icon={AlertTriangle} count={buckets.failed.length} tone="failed">
                {buckets.failed.length ? buckets.failed.map(renderTask) : <EmptyLine text="Ошибок в текущем фильтре нет." />}
              </TaskGroup>

              <TaskGroup title="Finished / history" icon={BadgeCheck} count={buckets.history.length}>
                {buckets.history.length ? buckets.history.map(renderTask) : <EmptyLine text="История задач пока пустая." />}
              </TaskGroup>
            </div>
          </QueryState>
        </main>

        <aside className={s.inspector}>
          <div className={s.panelHeader}>
            <span className={s.eyebrow}><FileJson /> Job details</span>
            {selectedTask ? <StatusPill status={selectedTask.status} /> : null}
          </div>

          {selectedTask ? (
            <TaskInspector task={selectedTask} model={selectedModel} groupId={group.id} />
          ) : (
            <div className={s.emptyInspector}>
              <CircleDashed />
              <strong>No job selected</strong>
              <span>Выбери задачу в списке, чтобы посмотреть payload, result, timeline и linked model.</span>
            </div>
          )}
        </aside>
      </section>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, hint, tone = "neutral" }: { icon: LucideIcon; label: string; value: string | number; hint: string; tone?: StatusTone }) {
  return (
    <article className={s.metricCard} data-tone={tone}>
      <Icon />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <small>{hint}</small>
    </article>
  );
}

function StatusPill({ status }: { status: string }) {
  const meta = statusMeta(status);
  const Icon = meta.icon;
  return (
    <span className={s.statusPill} data-tone={meta.tone}>
      <Icon />
      {meta.label}
    </span>
  );
}

function TaskGroup({ title, icon: Icon, count, tone = "neutral", children }: { title: string; icon: LucideIcon; count: number; tone?: StatusTone; children: ReactNode }) {
  return (
    <section className={s.taskGroup} data-tone={tone}>
      <header>
        <span><Icon /> {title}</span>
        <b>{count}</b>
      </header>
      <div className={s.taskGroupBody}>{children}</div>
    </section>
  );
}

function EmptyLine({ text }: { text: string }) {
  return <div className={s.emptyLine}>{text}</div>;
}

function TaskCard({
  task,
  model,
  isSelected,
  onSelect,
  onPause,
  onResume,
  onCancel,
  isPausing,
  isResuming,
  isCancelling,
  groupId,
}: {
  task: TaskResponse;
  model: MlModel | null;
  isSelected: boolean;
  onSelect: () => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  isPausing: boolean;
  isResuming: boolean;
  isCancelling: boolean;
  groupId: string;
}) {
  const progress = clampProgress(task.progress_percent);
  const live = isActiveTaskStatus(task.status);
  const paused = task.status === "paused";
  const meta = statusMeta(task.status);
  const StatusIcon = meta.icon;

  return (
    <article className={`${s.taskCard} ${isSelected ? s.taskCardSelected : ""}`} data-tone={meta.tone}>
      <button type="button" className={s.taskSelect} onClick={onSelect} aria-label={`Open job ${task.id}`}>
        <span className={s.taskIcon}><StatusIcon /></span>
        <span className={s.taskMain}>
          <strong>{taskTitle(task, model)}</strong>
          <span>{taskSubtitle(task, model)}</span>
          <small>{taskMessage(task)}</small>
        </span>
        <span className={s.taskSide}>
          <StatusPill status={task.status} />
          <b>{progress}%</b>
        </span>
      </button>

      <div className={s.progressTrack} aria-hidden="true">
        <span style={{ width: `${progress}%` }} />
      </div>

      <footer className={s.taskFooter}>
        <span><CalendarClock /> {formatDate(task.created_at)}</span>
        {model ? <Link to={paths.trainingModel(groupId, model.id)}><Link2 /> Model</Link> : <span><Database /> {task.queue ?? "queue"}</span>}
        <div className={s.taskActions}>
          {live ? <button type="button" onClick={onPause} disabled={isPausing}><Pause /> Pause</button> : null}
          {paused ? <button type="button" onClick={onResume} disabled={isResuming}><Play /> Resume</button> : null}
          {(live || paused) ? <button type="button" className={s.dangerButton} onClick={onCancel} disabled={isCancelling}><Trash2 /> Cancel</button> : null}
        </div>
      </footer>
    </article>
  );
}

function TaskInspector({ task, model, groupId }: { task: TaskResponse; model: MlModel | null; groupId: string }) {
  const progress = clampProgress(task.progress_percent);
  const end = task.finished_at ?? task.cancelled_at;
  const payload = objectPreview(task.payload);
  const result = objectPreview(task.result);
  const hasLiveMetrics = !!task.result && typeof task.result.live_metrics === "object" && task.result.live_metrics !== null;

  return (
    <div className={s.inspectorScroll}>
      <section className={s.inspectorHero}>
        <div className={s.inspectorIcon}><Activity /></div>
        <div>
          <strong>{taskTitle(task, model)}</strong>
          <span>{task.type} · {task.stage ?? "queued"}</span>
        </div>
      </section>

      <div className={s.progressLarge}>
        <div>
          <span>Progress</span>
          <b>{progress}%</b>
        </div>
        <div className={s.progressTrack}><span style={{ width: `${progress}%` }} /></div>
      </div>

      <section className={s.detailGrid}>
        <DetailItem label="Created" value={formatDate(task.created_at)} />
        <DetailItem label="Started" value={task.started_at ? formatDate(task.started_at) : "—"} />
        <DetailItem label="Finished" value={end ? formatDate(end) : "—"} />
        <DetailItem label="Duration" value={formatDuration(task.started_at, end)} />
        <DetailItem label="Queue" value={task.queue ?? "—"} />
        <DetailItem label="Priority" value={task.priority} />
        <DetailItem label="External job" value={task.external_job_id ?? "—"} />
        <DetailItem label="Live metrics" value={hasLiveMetrics ? "available" : "—"} />
      </section>

      {model ? (
        <Link to={paths.trainingModel(groupId, model.id)} className={s.linkedModel}>
          <Brain />
          <div>
            <strong>{model.architecture}{model.version ? ` v${model.version}` : ""}</strong>
            <span>{model.imgsz}px · score {modelScore(model)} · {model.num_classes ?? "—"} classes</span>
          </div>
        </Link>
      ) : null}

      {(task.error || task.message) ? (
        <section className={s.messageBox} data-error={task.error ? "true" : "false"}>
          {task.error ? <AlertTriangle /> : <Clock3 />}
          <div>
            <strong>{task.error ? "Error" : "Message"}</strong>
            <span>{task.error ?? task.message}</span>
          </div>
        </section>
      ) : null}

      <PreviewSection title="Payload" rows={payload} empty="No payload fields" />
      <PreviewSection title="Result" rows={result} empty="No result fields" />
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string | number }) {
  return (
    <div className={s.detailItem}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PreviewSection({ title, rows, empty }: { title: string; rows: Array<{ key: string; value: string }>; empty: string }) {
  return (
    <section className={s.previewSection}>
      <header>
        <span><FileJson /> {title}</span>
        <b>{rows.length}</b>
      </header>
      {rows.length ? (
        <dl>
          {rows.map((row) => (
            <div key={row.key}>
              <dt>{row.key}</dt>
              <dd>{row.value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p>{empty}</p>
      )}
    </section>
  );
}
