import QueryState from "@/components/ui/query-state/query-state";
import { useCancelTask } from "@/page-components/tasks/api/cancel-task";
import { usePauseTask } from "@/page-components/tasks/api/pause-task";
import { useResumeTask } from "@/page-components/tasks/api/resume-task";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import type { TaskResponse } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Activity, CheckCircle2, CircleDashed, Filter, Pause, Play, RotateCcw, Search, Trash2, XCircle } from "lucide-react";
import { useMemo, useState } from "react";
import p from "../platform-pages.module.scss";
import { useTrainingModelOutletContext } from "./_training-detail";

type TaskFilter = "all" | "active" | "failed" | "finished";

export function Component() {
  const { group, tasks } = useTrainingModelOutletContext();
  const cancelMutation = useCancelTask({ groupId: group.id });
  const pauseMutation = usePauseTask({ groupId: group.id });
  const resumeMutation = useResumeTask({ groupId: group.id });
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<TaskFilter>("all");

  const filteredTasks = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return tasks.filter((task) => {
      const isActive = isActiveTaskStatus(task.status) || task.status === "paused";
      const isFailed = task.status === "failed";
      const isFinished = !isActive && !isFailed;
      const matchesFilter =
        filter === "all" ? true :
        filter === "active" ? isActive :
        filter === "failed" ? isFailed :
        isFinished;
      const matchesQuery = !needle || [task.type, task.status, task.stage ?? "", task.message ?? "", task.error ?? "", task.queue ?? ""].some((value) => value.toLowerCase().includes(needle));
      return matchesFilter && matchesQuery;
    });
  }, [filter, query, tasks]);

  const activeTasks = filteredTasks.filter((task) => isActiveTaskStatus(task.status) || task.status === "paused");
  const failedTasks = filteredTasks.filter((task) => task.status === "failed");
  const finishedTasks = filteredTasks.filter((task) => task.status !== "failed" && !isActiveTaskStatus(task.status) && task.status !== "paused");
  const activeTotal = tasks.filter((task) => isActiveTaskStatus(task.status) || task.status === "paused").length;
  const failedTotal = tasks.filter((task) => task.status === "failed").length;
  const succeededTotal = tasks.filter((task) => task.status === "succeeded").length;

  const renderTask = (task: TaskResponse) => (
    <TaskRow
      key={task.id}
      task={task}
      onPause={() => pauseMutation.mutate(task.id)}
      onResume={() => resumeMutation.mutate(task.id)}
      onCancel={() => cancelMutation.mutate(task.id)}
      isPausing={pauseMutation.isPending && pauseMutation.variables === task.id}
      isResuming={resumeMutation.isPending && resumeMutation.variables === task.id}
      isCancelling={cancelMutation.isPending && cancelMutation.variables === task.id}
    />
  );

  return (
    <div className={`${p.page} ${p.trainPageV27} ${p.trainRunsScreenV29}`}>
      <header className={`${p.trainHeaderV27} ${p.trainHeaderCompactV27} ${p.trainHeaderSlimV29}`}>
        <div>
          <span className={p.eyebrow}><Activity /> Train / Runs</span>
          <h1>Training jobs</h1>
          <p>Runs отвечает только за jobs: train/import/export, статус, прогресс, ошибки и управление активными задачами.</p>
        </div>
      </header>

      <section className={p.trainMetricGridV27}>
        <Metric value={tasks.length} label="Total" />
        <Metric value={activeTotal} label="Active" />
        <Metric value={succeededTotal} label="Succeeded" />
        <Metric value={failedTotal} label="Failed" />
      </section>

      <section className={p.trainRunsWorkbenchV29}>
        <div className={p.trainToolbarV29}>
          <label className={p.searchBoxV29}>
            <Search />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search jobs" />
          </label>
          <div className={p.filterPillsV29}>
            <Filter />
            {(["all", "active", "failed", "finished"] as TaskFilter[]).map((value) => (
              <button key={value} type="button" className={filter === value ? p.filterPillActiveV29 : ""} onClick={() => setFilter(value)}>{value}</button>
            ))}
          </div>
        </div>

        <QueryState
          size="block"
          isEmpty={!filteredTasks.length}
          emptyTitle="No training jobs"
          emptyDescription="Запусти обучение или импорт модели — задачи появятся здесь."
        >
          <div className={p.trainRunsScrollerV29}>
            <details className={p.collapsiblePanelV28} open>
              <summary>
                <span><RotateCcw /> Active / paused</span>
                <b>{activeTasks.length}</b>
              </summary>
              <div className={`${p.trainRunsListV27} ${p.scrollListV28}`}>
                {activeTasks.length ? activeTasks.map(renderTask) : <EmptyLine text="Активных задач нет." />}
              </div>
            </details>

            <details className={p.collapsiblePanelV28} open>
              <summary>
                <span><XCircle /> Failed</span>
                <b>{failedTasks.length}</b>
              </summary>
              <div className={`${p.trainRunsListV27} ${p.scrollListV28}`}>
                {failedTasks.length ? failedTasks.map(renderTask) : <EmptyLine text="Ошибок обучения нет." />}
              </div>
            </details>

            <details className={p.collapsiblePanelV28}>
              <summary>
                <span><CheckCircle2 /> Finished / history</span>
                <b>{finishedTasks.length}</b>
              </summary>
              <div className={`${p.trainRunsListV27} ${p.scrollListV28}`}>
                {finishedTasks.length ? finishedTasks.map(renderTask) : <EmptyLine text="История задач пока пустая." />}
              </div>
            </details>
          </div>
        </QueryState>
      </section>
    </div>
  );
}

function Metric({ value, label }: { value: string | number; label: string }) {
  return (
    <div className={p.trainMetricV27}>
      <Activity />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>job queue</small>
    </div>
  );
}

function EmptyLine({ text }: { text: string }) {
  return <div className={p.emptyLineV28}>{text}</div>;
}

function TaskRow({
  task,
  onPause,
  onResume,
  onCancel,
  isPausing,
  isResuming,
  isCancelling,
}: {
  task: TaskResponse;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  isPausing: boolean;
  isResuming: boolean;
  isCancelling: boolean;
}) {
  const progress = task.progress_percent ?? 0;
  const isActive = isActiveTaskStatus(task.status);
  const isPaused = task.status === "paused";
  const Icon = task.status === "succeeded" ? CheckCircle2 : task.status === "failed" ? XCircle : isActive ? RotateCcw : CircleDashed;

  return (
    <article className={p.trainRunRowV27}>
      <div className={p.trainModelIconV27}><Icon /></div>
      <div className={p.trainRunMetaV27}>
        <strong>{task.type}</strong>
        <span>{task.status} · {task.stage ?? "queued"} · {formatDate(task.created_at)}</span>
        <small>{task.message ?? task.error ?? task.queue ?? "Training job"}</small>
        <div className={p.trainProgressV27}><span style={{ width: `${progress}%` }} /></div>
      </div>
      <b>{progress}%</b>
      <div className={p.trainTaskActionsV27}>
        {isActive ? <button type="button" onClick={onPause} disabled={isPausing}><Pause /> Pause</button> : null}
        {isPaused ? <button type="button" onClick={onResume} disabled={isResuming}><Play /> Resume</button> : null}
        {(isActive || isPaused) ? <button type="button" onClick={onCancel} disabled={isCancelling}><Trash2 /> Cancel</button> : null}
      </div>
    </article>
  );
}
