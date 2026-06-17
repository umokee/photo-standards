import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import { useGetModels } from "@/page-components/models/api/get-models";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { useGetTasks } from "@/page-components/tasks/api/get-tasks";
import { useTasksLive } from "@/page-components/tasks/hooks/use-task-live";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import type { GroupDetail, MlModel, TaskResponse } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import clsx from "clsx";
import { Activity, Brain, Clock3, Database, Image, Layers3, ListChecks, ShieldCheck, Sparkles } from "lucide-react";
import { Link, Outlet, useLoaderData, useLocation, useOutletContext } from "react-router-dom";
import p from "../platform-pages.module.scss";

type TrainingModelOutletContext = { group: GroupDetail; models: MlModel[]; tasks: TaskResponse[] };
export const useTrainingModelOutletContext = () => useOutletContext<TrainingModelOutletContext>();

const metricNames = ["mAP50", "mAP50_95", "precision", "recall"] as const;

function formatMetric(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "—";
  return value <= 1 ? `${Math.round(value * 100)}%` : String(Math.round(value * 100) / 100);
}

function taskProgress(task: TaskResponse) {
  if (task.progress_percent != null) return `${Math.round(task.progress_percent)}%`;
  if (task.progress_current != null && task.progress_total) return `${task.progress_current}/${task.progress_total}`;
  return task.stage || task.status;
}

export function Component() {
  const { groupId } = useLoaderData() as { groupId: string };
  const location = useLocation();
  const { data: group } = useGetGroup(groupId);
  const { data: models } = useGetModels(groupId);
  const { data: tasks } = useGetTasks(groupId);

  const activeTrainingTaskIds = tasks.filter((task) => isActiveTaskStatus(task.status)).map((task) => task.id);
  useTasksLive({ taskIds: activeTrainingTaskIds, groupId });

  const activeModel = models.find((model) => model.is_active) ?? models[0] ?? null;
  const latestTask = tasks[0] ?? null;
  const labeledPercent = group.stats.images_count ? Math.round((group.stats.annotated_images_count / group.stats.images_count) * 100) : 0;
  const hasMinimumTrainingData = group.stats.standards_count > 0 && group.stats.images_count > 0 && group.stats.annotated_images_count > 0 && group.stats.segment_classes_count > 0;
  const hasActiveTrainingTask = tasks.some((task) => isActiveTaskStatus(task.status));

  return (
    <div className={p.page}>
      <section className={p.trainWorkspaceHero}>
        <div className={p.datasetAvatarLarge}>{group.name.slice(0, 1).toUpperCase()}</div>
        <div className={p.trainWorkspaceTitle}>
          <span className={p.eyebrow}><Brain /> Model workspace</span>
          <h1>{group.name}</h1>
          <p>{group.description || "Контроль готовности dataset, обучение и выбор модели для Inspect."}</p>
          <div className={p.metaLine}>
            <span><Image /> {group.stats.annotated_images_count}/{group.stats.images_count} labeled</span>
            <span><Layers3 /> {group.stats.segment_classes_count} classes</span>
            <span><Brain /> {models.length} models</span>
            <span>Created {formatDate(group.created_at)}</span>
          </div>
        </div>
        <div className={p.headerActions}>
          <ImportModel groupId={group.id} />
          <ExportModel models={models} />
          <TrainModel groupId={group.id} canTrain={hasMinimumTrainingData} isTrainingLocked={hasActiveTrainingTask} />
        </div>
      </section>

      <div className={p.pipelineStrip}>
        <div className={clsx(p.pipelineStep, group.stats.images_count > 0 && p.pipelineStepReady)}><span>1</span><b>Data</b><small>{group.stats.images_count} images</small></div>
        <div className={clsx(p.pipelineStep, group.stats.segment_classes_count > 0 && p.pipelineStepReady)}><span>2</span><b>Classes</b><small>{group.stats.segment_classes_count} labels</small></div>
        <div className={clsx(p.pipelineStep, group.stats.annotated_images_count > 0 && p.pipelineStepReady)}><span>3</span><b>Annotations</b><small>{labeledPercent}% labeled</small></div>
        <div className={clsx(p.pipelineStep, models.length > 0 && p.pipelineStepReady)}><span>4</span><b>Model</b><small>{activeModel ? "ready" : "not trained"}</small></div>
      </div>

      <div className={p.trainDetailGrid}>
        <aside className={p.trainModelRail}>
          <div className={p.railHeader}>
            <div>
              <h3>Models</h3>
              <p>Weights and versions</p>
            </div>
            <span>{models.length}</span>
          </div>
          <QueryState isEmpty={!models.length} size="block" emptyTitle="No models" emptyDescription="Train or import the first model.">
            <div className={p.modelRailList}>
              {models.map((model) => {
                const title = `${model.architecture}${model.version ? ` v${model.version}` : ""}`;
                const href = paths.trainingModel(group.id, model.id);
                const selected = location.pathname === href;
                const map50 = model.metrics?.mAP50 ?? model.metrics?.["mAP50"];

                return (
                  <Link className={clsx(p.modelRailItem, selected && p.modelRailItemActive)} key={model.id} to={href}>
                    <span className={p.modelIcon}><Brain /></span>
                    <div>
                      <strong>{title}</strong>
                      <small>{model.imgsz}px · {model.num_classes ?? "—"} classes · {formatMetric(map50 as number | null | undefined)}</small>
                    </div>
                    {model.is_active ? <b>Active</b> : null}
                  </Link>
                );
              })}
            </div>
          </QueryState>
        </aside>

        <main className={p.trainMainColumn}>
          <div className={p.trainCardsRow}>
            <ReadinessCard icon={Database} label="Dataset" value={`${group.stats.images_count}`} hint="images" ready={group.stats.images_count > 0} />
            <ReadinessCard icon={Sparkles} label="Labeled" value={`${labeledPercent}%`} hint={`${group.stats.annotated_images_count} annotated`} ready={group.stats.annotated_images_count > 0} />
            <ReadinessCard icon={Layers3} label="Classes" value={`${group.stats.segment_classes_count}`} hint="segment classes" ready={group.stats.segment_classes_count > 0} />
            <ReadinessCard icon={ShieldCheck} label="Status" value={hasMinimumTrainingData ? "Ready" : "Setup"} hint={hasActiveTrainingTask ? "training running" : "training gate"} ready={hasMinimumTrainingData} />
          </div>

          <div className={p.trainOverviewGrid}>
            <section className={p.panelCard}>
              <div className={p.cardTitleRow}>
                <div>
                  <h3>Active model</h3>
                  <p>Модель, которую стоит использовать в Inspect.</p>
                </div>
                {activeModel ? <span className={p.softBadge}>ready</span> : <span className={p.softBadgeMuted}>missing</span>}
              </div>
              {activeModel ? (
                <div className={p.activeModelCard}>
                  <div className={p.activeModelIcon}><ListChecks /></div>
                  <div>
                    <strong>{activeModel.architecture} {activeModel.version ? `v${activeModel.version}` : ""}</strong>
                    <span>{activeModel.imgsz}px · batch {activeModel.batch_size ?? "—"} · {activeModel.epochs ?? "—"} epochs</span>
                  </div>
                  <div className={p.metricMiniGrid}>
                    {metricNames.map((name) => <small key={name}><b>{name}</b>{formatMetric(activeModel.metrics?.[name])}</small>)}
                  </div>
                </div>
              ) : (
                <div className={p.emptyFocusCard}><Brain /><strong>No active model</strong><span>Импортируй веса или запусти обучение.</span></div>
              )}
            </section>

            <section className={p.panelCard}>
              <div className={p.cardTitleRow}>
                <div>
                  <h3>Latest task</h3>
                  <p>Последний training/import/export job.</p>
                </div>
                {latestTask ? <span className={p.softBadge}>{latestTask.status}</span> : null}
              </div>
              {latestTask ? (
                <div className={p.taskFocusCard}>
                  <Activity />
                  <div>
                    <strong>{latestTask.type}</strong>
                    <span>{latestTask.message || latestTask.stage || "Waiting for worker"}</span>
                  </div>
                  <b>{taskProgress(latestTask)}</b>
                </div>
              ) : (
                <div className={p.emptyFocusCard}><Clock3 /><strong>No tasks yet</strong><span>Training jobs появятся здесь.</span></div>
              )}
            </section>
          </div>

          <Outlet context={{ group, models, tasks }} />
        </main>
      </div>
    </div>
  );
}

type IconComponent = typeof Brain;

function ReadinessCard({ icon: Icon, label, value, hint, ready }: { icon: IconComponent; label: string; value: string; hint: string; ready: boolean }) {
  return (
    <div className={clsx(p.readinessCard, ready && p.readinessCardReady)}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}
