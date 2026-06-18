import { paths } from "@/app/paths";
import { EmptyStateCard } from "@/components/ui/empty-state-card/empty-state-card";
import { EntityMiniCard } from "@/components/ui/entity-mini-card/entity-mini-card";
import { MetricCard } from "@/components/ui/metric-card/metric-card";
import { ReadinessItem } from "@/components/ui/readiness-item/readiness-item";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import { formatDate } from "@/utils/formatDate";
import { Activity, Brain, CircleDashed, Database, Image, ListChecks, Rocket, Tags } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";
import { useTrainingModelOutletContext } from "./_training-detail";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

export function Component() {
  const { group, models, tasks } = useTrainingModelOutletContext();
  const hasActiveTrainingTask = tasks.some((task) => isActiveTaskStatus(task.status));
  const canTrain =
    group.stats.standards_count > 0 &&
    group.stats.images_count > 0 &&
    group.stats.annotated_images_count > 0 &&
    group.stats.segment_classes_count > 0;
  const labeled = percent(group.stats.annotated_images_count, group.stats.images_count);
  const activeModel = models.find((model) => model.is_active) ?? null;
  const latestTask = tasks[0] ?? null;
  const readyChecks = [
    group.stats.standards_count > 0,
    group.stats.images_count > 0,
    group.stats.segment_classes_count > 0,
    group.stats.annotated_images_count > 0,
    models.length > 0,
  ];
  const readiness = percent(readyChecks.filter(Boolean).length, readyChecks.length);

  const nextAction =
    group.stats.standards_count === 0 ? { label: "Create reference", to: paths.assetReferences(group.id) } :
    group.stats.segment_classes_count === 0 ? { label: "Create classes", to: paths.assetClasses(group.id) } :
    group.stats.annotated_images_count === 0 ? { label: "Annotate images", to: paths.assetReferences(group.id) } :
    models.length === 0 ? { label: "Open models", to: paths.trainingModels(group.id) } :
    activeModel ? { label: "Inspect with active model", to: paths.inspectionGroup("photo", group.id) } :
    { label: "Choose active model", to: paths.trainingModels(group.id) };

  return (
    <div className={`${p.page} ${p.trainPageV27} ${p.trainOverviewScreenV29}`}>
      <header className={`${p.trainHeaderV27} ${p.trainHeaderCompactV27} ${p.trainHeaderSlimV29}`}>
        <div>
          <span className={p.eyebrow}><Brain /> Train / Overview</span>
          <h1>{group.name}</h1>
          <p>Train отвечает только за готовность данных к обучению, модели и training jobs. References/classes/annotations остаются в Assets.</p>
        </div>
        <div className={p.trainActionsV27}>
          <ImportModel groupId={group.id} />
          <ExportModel models={models} />
          <TrainModel groupId={group.id} canTrain={canTrain} isTrainingLocked={hasActiveTrainingTask} />
        </div>
      </header>

      <div className={p.trainBodyScrollV29}>
        <section className={p.trainMetricGridV27}>
          <MetricCard className={p.trainMetricV27} icon={Rocket} value={`${readiness}%`} label="Readiness" hint={canTrain ? "ready to train" : "needs assets"} />
          <MetricCard className={p.trainMetricV27} icon={Image} value={`${labeled}%`} label="Labeled" hint={`${group.stats.annotated_images_count}/${group.stats.images_count} images`} />
          <MetricCard className={p.trainMetricV27} icon={Brain} value={models.length} label="Models" hint={activeModel ? "active selected" : "no active model"} />
          <MetricCard className={p.trainMetricV27} icon={Activity} value={tasks.length} label="Jobs" hint={`${tasks.filter((task) => isActiveTaskStatus(task.status)).length} active`} />
        </section>

        <section className={p.trainWorkspaceGridV27}>
          <main className={p.trainMainPanelV27}>
            <div className={p.trainPanelHeaderV27}>
              <div>
                <span className={p.eyebrow}><ListChecks /> Quality gates</span>
                <h2>{canTrain ? "Assets are ready" : "Finish Assets before training"}</h2>
                <p>Эта страница не создаёт classes/references сама, а показывает зависимости и ведёт в нужную зону.</p>
              </div>
              <Link to={nextAction.to} className={p.trainPrimaryLinkV27}>{nextAction.label}</Link>
            </div>

            <div className={p.trainReadinessListV27}>
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.standards_count > 0} title="References exist" description={`${group.stats.standards_count} reference views`} to={paths.assetReferences(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.images_count > 0} title="Images uploaded" description={`${group.stats.images_count} images in annotation queue`} to={paths.assetReferences(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.segment_classes_count > 0} title="Classes configured" description={`${group.stats.segment_classes_count} classes in Assets / Classes`} to={paths.assetClasses(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.annotated_images_count > 0} title="Annotations exist" description={`${labeled}% labeled images`} to={paths.assetReferences(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={models.length > 0} title="Model exists" description={`${models.length} trained/imported models`} to={paths.trainingModels(group.id)} />
            </div>
          </main>

          <aside className={p.trainSideRailV27}>
            <section className={p.trainRailCardV27}>
              <span className={p.eyebrow}><Brain /> Active model</span>
              {activeModel ? (
                <EntityMiniCard
                  to={paths.trainingModel(group.id, activeModel.id)}
                  className={p.trainModelMiniV27}
                  icon={Brain}
                  title={`${activeModel.architecture} ${activeModel.version ? `v${activeModel.version}` : ""}`}
                  meta={`${activeModel.imgsz}px · ${activeModel.epochs ?? "—"} epochs · ${activeModel.num_classes ?? group.stats.segment_classes_count} classes`}
                />
              ) : (
                <EmptyStateCard className={p.trainEmptyV27} icon={CircleDashed} title="No active model" description="Train/import model, then activate it for Inspect." />
              )}
            </section>

            <section className={p.trainRailCardV27}>
              <span className={p.eyebrow}><Activity /> Latest job</span>
              {latestTask ? (
                <EntityMiniCard
                  to={paths.trainingRuns(group.id)}
                  className={p.trainTaskMiniV27}
                  icon={Activity}
                  title={latestTask.type}
                  meta={`${latestTask.status} · ${latestTask.stage ?? "queued"} · ${formatDate(latestTask.created_at)}`}
                  trailing={<b>{latestTask.progress_percent ?? 0}%</b>}
                />
              ) : (
                <EmptyStateCard className={p.trainEmptyV27} icon={CircleDashed} title="No training jobs" description="Jobs appear after Train/Import/Export." />
              )}
            </section>
          </aside>
        </section>
      </div>
    </div>
  );
}
