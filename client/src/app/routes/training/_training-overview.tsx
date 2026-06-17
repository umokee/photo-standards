import { paths } from "@/app/paths";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import { formatDate } from "@/utils/formatDate";
import { Activity, Brain, CheckCircle2, CircleDashed, Database, Image, ListChecks, Rocket, Tags, type LucideIcon } from "lucide-react";
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
          <Metric icon={Rocket} value={`${readiness}%`} label="Readiness" hint={canTrain ? "ready to train" : "needs assets"} />
          <Metric icon={Image} value={`${labeled}%`} label="Labeled" hint={`${group.stats.annotated_images_count}/${group.stats.images_count} images`} />
          <Metric icon={Brain} value={models.length} label="Models" hint={activeModel ? "active selected" : "no active model"} />
          <Metric icon={Activity} value={tasks.length} label="Jobs" hint={`${tasks.filter((task) => isActiveTaskStatus(task.status)).length} active`} />
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
              <ReadinessItem done={group.stats.standards_count > 0} title="References exist" text={`${group.stats.standards_count} reference views`} to={paths.assetReferences(group.id)} />
              <ReadinessItem done={group.stats.images_count > 0} title="Images uploaded" text={`${group.stats.images_count} images in annotation queue`} to={paths.assetReferences(group.id)} />
              <ReadinessItem done={group.stats.segment_classes_count > 0} title="Classes configured" text={`${group.stats.segment_classes_count} classes in Assets / Classes`} to={paths.assetClasses(group.id)} />
              <ReadinessItem done={group.stats.annotated_images_count > 0} title="Annotations exist" text={`${labeled}% labeled images`} to={paths.assetReferences(group.id)} />
              <ReadinessItem done={models.length > 0} title="Model exists" text={`${models.length} trained/imported models`} to={paths.trainingModels(group.id)} />
            </div>
          </main>

          <aside className={p.trainSideRailV27}>
            <section className={p.trainRailCardV27}>
              <span className={p.eyebrow}><Brain /> Active model</span>
              {activeModel ? (
                <Link to={paths.trainingModel(group.id, activeModel.id)} className={p.trainModelMiniV27}>
                  <Brain />
                  <div>
                    <strong>{activeModel.architecture} {activeModel.version ? `v${activeModel.version}` : ""}</strong>
                    <span>{activeModel.imgsz}px · {activeModel.epochs ?? "—"} epochs · {activeModel.num_classes ?? group.stats.segment_classes_count} classes</span>
                  </div>
                </Link>
              ) : (
                <div className={p.trainEmptyV27}><CircleDashed /><strong>No active model</strong><span>Train/import model, then activate it for Inspect.</span></div>
              )}
            </section>

            <section className={p.trainRailCardV27}>
              <span className={p.eyebrow}><Activity /> Latest job</span>
              {latestTask ? (
                <Link to={paths.trainingRuns(group.id)} className={p.trainTaskMiniV27}>
                  <Activity />
                  <div>
                    <strong>{latestTask.type}</strong>
                    <span>{latestTask.status} · {latestTask.stage ?? "queued"} · {formatDate(latestTask.created_at)}</span>
                  </div>
                  <b>{latestTask.progress_percent ?? 0}%</b>
                </Link>
              ) : (
                <div className={p.trainEmptyV27}><CircleDashed /><strong>No training jobs</strong><span>Jobs appear after Train/Import/Export.</span></div>
              )}
            </section>
          </aside>
        </section>
      </div>
    </div>
  );
}

function Metric({ icon: Icon, value, label, hint }: { icon: LucideIcon; value: string | number; label: string; hint: string }) {
  return (
    <div className={p.trainMetricV27}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}

function ReadinessItem({ done, title, text, to }: { done: boolean; title: string; text: string; to: string }) {
  return (
    <Link to={to} className={`${p.trainReadinessItemV27} ${done ? p.trainReadinessDoneV27 : ""}`}>
      {done ? <CheckCircle2 /> : <CircleDashed />}
      <div>
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </Link>
  );
}
