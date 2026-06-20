import { paths } from "@/app/paths";
import { EntityMiniCard } from "@/components/ui/entity-mini-card/entity-mini-card";
import { MetricCard } from "@/components/ui/metric-card/metric-card";
import { ReadinessItem } from "@/components/ui/readiness-item/readiness-item";
import { ExportModel } from "@/page-components/models/components/export-model/export-model";
import { ImportModel } from "@/page-components/models/components/import-model/import-model";
import { TrainModel } from "@/page-components/models/components/train-model/train-model";
import { isActiveTaskStatus } from "@/page-components/tasks/lib/task-helpers";
import { formatDate } from "@/utils/formatDate";
import { Activity, Brain, Image, ListChecks, Rocket } from "lucide-react";
import { Link } from "react-router-dom";
import p from "./_training-overview.module.scss";
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
    group.stats.standards_count === 0 ? { label: "Добавить эталон", to: paths.assetReferences(group.id) } :
    group.stats.segment_classes_count === 0 ? { label: "Настроить классы", to: paths.assetClasses(group.id) } :
    group.stats.annotated_images_count === 0 ? { label: "Разметить фото", to: paths.assetReferences(group.id) } :
    models.length === 0 ? { label: "Открыть модели", to: paths.trainingModels(group.id) } :
    activeModel ? { label: "Проверить с моделью", to: paths.inspectionGroup("photo", group.id) } :
    { label: "Выбрать модель", to: paths.trainingModels(group.id) };

  return (
    <div className={`${p.page} ${p.trainPageV27} ${p.trainOverviewScreenV29}`}>
      <header className={`${p.trainHeaderV27} ${p.trainHeaderCompactV27} ${p.trainHeaderSlimV29}`}>
        <div>
          <h1>{group.name}</h1>
          <p>Данные, модели и задания обучения.</p>
        </div>
        <div className={p.trainActionsV27}>
          <TrainModel groupId={group.id} canTrain={canTrain} isTrainingLocked={hasActiveTrainingTask} triggerClassName={p.trainPrimaryAction} />
          <ImportModel groupId={group.id} triggerClassName={p.trainSecondaryAction} />
          <ExportModel models={models} triggerClassName={p.trainSecondaryAction} />
        </div>
      </header>

      <div className={p.trainBodyScrollV29}>
        <section className={p.trainMetricGridV27}>
          <MetricCard className={p.trainMetricV27} icon={Rocket} value={`${readiness}%`} label="Готовность" hint={canTrain ? "можно обучать" : "нужны данные"} />
          <MetricCard className={p.trainMetricV27} icon={Image} value={`${labeled}%`} label="Разметка" hint={`${group.stats.annotated_images_count}/${group.stats.images_count} фото`} />
          <MetricCard className={p.trainMetricV27} icon={Brain} value={models.length} label="Модели" hint={activeModel ? "активная выбрана" : "нет активной"} />
          <MetricCard className={p.trainMetricV27} icon={Activity} value={tasks.length} label="Задания" hint={`${tasks.filter((task) => isActiveTaskStatus(task.status)).length} активных`} />
        </section>

        <section className={activeModel || latestTask ? p.trainWorkspaceGridV27 : `${p.trainWorkspaceGridV27} ${p.trainWorkspaceFullV87}`}>
          <main className={p.trainMainPanelV27}>
            <div className={p.trainPanelHeaderV27}>
              <div>
                <span className={p.eyebrow}><ListChecks /> Готовность</span>
                <h2>{canTrain ? "Данные готовы" : "Доделай Assets"}</h2>
                <p>Проверь зависимости перед обучением.</p>
              </div>
              <Link to={nextAction.to} className={p.trainPrimaryLinkV27}>{nextAction.label}</Link>
            </div>

            <div className={p.trainReadinessListV27}>
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.standards_count > 0} title="Эталоны" description={`${group.stats.standards_count} видов`} to={paths.assetReferences(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.images_count > 0} title="Фото" description={`${group.stats.images_count} загружено`} to={paths.assetReferences(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.segment_classes_count > 0} title="Классы" description={`${group.stats.segment_classes_count} настроено`} to={paths.assetClasses(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={group.stats.annotated_images_count > 0} title="Разметка" description={`${labeled}% готово`} to={paths.assetReferences(group.id)} />
              <ReadinessItem className={p.trainReadinessItemV27} done={models.length > 0} title="Модель" description={`${models.length} обучено/импортировано`} to={paths.trainingModels(group.id)} />
            </div>
          </main>

          {activeModel || latestTask ? (
            <aside className={p.trainSideRailV27}>
              {activeModel ? (
                <section className={p.trainRailCardV27}>
                  <span className={p.eyebrow}><Brain /> Активная модель</span>
                  <EntityMiniCard
                    to={paths.trainingModel(group.id, activeModel.id)}
                    className={p.trainModelMiniV27}
                    icon={Brain}
                    title={`${activeModel.architecture} ${activeModel.version ? `v${activeModel.version}` : ""}`}
                    meta={`${activeModel.imgsz}px · ${activeModel.epochs ?? "—"} epochs · ${activeModel.num_classes ?? group.stats.segment_classes_count} classes`}
                  />
                </section>
              ) : null}

              {latestTask ? (
                <section className={p.trainRailCardV27}>
                  <span className={p.eyebrow}><Activity /> Последнее задание</span>
                  <EntityMiniCard
                    to={paths.trainingRuns(group.id)}
                    className={p.trainTaskMiniV27}
                    icon={Activity}
                    title={latestTask.type}
                    meta={`${latestTask.status} · ${latestTask.stage ?? "queued"} · ${formatDate(latestTask.created_at)}`}
                    trailing={<b>{latestTask.progress_percent ?? 0}%</b>}
                  />
                </section>
              ) : null}
            </aside>
          ) : null}
        </section>
      </div>
    </div>
  );
}
