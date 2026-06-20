import { paths } from "@/app/paths";
import Button from "@/components/ui/button/button";
import { ReadinessItem } from "@/components/ui/readiness-item/readiness-item";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { getInspectionHistoryQueryOptions } from "@/page-components/inspections/api/get-inspection-history";
import { useQuery } from "@tanstack/react-query";
import { formatDate } from "@/utils/formatDate";
import { Activity, Camera, CheckCircle2, CircleDot, Database, FolderKanban, Image, ListChecks, Sparkles, type LucideIcon } from "lucide-react";
import type { CSSProperties } from "react";
import { Link } from "react-router-dom";
import p from "./home.module.scss";

export function Component() {
  const { data: groups = [] } = useGetGroups();
  const { data: runs = [] } = useQuery(getInspectionHistoryQueryOptions(null));

  const totals = groups.reduce(
    (acc, group) => ({
      references: acc.references + group.stats.standards_count,
      images: acc.images + group.stats.images_count,
      labeled: acc.labeled + group.stats.annotated_images_count,
      polygons: acc.polygons + group.stats.polygons_count,
      classes: acc.classes + group.stats.segment_classes_count,
      models: acc.models + group.stats.models_count,
      runs: acc.runs + group.stats.inspections_count,
    }),
    { references: 0, images: 0, labeled: 0, polygons: 0, classes: 0, models: 0, runs: 0 }
  );

  const labelingPercent = totals.images ? Math.round((totals.labeled / totals.images) * 100) : 0;
  const passedRuns = runs.filter((run) => run.status === "passed").length;
  const failedRuns = runs.length - passedRuns;
  const topProjects = [...groups]
    .sort((a, b) => b.stats.images_count + b.stats.inspections_count - (a.stats.images_count + a.stats.inspections_count))
    .slice(0, 4);
  const recentRuns = runs.slice(0, 5);
  const nextProject = groups.find((group) => group.stats.standards_count === 0 || group.stats.segment_classes_count === 0 || group.stats.annotated_images_count === 0) ?? groups[0];

  return (
    <div className={`${p.page} ${p.homeCompactV18}`}>
      <section className={p.homeCockpitV18}>
        <div className={p.homeWelcomeV18}>
          <h1>Обзор</h1>
          <p>Проекты, эталоны, модели и проверки.</p>
          <div className={p.homeActionRowV18}>
            <Link to={paths.groups()}><Button icon={FolderKanban}>Открыть проекты</Button></Link>
            <Link to={paths.inspection()}><Button variant="ghost" icon={ListChecks}>Запустить проверку</Button></Link>
          </div>
        </div>

        <div className={p.homeScoreCardV18}>
          <span>Разметка</span>
          <strong>{labelingPercent}%</strong>
          <p>{totals.labeled}/{totals.images} размечено</p>
          <div className={p.homeScoreRingV18} style={{ "--value": `${Math.max(4, labelingPercent)}%` } as CSSProperties} />
        </div>

        <div className={p.homeKpiGridV18}>
          <Stat icon={FolderKanban} value={groups.length} label="Проекты" />
          <Stat icon={Database} value={totals.references} label="Эталоны" />
          <Stat icon={Image} value={totals.images} label="Фото" />
          <Stat icon={CircleDot} value={totals.classes} label="Классы" />
          <Stat icon={Sparkles} value={totals.models} label="Модели" />
          <Stat icon={Activity} value={runs.length || totals.runs} label="Проверки" />
        </div>
      </section>

      <section className={p.homeMainGridV18}>
        <div className={p.homePanelV18}>
          <div className={p.homePanelHeadV18}>
            <div>
              <span className={p.eyebrow}><FolderKanban /> Проекты</span>
              <h2>Изделия и эталоны</h2>
            </div>
            <Link to={paths.groups()}>Все →</Link>
          </div>
          <div className={p.homeProjectListV18}>
            {topProjects.length ? topProjects.map((group) => {
              const progress = group.stats.images_count ? Math.round((group.stats.annotated_images_count / group.stats.images_count) * 100) : 0;

              return (
                <Link className={p.homeProjectRowV18} key={group.id} to={paths.groupDetail(group.id)}>
                  <span className={p.homeProjectAvatarV18}>{group.name.slice(0, 1).toUpperCase()}</span>
                  <span>
                    <strong>{group.name}</strong>
                    <small>{group.stats.standards_count} refs · {group.stats.images_count} images · {group.stats.segment_classes_count} classes</small>
                  </span>
                  <b>{progress}%</b>
                </Link>
              );
            }) : (
              <Link className={p.homeEmptyActionV18} to={paths.groups()}><FolderKanban /> Открыть проекты</Link>
            )}
          </div>
        </div>

        <div className={p.homePanelV18}>
          <div className={p.homePanelHeadV18}>
            <div>
              <span className={p.eyebrow}><ListChecks /> История</span>
              <h2>Проверки</h2>
            </div>
            <Link to={paths.inspectionHistory()}>История →</Link>
          </div>
          <div className={p.homeRunSummaryV18}>
            <div><strong>{passedRuns}</strong><span>успешно</span></div>
            <div><strong>{failedRuns}</strong><span>к проверке</span></div>
            <div><strong>{recentRuns.length}</strong><span>последние</span></div>
          </div>
          <div className={p.homeRunListV18}>
            {recentRuns.length ? recentRuns.map((run) => (
              <Link className={p.homeRunRowV18} key={run.id} to={run.group_id ? paths.inspectionHistoryDetail(run.group_id, run.id) : paths.inspectionHistory()}>
                <span className={run.status === "passed" ? p.homeRunOkV18 : p.homeRunBadV18} />
                <strong>{run.standard_name ?? "Проверка"}</strong>
                <small>{run.mode} · {formatDate(run.inspected_at)}</small>
              </Link>
            )) : (
              <Link className={p.homeEmptyActionV18} to={nextProject ? paths.inspectionGroup("photo", nextProject.id) : paths.inspection()}><ListChecks /> Запустить первую проверку</Link>
            )}
          </div>
        </div>

        <aside className={p.homePanelV18}>
          <div className={p.homePanelHeadV18}>
            <div>
              <span className={p.eyebrow}><CheckCircle2 /> Готовность</span>
              <h2>Следующие шаги</h2>
            </div>
          </div>
          <div className={p.homeChecklistV18}>
            <ChecklistItem done={groups.length > 0} title="Проект создан" text={`${groups.length} изделий`} />
            <ChecklistItem done={totals.references > 0} title="Эталоны добавлены" text={`${totals.references} эталонов`} />
            <ChecklistItem done={totals.classes > 0} title="Классы настроены" text={`${totals.classes} классов`} />
            <ChecklistItem done={labelingPercent >= 80} title="Изображения размечены" text={`${labelingPercent}% готово`} />
            <ChecklistItem done={totals.models > 0} title="Модель готова" text={`${totals.models} моделей`} />
          </div>
          <div className={p.homeQuickLinksV18}>
            <Link to={nextProject ? paths.groupDetail(nextProject.id) : paths.groups()}><Database /> Активы</Link>
            <Link to={nextProject ? paths.trainingGroup(nextProject.id) : paths.training()}><Sparkles /> Обучение</Link>
            <Link to={nextProject ? paths.inspectionGroup("photo", nextProject.id) : paths.inspection()}><ListChecks /> Проверка</Link>
            <Link to={paths.cameras()}><Camera /> Камеры</Link>
          </div>
        </aside>
      </section>
    </div>
  );
}

function Stat({ icon: Icon, value, label }: { icon: LucideIcon; value: number; label: string }) {
  return <div className={p.homeStatV18}><Icon /><strong>{value}</strong><span>{label}</span></div>;
}

function ChecklistItem({ done, title, text }: { done: boolean; title: string; text: string }) {
  return <ReadinessItem className={done ? `${p.homeCheckItemV18} ${p.homeCheckDoneV18}` : p.homeCheckItemV18} done={done} pendingIcon={CircleDot} title={title} description={text} variant="compact" />;
}
