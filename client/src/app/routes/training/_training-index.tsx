import { paths } from "@/app/paths";
import { MetricCard } from "@/components/ui/metric-card/metric-card";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import type { GroupListItem } from "@/types/contracts";
import { Brain, CheckCircle2, Database, Image, ListChecks, Tags, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import p from "./_training-index.module.scss";

function getReadiness(group: GroupListItem) {
  const checks = [
    group.stats.standards_count > 0,
    group.stats.images_count > 0,
    group.stats.segment_classes_count > 0,
    group.stats.annotated_images_count > 0,
  ];
  return Math.round((checks.filter(Boolean).length / checks.length) * 100);
}

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

export function Component() {
  const groupsQuery = useGetGroups();
  const groups = groupsQuery.data ?? [];
  const totalModels = groups.reduce((sum, group) => sum + group.stats.models_count, 0);
  const totalImages = groups.reduce((sum, group) => sum + group.stats.images_count, 0);
  const totalLabeled = groups.reduce((sum, group) => sum + group.stats.annotated_images_count, 0);
  const trainingReady = groups.filter((group) => getReadiness(group) >= 75).length;

  return (
    <div className={`${p.page} ${p.trainPageV27}`}>
      <header className={p.trainHeaderV27}>
        <div className={p.trainHeaderIconV27}><Brain /></div>
        <div>
          <h1>Модели</h1>
          <p>Готовность данных и модели по изделиям.</p>
        </div>
      </header>

      <section className={p.trainMetricGridV27}>
        <MetricCard className={p.trainMetricV27} icon={Database} value={groups.length} label="Изделия" hint={`${trainingReady} готово`} />
        <MetricCard className={p.trainMetricV27} icon={Image} value={totalImages} label="Фото" hint={`${percent(totalLabeled, totalImages)}% размечено`} />
        <MetricCard className={p.trainMetricV27} icon={Tags} value={groups.reduce((sum, group) => sum + group.stats.segment_classes_count, 0)} label="Классы" hint="детали" />
        <MetricCard className={p.trainMetricV27} icon={Brain} value={totalModels} label="Модели" hint="веса" />
      </section>

      <QueryState
        size="block"
        isLoading={groupsQuery.isLoading}
        isError={groupsQuery.isError}
        isEmpty={!groups.length}
        emptyTitle="Нет изделий"
        emptyDescription="Создай изделие и подготовь данные в Assets."
      >
        <details className={p.collapsiblePanelV28} open>
          <summary><span><Brain /> Изделия</span><b>{groups.length}</b></summary>
          <div className={`${p.trainProjectQueueV27} ${p.scrollListV28}`}>
          {groups.map((group) => {
            const readiness = getReadiness(group);
            const labeled = percent(group.stats.annotated_images_count, group.stats.images_count);
            const nextAction =
              group.stats.standards_count === 0 ? "Добавить эталон" :
              group.stats.segment_classes_count === 0 ? "Настроить классы" :
              group.stats.annotated_images_count === 0 ? "Разметить фото" :
              group.stats.models_count === 0 ? "Обучить модель" :
              "Открыть модели";

            return (
              <Link key={group.id} to={paths.trainingOverview(group.id)} className={p.trainProjectCardV27}>
                <div className={p.trainProjectAvatarV27}>{group.name.slice(0, 1).toUpperCase()}</div>
                <div>
                  <strong>{group.name}</strong>
                  <span>{group.description || "Изделие для контроля качества"}</span>
                </div>
                <div className={p.trainProjectMetaV27}>
                  <small><Image /> {group.stats.images_count} фото</small>
                  <small><Tags /> {group.stats.segment_classes_count} классов</small>
                  <small><Brain /> {group.stats.models_count} моделей</small>
                  <small><ListChecks /> {labeled}% разметки</small>
                </div>
                <div className={p.trainProjectProgressV27}>
                  <span style={{ width: `${readiness}%` }} />
                </div>
                <b>{nextAction}</b>
              </Link>
            );
          })}
          </div>
        </details>
      </QueryState>
    </div>
  );
}
