
import { paths } from "@/app/paths";
import { MetricCard } from "@/components/ui/metric-card/metric-card";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import type { GroupListItem } from "@/types/contracts";
import {
  ArrowRight,
  CheckCircle2,
  CircleDot,
  Database,
  History,
  Image,
  ListChecks,
  PlayCircle,
  Search,
  type LucideIcon,
} from "lucide-react";
import { Link } from "react-router-dom";
import s from "./_inspection-history-strict.module.scss";

export function Component() {
  const { data: groups } = useGetGroups();
  const totalRuns = groups.reduce((sum, group) => sum + group.stats.inspections_count, 0);
  const scopesWithRuns = groups.filter((group) => group.stats.inspections_count > 0).length;
  const totalReferences = groups.reduce((sum, group) => sum + group.stats.standards_count, 0);
  const totalImages = groups.reduce((sum, group) => sum + group.stats.images_count, 0);
  const sortedGroups = [...groups].sort((a, b) => b.stats.inspections_count - a.stats.inspections_count || a.name.localeCompare(b.name, "ru"));

  return (
    <main className={s.page}>
      <section className={s.heroPanel}>
        <div className={s.heroContent}>
          <div className={s.titleRow}>
            <div>
              <h1>История проверок</h1>
              <p>Сохранённые результаты по изделиям.</p>
            </div>
            <Link className={s.primaryAction} to={paths.inspectionMode("photo")}>
              <PlayCircle /> Новая проверка
            </Link>
          </div>
        </div>
      </section>

      <section className={s.metricGrid}>
        <MetricCard className={s.metricCard} icon={ListChecks} value={totalRuns} label="Проверки" hint="отчёты" variant="valueFirst" />
        <MetricCard className={s.metricCard} icon={Database} value={scopesWithRuns} label="Изделия" hint="с историей" variant="valueFirst" />
        <MetricCard className={s.metricCard} icon={Image} value={totalImages} label="Фото" hint="источники" variant="valueFirst" />
        <MetricCard className={s.metricCard} icon={Search} value={totalReferences} label="Эталоны" hint="виды" variant="valueFirst" />
      </section>

      <QueryState
        isEmpty={!groups.length}
        size="page"
        emptyTitle="Нет изделий"
        emptyDescription="История появится после создания изделия и запуска первой проверки."
      >
        <section className={s.scopeShell}>
          <aside className={s.scopeIntro}>
            <h2>Изделия</h2>
            <p>Открой изделие, чтобы посмотреть проверки.</p>
            <div className={s.sideFacts}>
              <span><CircleDot /> {groups.length} изделий</span>
              <span><CheckCircle2 /> {scopesWithRuns} с историей</span>
              <span><ListChecks /> {totalRuns} отчётов</span>
            </div>
          </aside>

          <div className={s.scopeGrid}>
            {sortedGroups.map((group) => <ScopeCard key={group.id} group={group} />)}
          </div>
        </section>
      </QueryState>
    </main>
  );
}

function ScopeCard({ group }: { group: GroupListItem }) {
  const hasRuns = group.stats.inspections_count > 0;
  const readiness = getReadiness(group);

  return (
    <Link className={s.scopeCard} to={paths.inspectionHistoryGroup(group.id)}>
      <div className={s.scopeAvatar}>{group.name.slice(0, 1).toUpperCase()}</div>
      <div className={s.scopeBody}>
        <div className={s.scopeTopline}>
          <strong>{group.name}</strong>
          <span className={hasRuns ? s.statusReady : s.statusEmpty}>{hasRuns ? "есть проверки" : "пусто"}</span>
        </div>
        <p>{group.description || `${group.stats.standards_count} эталонов · ${group.stats.segment_classes_count} классов · ${group.stats.models_count} моделей`}</p>
        <div className={s.scopeMeta}>
          <span><ListChecks /> {group.stats.inspections_count} проверок</span>
          <span><Image /> {group.stats.images_count} фото</span>
          <span><Database /> {readiness}</span>
        </div>
      </div>
      <div className={s.scopeOpen}>
        <b>{group.stats.inspections_count}</b>
        <ArrowRight />
      </div>
    </Link>
  );
}

function getReadiness(group: GroupListItem) {
  if (!group.stats.standards_count) return "нет эталонов";
  if (!group.stats.segment_classes_count) return "нет классов";
  if (!group.stats.models_count) return "нет модели";
  return "готово";
}
