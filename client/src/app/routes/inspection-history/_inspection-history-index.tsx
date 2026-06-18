
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
          <span className={s.eyebrow}><History /> Inspection history</span>
          <div className={s.titleRow}>
            <div>
              <h1>Runs</h1>
              <p>Сохранённые проверки по изделиям: результат, исходное фото, эталон, режим запуска и детализация по компонентам.</p>
            </div>
            <Link className={s.primaryAction} to={paths.inspectionMode("photo")}>
              <PlayCircle /> Новая проверка
            </Link>
          </div>
        </div>
      </section>

      <section className={s.metricGrid}>
        <MetricCard className={s.metricCard} icon={ListChecks} value={totalRuns} label="Total runs" hint="saved reports" variant="valueFirst" />
        <MetricCard className={s.metricCard} icon={Database} value={scopesWithRuns} label="Изделия" hint="with history" variant="valueFirst" />
        <MetricCard className={s.metricCard} icon={Image} value={totalImages} label="Images" hint="source/reference" variant="valueFirst" />
        <MetricCard className={s.metricCard} icon={Search} value={totalReferences} label="References" hint="available views" variant="valueFirst" />
      </section>

      <QueryState
        isEmpty={!groups.length}
        size="page"
        emptyTitle="Нет изделий"
        emptyDescription="История появится после создания изделия и запуска первой проверки."
      >
        <section className={s.scopeShell}>
          <aside className={s.scopeIntro}>
            <span className={s.sideLabel}>Registry</span>
            <h2>История по изделиям</h2>
            <p>Открой изделие, чтобы посмотреть timeline запусков и полный отчёт выбранной проверки.</p>
            <div className={s.sideFacts}>
              <span><CircleDot /> {groups.length} scopes</span>
              <span><CheckCircle2 /> {scopesWithRuns} active</span>
              <span><ListChecks /> {totalRuns} reports</span>
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
          <span className={hasRuns ? s.statusReady : s.statusEmpty}>{hasRuns ? "has runs" : "empty"}</span>
        </div>
        <p>{group.description || `${group.stats.standards_count} эталонов · ${group.stats.segment_classes_count} классов · ${group.stats.models_count} моделей`}</p>
        <div className={s.scopeMeta}>
          <span><ListChecks /> {group.stats.inspections_count} runs</span>
          <span><Image /> {group.stats.images_count} images</span>
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
  if (!group.stats.standards_count) return "no references";
  if (!group.stats.segment_classes_count) return "no classes";
  if (!group.stats.models_count) return "no model";
  return "ready";
}
