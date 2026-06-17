import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { AlertTriangle, CheckCircle2, Database, History, Image, ListChecks, Search, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { data: groups } = useGetGroups();
  const totalRuns = groups.reduce((sum, group) => sum + group.stats.inspections_count, 0);
  const datasetsWithRuns = groups.filter((group) => group.stats.inspections_count > 0).length;
  const totalImages = groups.reduce((sum, group) => sum + group.stats.images_count, 0);

  return (
    <div className={p.page}>
      <header className={p.workspaceHeaderV16}>
        <div>
          <span className={p.eyebrow}><History /> Run history</span>
          <h1>Runs</h1>
          <p>Сохранённые проверки: фото, snapshot и realtime. Здесь должен открываться отчёт, а не техническая таблица.</p>
        </div>
      </header>

      <section className={p.commandMetricGridV16}>
        <Metric icon={ListChecks} value={totalRuns} label="Total runs" hint="saved checks" />
        <Metric icon={Database} value={datasetsWithRuns} label="Datasets" hint="with history" />
        <Metric icon={Image} value={totalImages} label="Images" hint="reference base" />
        <Metric icon={Search} value={groups.length} label="Scopes" hint="available datasets" />
      </section>

      <QueryState isEmpty={!groups.length} size="page" emptyTitle="No datasets" emptyDescription="Проверки появятся после создания dataset и запуска Inspect.">
        <div className={p.runsHubGridV16}>
          <section className={p.surfacePanelV16}>
            <div className={p.panelHeadV16}>
              <div><h3>Inspection scopes</h3><p>Выбери dataset, чтобы открыть timeline и детальный отчёт.</p></div>
              <span>{totalRuns} runs</span>
            </div>
            <div className={p.runScopeGridV16}>
              {groups.map((group) => {
                const hasRuns = group.stats.inspections_count > 0;
                return (
                  <Link className={p.runScopeCardV16} key={group.id} to={paths.inspectionHistoryGroup(group.id)}>
                    <div className={p.queueAvatarV16}>{group.name.slice(0, 1).toUpperCase()}</div>
                    <div className={p.queueMainV16}>
                      <div className={p.queueTitleV16}>
                        <strong>{group.name}</strong>
                        <span>{hasRuns ? "has runs" : "empty"}</span>
                      </div>
                      <p>{group.stats.inspections_count} checks · {group.stats.standards_count} references · {group.stats.segment_classes_count} classes</p>
                      <div className={p.queueMetaV16}>
                        <span><ListChecks /> {group.stats.inspections_count} runs</span>
                        <span><Image /> {group.stats.images_count} images</span>
                        <span>{hasRuns ? <CheckCircle2 /> : <AlertTriangle />} {hasRuns ? "open report" : "run Inspect first"}</span>
                      </div>
                    </div>
                    <b className={p.scopeCountV16}>{group.stats.inspections_count}</b>
                  </Link>
                );
              })}
            </div>
          </section>
        </div>
      </QueryState>
    </div>
  );
}

function Metric({ icon: Icon, value, label, hint }: { icon: LucideIcon; value: number; label: string; hint: string }) {
  return <div className={p.metricCardV16}><Icon /><b>{value}</b><span>{label}</span><small>{hint}</small></div>;
}
