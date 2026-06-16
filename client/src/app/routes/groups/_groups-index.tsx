import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { CreateGroup } from "@/page-components/groups/components/create-group";
import { formatDate } from "@/utils/formatDate";
import { BarChart3, Box, CheckCircle2, Database, FolderOpen, Image, Rocket, Tags } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

export function Component() {
  const { data: groups } = useGetGroups();
  const totals = groups.reduce(
    (acc, group) => ({
      standards: acc.standards + group.stats.standards_count,
      images: acc.images + group.stats.images_count,
      labels: acc.labels + group.stats.polygons_count,
      models: acc.models + group.stats.models_count,
      runs: acc.runs + group.stats.inspections_count,
    }),
    { standards: 0, images: 0, labels: 0, models: 0, runs: 0 }
  );

  return (
    <div className={p.page}>
      <header className={p.ultraHeader}>
        <div>
          <h1>Datasets</h1>
          <p>Изделия, эталонные виды, изображения и полигоны собраны как datasets для QC workflow.</p>
        </div>
        <div className={p.headerActions}><CreateGroup /></div>
      </header>

      <section className={p.datasetOverviewStrip}>
        <DatasetMetric icon={FolderOpen} value={groups.length} label="Datasets" hint="groups" />
        <DatasetMetric icon={Database} value={totals.standards} label="References" hint="standard views" />
        <DatasetMetric icon={Image} value={totals.images} label="Images" hint="frames" />
        <DatasetMetric icon={Tags} value={totals.labels} label="Polygons" hint="annotations" />
        <DatasetMetric icon={Rocket} value={totals.runs} label="Runs" hint="checks" />
      </section>

      <div className={p.datasetHubGrid}>
        <section className={p.panelCard}>
          <div className={p.cardTitleRow}>
            <div>
              <h3>Annotate datasets</h3>
              <p>Создай dataset, добавь reference views и размечай контрольные зоны.</p>
            </div>
            <span className={p.softPill}>Annotate</span>
          </div>

          <QueryState isEmpty={!groups.length} size="block" emptyTitle="Нет datasets" emptyDescription="Создай первый dataset изделия для эталонов.">
            <div className={p.datasetListGrid}>
              {groups.map((group) => {
                const labeledPercent = group.stats.images_count
                  ? Math.round((group.stats.annotated_images_count / group.stats.images_count) * 100)
                  : 0;

                return (
                  <Link className={p.datasetListCard} key={group.id} to={paths.groupDetail(group.id)}>
                    <div className={p.datasetListCover}>
                      <Database />
                      <span>{labeledPercent}% labeled</span>
                    </div>
                    <div className={p.datasetListBody}>
                      <div>
                        <strong>{group.name}</strong>
                        <small>{group.description || "Standard-based visual quality dataset"}</small>
                      </div>
                      <div className={p.datasetListStats}>
                        <span><Image /> {group.stats.images_count}</span>
                        <span><Tags /> {group.stats.segment_classes_count}</span>
                        <span><Box /> {group.stats.polygons_count}</span>
                      </div>
                      <div className={p.referenceProgress}><span style={{ width: `${labeledPercent}%` }} /></div>
                      <div className={p.datasetListFooter}>
                        <span>Created {formatDate(group.created_at)}</span>
                        <b>Open →</b>
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </QueryState>
        </section>

        <aside className={p.sidePanel}>
          <h3>Dataset workflow</h3>
          <p>Лучший порядок для твоего приложения: reference → polygons → model → deploy → runs.</p>
          <div className={p.workflowList}>
            <WorkflowItem icon={Database} title="Reference views" text="Один или несколько эталонов под разные ракурсы." />
            <WorkflowItem icon={Tags} title="Classes & polygons" text="Контрольные зоны, которые потом переносятся на кадр." />
            <WorkflowItem icon={BarChart3} title="Model training" text="YOLO-модель для верификации деталей и классов." />
            <WorkflowItem icon={CheckCircle2} title="Inspection runs" text="Фото, snapshot или realtime проверка с историей." />
          </div>
        </aside>
      </div>
    </div>
  );
}

function DatasetMetric({ icon: Icon, value, label, hint }: { icon: typeof FolderOpen; value: number; label: string; hint: string }) {
  return (
    <div className={p.datasetMetricCard}>
      <Icon />
      <b>{value}</b>
      <span>{label}</span>
      <small>{hint}</small>
    </div>
  );
}

function WorkflowItem({ icon: Icon, title, text }: { icon: typeof Database; title: string; text: string }) {
  return (
    <div className={p.workflowItem}>
      <Icon />
      <div>
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  );
}
