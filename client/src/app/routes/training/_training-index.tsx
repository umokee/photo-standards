import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import type { GroupListItem } from "@/types/contracts";
import { Brain, CheckCircle2, Database, Image, ListChecks, Tags, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

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
          <span className={p.eyebrow}><Brain /> Train</span>
          <h1>Model workspace</h1>
          <p>Обучение привязано к изделию: сначала Assets готовят references/classes/annotations, затем Train управляет моделями и training jobs.</p>
        </div>
      </header>

      <section className={p.trainMetricGridV27}>
        <Metric icon={Database} value={groups.length} label="Projects" hint={`${trainingReady} train-ready`} />
        <Metric icon={Image} value={totalImages} label="Images" hint={`${percent(totalLabeled, totalImages)}% labeled`} />
        <Metric icon={Tags} value={groups.reduce((sum, group) => sum + group.stats.segment_classes_count, 0)} label="Classes" hint="segment labels" />
        <Metric icon={Brain} value={totalModels} label="Models" hint="weights" />
      </section>

      <QueryState
        size="block"
        isLoading={groupsQuery.isLoading}
        isError={groupsQuery.isError}
        isEmpty={!groups.length}
        emptyTitle="No projects"
        emptyDescription="Создай project, добавь references/classes в Assets и вернись к Train."
      >
        <details className={p.collapsiblePanelV28} open>
          <summary><span><Brain /> Project queue</span><b>{groups.length}</b></summary>
          <div className={`${p.trainProjectQueueV27} ${p.scrollListV28}`}>
          {groups.map((group) => {
            const readiness = getReadiness(group);
            const labeled = percent(group.stats.annotated_images_count, group.stats.images_count);
            const nextAction =
              group.stats.standards_count === 0 ? "Create reference" :
              group.stats.segment_classes_count === 0 ? "Create classes" :
              group.stats.annotated_images_count === 0 ? "Annotate images" :
              group.stats.models_count === 0 ? "Train first model" :
              "Open model lab";

            return (
              <Link key={group.id} to={paths.trainingOverview(group.id)} className={p.trainProjectCardV27}>
                <div className={p.trainProjectAvatarV27}>{group.name.slice(0, 1).toUpperCase()}</div>
                <div>
                  <strong>{group.name}</strong>
                  <span>{group.description || "Quality-control project"}</span>
                </div>
                <div className={p.trainProjectMetaV27}>
                  <small><Image /> {group.stats.images_count} images</small>
                  <small><Tags /> {group.stats.segment_classes_count} classes</small>
                  <small><Brain /> {group.stats.models_count} models</small>
                  <small><ListChecks /> {labeled}% labeled</small>
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
