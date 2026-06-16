import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import type { GroupListItem } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Brain, CheckCircle2, Database, Image, Layers3, Rocket, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

function getReadiness(group: GroupListItem) {
  const checks = [
    group.stats.images_count > 0,
    group.stats.annotated_images_count > 0,
    group.stats.segment_classes_count > 0,
    group.stats.standards_count > 0,
  ];
  return Math.round((checks.filter(Boolean).length / checks.length) * 100);
}

export function Component() {
  const { data: groups } = useGetGroups();
  const totalModels = groups.reduce((sum, group) => sum + group.stats.models_count, 0);
  const readyProjects = groups.filter((group) => getReadiness(group) >= 75).length;
  const totalClasses = groups.reduce((sum, group) => sum + group.stats.segment_classes_count, 0);
  const totalImages = groups.reduce((sum, group) => sum + group.stats.images_count, 0);

  return (
    <div className={p.page}>
      <section className={p.trainHero}>
        <div className={p.trainHeroCopy}>
          <span className={p.eyebrow}><Brain /> Train</span>
          <h1>Model projects</h1>
          <p>Обучение, импорт и выбор активной YOLO-модели для каждого dataset изделия.</p>
        </div>
        <div className={p.trainHeroStats}>
          <MetricPill icon={Database} label="Datasets" value={groups.length} />
          <MetricPill icon={Brain} label="Models" value={totalModels} />
          <MetricPill icon={CheckCircle2} label="Ready" value={readyProjects} />
          <MetricPill icon={Layers3} label="Classes" value={totalClasses} />
        </div>
      </section>

      <div className={p.trainIndexGrid}>
        <section className={p.panelCard}>
          <div className={p.cardTitleRow}>
            <div>
              <h3>Projects</h3>
              <p>Выбери dataset, проверь готовность данных и запускай training.</p>
            </div>
          </div>

          <QueryState isEmpty={!groups.length} size="block" emptyTitle="No projects" emptyDescription="Сначала создай dataset в Annotate.">
            <div className={p.trainProjectGrid}>
              {groups.map((group) => {
                const readiness = getReadiness(group);
                const labeledPercent = group.stats.images_count ? Math.round((group.stats.annotated_images_count / group.stats.images_count) * 100) : 0;

                return (
                  <Link className={p.trainProjectCard} key={group.id} to={paths.trainingGroup(group.id)}>
                    <div className={p.trainProjectTop}>
                      <span className={p.datasetAvatar}>{group.name.slice(0, 1).toUpperCase()}</span>
                      <div>
                        <strong>{group.name}</strong>
                        <small>Updated {formatDate(group.created_at)}</small>
                      </div>
                      <b>{readiness}%</b>
                    </div>
                    <p>{group.description || "Dataset для обучения модели проверки изделия."}</p>
                    <div className={p.trainProgressTrack}><span style={{ width: `${readiness}%` }} /></div>
                    <div className={p.trainProjectMeta}>
                      <span><Image /> {group.stats.images_count} images</span>
                      <span><Layers3 /> {group.stats.segment_classes_count} classes</span>
                      <span><Brain /> {group.stats.models_count} models</span>
                      <span>{labeledPercent}% labeled</span>
                    </div>
                  </Link>
                );
              })}
            </div>
          </QueryState>
        </section>

        <aside className={p.sidePanel}>
          <h3>Training pipeline</h3>
          <p>Для устойчивой проверки лучше держать один понятный pipeline от разметки до deployment.</p>
          <div className={p.workflowList}>
            <WorkflowItem icon={Database} title="Dataset" text={`${totalImages} images across all projects`} />
            <WorkflowItem icon={Sparkles} title="Annotate" text="Reference zones and class labels" />
            <WorkflowItem icon={Brain} title="Train" text={`${totalModels} trained/imported models`} />
            <WorkflowItem icon={Rocket} title="Deploy" text="Use active model in inspection" />
          </div>
        </aside>
      </div>
    </div>
  );
}

type IconComponent = typeof Brain;

function MetricPill({ icon: Icon, label, value }: { icon: IconComponent; label: string; value: number }) {
  return (
    <div className={p.metricPill}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function WorkflowItem({ icon: Icon, title, text }: { icon: IconComponent; title: string; text: string }) {
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
