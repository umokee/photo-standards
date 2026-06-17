import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { CreateGroup } from "@/page-components/groups/components/create-group";
import type { GroupListItem } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Activity, Box, CheckCircle2, FolderKanban, Image, ListChecks, Plus, Sparkles, Tags, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import p from "../platform-pages.module.scss";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

function readiness(group: GroupListItem) {
  const checks = [
    group.stats.standards_count > 0,
    group.stats.images_count > 0,
    group.stats.segment_classes_count > 0,
    group.stats.polygons_count > 0,
    group.stats.models_count > 0,
  ];
  return Math.round((checks.filter(Boolean).length / checks.length) * 100);
}

function nextAction(group: GroupListItem) {
  if (!group.stats.standards_count) return "Create reference";
  if (!group.stats.images_count) return "Upload images";
  if (!group.stats.segment_classes_count) return "Configure classes";
  if (!group.stats.polygons_count) return "Annotate polygons";
  if (!group.stats.models_count) return "Train model";
  return group.stats.inspections_count ? "Review runs" : "Run inspection";
}

export function Component() {
  const { data: groups } = useGetGroups();
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
  const readyProjects = groups.filter((group) => readiness(group) >= 80).length;
  const globalLabeling = percent(totals.labeled, totals.images);

  return (
    <div className={`${p.page} ${p.projectsWorkspaceV19}`}>
      <header className={p.workspaceHeroV19}>
        <div>
          <span className={p.eyebrow}><FolderKanban /> Project workspace</span>
          <h1>Projects</h1>
          <p>Проект — это изделие. Внутри него связаны references, images, classes, polygons, models и inspection runs.</p>
        </div>
        <div className={p.headerActions}><CreateGroup /></div>
      </header>

      <section className={p.projectMetricGridV19}>
        <Metric icon={FolderKanban} value={groups.length} label="Projects" hint={`${readyProjects} inspection-ready`} />
        <Metric icon={Image} value={totals.images} label="Images" hint={`${globalLabeling}% labeled`} />
        <Metric icon={Tags} value={totals.classes} label="Classes" hint={`${totals.polygons} polygons`} />
        <Metric icon={Sparkles} value={totals.models} label="Models" hint="trained/imported" />
        <Metric icon={ListChecks} value={totals.runs} label="Runs" hint="inspection history" />
      </section>

      <div className={p.projectsLayoutV19}>
        <section className={p.surfacePanelV16}>
          <div className={p.panelHeadV16}>
            <div>
              <h3>Product queue</h3>
              <p>Выбирай изделие и работай дальше внутри одного project context.</p>
            </div>
            <span>{groups.length} projects</span>
          </div>

          <QueryState isEmpty={!groups.length} size="block" emptyTitle="No projects yet" emptyDescription="Создай первое изделие, затем добавь эталоны и классы.">
            <div className={p.projectQueueV19}>
              {groups.map((group) => {
                const score = readiness(group);
                const labeled = percent(group.stats.annotated_images_count, group.stats.images_count);

                return (
                  <Link className={p.projectQueueItemV19} key={group.id} to={paths.groupDetail(group.id)}>
                    <div className={p.projectQueueAvatarV19}>{group.name.slice(0, 1).toUpperCase()}</div>
                    <div className={p.projectQueueMainV19}>
                      <div className={p.projectQueueTitleV19}>
                        <strong>{group.name}</strong>
                        <span>{score}% ready</span>
                      </div>
                      <p>{group.description || "Изделие с эталонами, разметкой, моделями и проверками."}</p>
                      <div className={p.progressTrackV16}><span style={{ width: `${score}%` }} /></div>
                      <div className={p.projectQueueMetaV19}>
                        <span><Image /> {group.stats.images_count} images</span>
                        <span><CheckCircle2 /> {labeled}% labeled</span>
                        <span><Box /> {group.stats.standards_count} refs</span>
                        <span><Sparkles /> {group.stats.models_count} models</span>
                        <span><Activity /> {group.stats.inspections_count} runs</span>
                      </div>
                    </div>
                    <div className={p.projectQueueActionV19}>
                      <small>Next action</small>
                      <b>{nextAction(group)}</b>
                      <em>{formatDate(group.created_at)}</em>
                    </div>
                  </Link>
                );
              })}
            </div>
          </QueryState>
        </section>

        <aside className={p.commandRailV16}>
          <section className={p.surfacePanelV16}>
            <div className={p.panelHeadV16}><div><h3>Project structure</h3><p>Один project собирает весь QC lifecycle.</p></div></div>
            <div className={p.workflowStepsV16}>
              <Step icon={Image} title="Assets" text="References, images, classes and polygons." />
              <Step icon={Sparkles} title="Train" text="Models trained/imported for this product." />
              <Step icon={ListChecks} title="Inspect" text="Photo, snapshot or realtime quality check." />
              <Step icon={Activity} title="Runs" text="Saved reports with matched/missing components." />
            </div>
          </section>

          <section className={p.surfacePanelV16}>
            <div className={p.panelHeadV16}><div><h3>Next global action</h3><p>Самая частая причина, почему проверка ещё не готова.</p></div></div>
            <div className={p.actionHintV16}>
              <Plus />
              <div>
                <strong>{groups.length ? "Open project overview" : "Create first project"}</strong>
                <span>{groups.length ? "Теперь основная работа идёт внутри выбранного изделия." : "Проект создаёт контейнер для эталонов, моделей и проверок."}</span>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}

function Metric({ icon: Icon, value, label, hint }: { icon: LucideIcon; value: number; label: string; hint: string }) {
  return (
    <div className={p.metricCardV16}>
      <Icon />
      <b>{value}</b>
      <span>{label}</span>
      <small>{hint}</small>
    </div>
  );
}

function Step({ icon: Icon, title, text }: { icon: LucideIcon; title: string; text: string }) {
  return (
    <div className={p.workflowStepV16}>
      <Icon />
      <div><strong>{title}</strong><span>{text}</span></div>
    </div>
  );
}
