import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { CreateGroup } from "@/page-components/groups/components/create-group";
import type { GroupListItem } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Activity, ArrowRight, Box, CheckCircle2, FolderKanban, Image, ListChecks, Search, Sparkles, Tags, type LucideIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import s from "./_project-assets-strict.module.scss";

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
  const { data } = useGetGroups();
  const groups = data ?? [];
  const [query, setQuery] = useState("");

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
    { references: 0, images: 0, labeled: 0, polygons: 0, classes: 0, models: 0, runs: 0 },
  );

  const filteredGroups = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return groups;
    return groups.filter((group) => [group.name, group.description ?? ""].some((value) => value.toLowerCase().includes(normalized)));
  }, [groups, query]);

  const readyProjects = groups.filter((group) => readiness(group) >= 80).length;
  const globalLabeling = percent(totals.labeled, totals.images);

  return (
    <div className={s.workspacePage}>
      <div className={s.page}>
        <header className={s.header}>
          <div>
            <span className={s.eyebrow}><FolderKanban /> Product workspace</span>
            <h1>Projects</h1>
            <p>Изделия, эталоны, классы, разметка, модели и проверки. Страница теперь работает как строгий project registry, а не как промо-блок.</p>
          </div>
          <div className={s.headerActions}>
            <CreateGroup />
          </div>
        </header>

        <section className={`${s.summaryGrid} ${s.summaryGridSix}`}>
          <Metric icon={FolderKanban} value={groups.length} label="Projects" hint={`${readyProjects} ready`} />
          <Metric icon={Box} value={totals.references} label="References" hint="эталонные виды" />
          <Metric icon={Image} value={totals.images} label="Images" hint={`${globalLabeling}% labeled`} />
          <Metric icon={Tags} value={totals.classes} label="Classes" hint={`${totals.polygons} polygons`} />
          <Metric icon={Sparkles} value={totals.models} label="Models" hint="trained/imported" />
          <Metric icon={ListChecks} value={totals.runs} label="Runs" hint="inspection history" />
        </section>

        <section className={s.contentGrid}>
          <main className={s.panel}>
            <div className={s.panelHead}>
              <div>
                <span>Registry</span>
                <h3>Изделия</h3>
              </div>
              <label className={s.searchBox}>
                <Search />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search projects..." />
              </label>
            </div>

            <div className={s.panelBody}>
              <QueryState isEmpty={!groups.length} size="block" emptyTitle="No projects yet" emptyDescription="Создай первое изделие, затем добавь эталоны, классы и модели.">
                <div className={s.projectGrid}>
                  {filteredGroups.map((group) => {
                    const score = readiness(group);
                    const labeled = percent(group.stats.annotated_images_count, group.stats.images_count);

                    return (
                      <Link className={s.projectCard} key={group.id} to={paths.groupDetail(group.id)}>
                        <div className={s.projectTop}>
                          <div className={s.avatar}>{group.name.slice(0, 1).toUpperCase()}</div>
                          <span className={score >= 80 ? s.donePill : s.openPill}>{score}% ready</span>
                        </div>

                        <div className={s.projectTitle}>
                          <strong>{group.name}</strong>
                          <p>{group.description || "Изделие с эталонами, разметкой, моделями и проверками."}</p>
                        </div>

                        <div className={s.progressTrack}><i style={{ width: `${score}%` }} /></div>

                        <div className={s.metaGrid}>
                          <span><Box /> {group.stats.standards_count} refs</span>
                          <span><Image /> {group.stats.images_count} images</span>
                          <span><CheckCircle2 /> {labeled}% labeled</span>
                          <span><Tags /> {group.stats.segment_classes_count} classes</span>
                          <span><Sparkles /> {group.stats.models_count} models</span>
                          <span><Activity /> {group.stats.inspections_count} runs</span>
                        </div>

                        <div className={s.taskRow}>
                          <ArrowRight />
                          <span><strong>{nextAction(group)}</strong><small>Created {formatDate(group.created_at)}</small></span>
                          <ArrowRight />
                        </div>
                      </Link>
                    );
                  })}
                </div>

                {groups.length > 0 && filteredGroups.length === 0 ? (
                  <div className={s.emptyInline}><Search /><strong>No projects match search</strong><span>Очисти поиск или создай новый проект.</span></div>
                ) : null}
              </QueryState>
            </div>
          </main>

          <aside className={s.panel}>
            <div className={s.panelHead}>
              <div><span>Lifecycle</span><h3>Зоны ответственности</h3></div>
            </div>
            <div className={s.panelBody}>
              <div className={s.taskList}>
                <Lifecycle icon={Image} title="Assets" text="References, images, classes and polygons." />
                <Lifecycle icon={Sparkles} title="Train" text="Models and training/import jobs." />
                <Lifecycle icon={ListChecks} title="Inspect" text="Photo, snapshot and realtime station." />
                <Lifecycle icon={Activity} title="Runs" text="Saved inspection history and reports." />
              </div>
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
}

function Metric({ icon: Icon, value, label, hint }: { icon: LucideIcon; value: number; label: string; hint: string }) {
  return (
    <div className={s.metricCard}>
      <Icon />
      <div><span>{label}</span><strong>{value}</strong><small>{hint}</small></div>
    </div>
  );
}

function Lifecycle({ icon: Icon, title, text }: { icon: LucideIcon; title: string; text: string }) {
  return (
    <div className={s.taskRow}>
      <Icon />
      <span><strong>{title}</strong><small>{text}</small></span>
      <ArrowRight />
    </div>
  );
}
