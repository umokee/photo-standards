import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import { CreateGroup } from "@/page-components/groups/components/create-group";
import type { GroupListItem } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { Activity, BarChart3, Box, CheckCircle2, Database, Image, Layers3, ListChecks, Plus, Tags, type LucideIcon } from "lucide-react";
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
    group.stats.annotated_images_count > 0,
  ];
  return Math.round((checks.filter(Boolean).length / checks.length) * 100);
}

export function Component() {
  const { data: groups } = useGetGroups();
  const totals = groups.reduce(
    (acc, group) => ({
      standards: acc.standards + group.stats.standards_count,
      images: acc.images + group.stats.images_count,
      labeled: acc.labeled + group.stats.annotated_images_count,
      polygons: acc.polygons + group.stats.polygons_count,
      classes: acc.classes + group.stats.segment_classes_count,
      models: acc.models + group.stats.models_count,
      runs: acc.runs + group.stats.inspections_count,
    }),
    { standards: 0, images: 0, labeled: 0, polygons: 0, classes: 0, models: 0, runs: 0 }
  );
  const globalLabeling = percent(totals.labeled, totals.images);
  const readyDatasets = groups.filter((group) => readiness(group) >= 80).length;

  return (
    <div className={p.page}>
      <header className={p.workspaceHeaderV16}>
        <div>
          <span className={p.eyebrow}><Database /> Annotate workspace</span>
          <h1>Datasets</h1>
          <p>Изделия, эталонные виды, изображения и полигоны. Отсюда начинается вся проверка.</p>
        </div>
        <div className={p.headerActions}><CreateGroup /></div>
      </header>

      <section className={p.commandMetricGridV16}>
        <Metric icon={Database} value={groups.length} label="Datasets" hint={`${readyDatasets} ready`} />
        <Metric icon={Image} value={totals.images} label="Images" hint={`${globalLabeling}% labeled`} />
        <Metric icon={Tags} value={totals.classes} label="Classes" hint="segment labels" />
        <Metric icon={Box} value={totals.polygons} label="Polygons" hint={`${totals.standards} refs`} />
        <Metric icon={ListChecks} value={totals.runs} label="Checks" hint="saved runs" />
      </section>

      <div className={p.datasetsCommandGridV16}>
        <section className={p.surfacePanelV16}>
          <div className={p.panelHeadV16}>
            <div>
              <h3>Dataset queue</h3>
              <p>Открывай dataset, добавляй эталоны и добивай готовность до проверки.</p>
            </div>
            <span>{globalLabeling}% labeled</span>
          </div>

          <QueryState isEmpty={!groups.length} size="block" emptyTitle="Нет datasets" emptyDescription="Создай первый dataset изделия для эталонов.">
            <div className={p.datasetQueueV16}>
              {groups.map((group) => {
                const labelPercent = percent(group.stats.annotated_images_count, group.stats.images_count);
                const score = readiness(group);

                return (
                  <Link className={p.datasetQueueItemV16} key={group.id} to={paths.groupDetail(group.id)}>
                    <div className={p.queueAvatarV16}>{group.name.slice(0, 1).toUpperCase()}</div>
                    <div className={p.queueMainV16}>
                      <div className={p.queueTitleV16}>
                        <strong>{group.name}</strong>
                        <span>{score}% ready</span>
                      </div>
                      <p>{group.description || "Dataset изделия с эталонами и контрольными зонами."}</p>
                      <div className={p.progressTrackV16}><span style={{ width: `${score}%` }} /></div>
                      <div className={p.queueMetaV16}>
                        <span><Image /> {group.stats.images_count} images</span>
                        <span><CheckCircle2 /> {group.stats.annotated_images_count} labeled</span>
                        <span><Tags /> {group.stats.segment_classes_count} classes</span>
                        <span><Box /> {group.stats.polygons_count} polygons</span>
                      </div>
                    </div>
                    <div className={p.queueAsideV16}>
                      <b>{labelPercent}%</b>
                      <small>labeling</small>
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
            <div className={p.panelHeadV16}><div><h3>QC pipeline</h3><p>Минимальный путь без лишней сложности.</p></div></div>
            <div className={p.workflowStepsV16}>
              <Step icon={Database} title="1. Reference" text="Создай один или несколько эталонных видов." />
              <Step icon={Image} title="2. Images" text="Загрузи набор кадров под этот эталон." />
              <Step icon={Tags} title="3. Polygons" text="Разметь обязательные детали и зоны." />
              <Step icon={BarChart3} title="4. Model" text="Обучи или импортируй YOLO-веса." />
              <Step icon={Activity} title="5. Inspect" text="Запусти проверку фото или камеры." />
            </div>
          </section>

          <section className={p.surfacePanelV16}>
            <div className={p.panelHeadV16}><div><h3>Next action</h3><p>Что обычно мешает проверке.</p></div></div>
            <div className={p.actionHintV16}>
              <Plus />
              <div>
                <strong>{totals.standards ? "Добей разметку" : "Создай эталон"}</strong>
                <span>{totals.standards ? `${totals.images - totals.labeled} images still without polygons` : "Нужен reference view, чтобы начать dataset workflow."}</span>
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
