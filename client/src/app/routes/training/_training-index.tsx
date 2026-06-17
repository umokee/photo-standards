import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetGroups } from "@/page-components/groups/api/get-groups";
import type { GroupListItem } from "@/types/contracts";
import { Activity, Brain, CheckCircle2, Database, Image, Layers3, ListChecks, Play, ShieldCheck, Tags, type LucideIcon } from "lucide-react";
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

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

export function Component() {
  const { data: groups } = useGetGroups();
  const totalModels = groups.reduce((sum, group) => sum + group.stats.models_count, 0);
  const totalImages = groups.reduce((sum, group) => sum + group.stats.images_count, 0);
  const totalLabeled = groups.reduce((sum, group) => sum + group.stats.annotated_images_count, 0);
  const readyProjects = groups.filter((group) => getReadiness(group) >= 75).length;
  const trainingReady = groups.filter((group) => group.stats.images_count > 0 && group.stats.annotated_images_count > 0 && group.stats.segment_classes_count > 0).length;

  return (
    <div className={p.page}>
      <header className={p.workspaceHeaderV16}>
        <div>
          <span className={p.eyebrow}><Brain /> Train workspace</span>
          <h1>Models</h1>
          <p>Очередь проектов для обучения: готовность данных, классы, активные веса и связь с Inspect.</p>
        </div>
      </header>

      <section className={p.commandMetricGridV16}>
        <Metric icon={Database} value={groups.length} label="Projects" hint={`${trainingReady} train-ready`} />
        <Metric icon={Image} value={totalImages} label="Images" hint={`${percent(totalLabeled, totalImages)}% labeled`} />
        <Metric icon={Brain} value={totalModels} label="Models" hint="weights" />
        <Metric icon={CheckCircle2} value={readyProjects} label="Ready" hint="quality gate" />
      </section>

      <div className={p.trainCommandGridV16}>
        <section className={p.surfacePanelV16}>
          <div className={p.panelHeadV16}>
            <div><h3>Training queue</h3><p>Проекты отсортированы как карточки готовности к обучению.</p></div>
            <span>{totalModels} models</span>
          </div>

          <QueryState isEmpty={!groups.length} size="block" emptyTitle="No training projects" emptyDescription="Сначала создай dataset в Annotate.">
            <div className={p.trainingQueueV16}>
              {groups.map((group) => {
                const readiness = getReadiness(group);
                const labeled = percent(group.stats.annotated_images_count, group.stats.images_count);
                const canTrain = group.stats.images_count > 0 && group.stats.annotated_images_count > 0 && group.stats.segment_classes_count > 0;

                return (
                  <Link className={p.trainingQueueItemV16} key={group.id} to={paths.trainingGroup(group.id)}>
                    <div className={p.queueAvatarV16}>{group.name.slice(0, 1).toUpperCase()}</div>
                    <div className={p.queueMainV16}>
                      <div className={p.queueTitleV16}>
                        <strong>{group.name}</strong>
                        <span>{canTrain ? "ready" : "needs data"}</span>
                      </div>
                      <div className={p.progressTrackV16}><span style={{ width: `${readiness}%` }} /></div>
                      <div className={p.queueMetaV16}>
                        <span><Image /> {group.stats.images_count} images</span>
                        <span><Tags /> {group.stats.segment_classes_count} classes</span>
                        <span><Layers3 /> {labeled}% labeled</span>
                        <span><Brain /> {group.stats.models_count} models</span>
                      </div>
                    </div>
                    <div className={p.trainingLaunchV16}>
                      <Play />
                      <b>{readiness}%</b>
                    </div>
                  </Link>
                );
              })}
            </div>
          </QueryState>
        </section>

        <aside className={p.commandRailV16}>
          <section className={p.surfacePanelV16}>
            <div className={p.panelHeadV16}><div><h3>Quality gates</h3><p>Не запускай обучение вслепую.</p></div></div>
            <div className={p.workflowStepsV16}>
              <Step icon={Database} title="Dataset" text="Есть reference views и изображения." />
              <Step icon={Tags} title="Classes" text="Классы сегментации настроены." />
              <Step icon={ShieldCheck} title="Annotations" text="Есть полигоны на кадрах." />
              <Step icon={Brain} title="Model" text="После обучения модель попадёт в Inspect." />
            </div>
          </section>

          <section className={p.surfacePanelV16}>
            <div className={p.panelHeadV16}><div><h3>Lifecycle</h3><p>Что должно происходить после обучения.</p></div></div>
            <div className={p.actionHintV16}>
              <Activity />
              <div>
                <strong>Train → activate → inspect</strong>
                <span>Модель имеет смысл только как проверка эталонных зон, поэтому активную версию держим рядом с проектом.</span>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}

function Metric({ icon: Icon, value, label, hint }: { icon: LucideIcon; value: number; label: string; hint: string }) {
  return <div className={p.metricCardV16}><Icon /><b>{value}</b><span>{label}</span><small>{hint}</small></div>;
}

function Step({ icon: Icon, title, text }: { icon: LucideIcon; title: string; text: string }) {
  return <div className={p.workflowStepV16}><Icon /><div><strong>{title}</strong><span>{text}</span></div></div>;
}
