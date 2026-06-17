import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { ArrowRight, CheckCircle2, CircleAlert, Image, Images, ListChecks, Plus, Sparkles, Tags, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import s from "./_project-assets-strict.module.scss";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

function Metric({ icon: Icon, label, value, hint }: { icon: LucideIcon; label: string; value: string | number; hint: string }) {
  return (
    <div className={s.metricCard}>
      <Icon />
      <div><span>{label}</span><strong>{value}</strong><small>{hint}</small></div>
    </div>
  );
}

function TaskRow({ done, title, text, to }: { done: boolean; title: string; text: string; to: string }) {
  return (
    <Link className={`${s.taskRow} ${done ? s.taskRowDone : s.taskRowOpen}`} to={to}>
      {done ? <CheckCircle2 /> : <CircleAlert />}
      <span><strong>{title}</strong><small>{text}</small></span>
      <ArrowRight />
    </Link>
  );
}

export function Component() {
  const { group } = useGroupDetailOutletContext();
  const labeledPercent = percent(group.stats.annotated_images_count, group.stats.images_count);
  const hasReferences = group.stats.standards_count > 0;
  const hasImages = group.stats.images_count > 0;
  const hasClasses = group.stats.segment_classes_count > 0;
  const hasPolygons = group.stats.polygons_count > 0;
  const firstReference = group.standards.find((item) => item.reference_path) ?? group.standards[0] ?? null;

  const readyItems = [hasReferences, hasImages, hasClasses, hasPolygons];
  const readiness = Math.round((readyItems.filter(Boolean).length / readyItems.length) * 100);

  const nextAction = !hasReferences
    ? { title: "Create reference", text: "Добавь первый эталонный вид изделия.", to: paths.assetReferences(group.id), icon: Plus }
    : !hasImages
      ? { title: "Upload images", text: "Загрузи фото эталонного вида.", to: firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id), icon: Image }
      : !hasClasses
        ? { title: "Configure classes", text: "Создай детали изделия на странице Classes.", to: paths.assetClasses(group.id), icon: Tags }
        : group.stats.annotated_images_count < group.stats.images_count
          ? { title: "Continue labeling", text: "Разметь оставшиеся изображения в editor.", to: firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id), icon: Images }
          : { title: "Run inspection", text: "Эталонная база готова для проверки.", to: paths.inspectionGroup("photo", group.id), icon: ListChecks };

  const NextActionIcon = nextAction.icon;

  return (
    <div className={s.page}>
      <header className={`${s.header} ${s.headerCompact}`}>
        <div>
          <span className={s.eyebrow}><Images /> Assets / Overview</span>
          <h2>{group.name}</h2>
          <p>Готовность эталонной базы: references, изображения, классы деталей и полигоны.</p>
        </div>
        <div className={s.headerActions}>
          <CreateStandard groupId={group.id} />
          <Link to={paths.assetReferences(group.id)}>References</Link>
          <Link to={paths.assetClasses(group.id)}>Classes</Link>
        </div>
      </header>

      <section className={s.summaryGrid}>
        <Metric icon={Sparkles} label="Readiness" value={`${readiness}%`} hint="assets pipeline" />
        <Metric icon={Images} label="References" value={group.stats.standards_count} hint="эталонные виды" />
        <Metric icon={Image} label="Images" value={group.stats.images_count} hint={`${labeledPercent}% labeled`} />
        <Metric icon={Tags} label="Classes" value={group.stats.segment_classes_count} hint={`${group.stats.polygons_count} polygons`} />
      </section>

      <section className={s.contentGrid}>
        <main className={s.panel}>
          <div className={s.panelHead}>
            <div><span>References</span><h3>Рабочие эталоны</h3></div>
            <Link to={paths.assetReferences(group.id)}>Open library <ArrowRight /></Link>
          </div>
          <div className={s.panelBody}>
            <QueryState isEmpty={!group.standards.length} size="block" emptyTitle="References not created" emptyDescription="Создай эталонный вид, загрузи изображения и разметь обязательные детали.">
              <div className={s.scrollList}>
                {group.standards.slice(0, 12).map((reference) => {
                  const labeled = percent(reference.annotated_images_count, reference.images_count);
                  return (
                    <Link className={s.stackRow} key={reference.id} to={paths.standardDetail(group.id, reference.id)}>
                      <div className={s.previewThumb}>{reference.reference_path ? <img src={`/storage/${reference.reference_path}`} alt="" /> : <Images />}</div>
                      <span><strong>{reference.name}</strong><small>{reference.angle || "view"} · {reference.images_count} images · {labeled}% labeled</small></span>
                      <b className={reference.is_active ? s.activePill : s.draftPill}>{reference.is_active ? "active" : "draft"}</b>
                    </Link>
                  );
                })}
              </div>
            </QueryState>
          </div>
        </main>

        <aside className={s.panel}>
          <div className={s.panelHead}>
            <div><span>Next action</span><h3>{nextAction.title}</h3></div>
            <NextActionIcon />
          </div>
          <div className={s.panelBody}>
            <p className={s.mutedText}>{nextAction.text}</p>
            <div style={{ height: 12 }} />
            <Link className={s.primaryAction} to={nextAction.to}>Continue <ArrowRight /></Link>
            <div style={{ height: 14 }} />
            <div className={s.taskList}>
              <TaskRow done={hasReferences} title="Reference created" text={`${group.stats.standards_count} эталонных видов`} to={paths.assetReferences(group.id)} />
              <TaskRow done={hasImages} title="Images uploaded" text={`${group.stats.images_count} изображений`} to={paths.assetReferences(group.id)} />
              <TaskRow done={hasClasses} title="Classes configured" text={`${group.stats.segment_classes_count} классов деталей`} to={paths.assetClasses(group.id)} />
              <TaskRow done={hasPolygons} title="Polygons annotated" text={`${group.stats.polygons_count} полигонов`} to={firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id)} />
            </div>
          </div>
        </aside>
      </section>
    </div>
  );
}
