import { paths } from "@/app/paths";
import { RouteHero } from "@/components/layouts/route-hero/route-hero";
import { RoutePanel } from "@/components/layouts/route-panel/route-panel";
import { MetricCard } from "@/components/ui/metric-card/metric-card";
import { ReadinessItem } from "@/components/ui/readiness-item/readiness-item";
import QueryState from "@/components/ui/query-state/query-state";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { ArrowRight, CircleAlert, Image, Images, ListChecks, Plus, Sparkles, Tags } from "lucide-react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import s from "./_project-assets-strict.module.scss";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
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
      <RouteHero
        eyebrow="Assets / Overview"
        icon={Images}
        title={group.name}
        description="Готовность эталонной базы: references, изображения, классы деталей и полигоны."
        actionsClassName={s.headerActions}
        actions={(
          <>
          <CreateStandard groupId={group.id} />
          <Link to={paths.assetReferences(group.id)}>References</Link>
          <Link to={paths.assetClasses(group.id)}>Classes</Link>
          </>
        )}
      />

      <section className={s.summaryGrid}>
        <MetricCard className={s.metricCard} icon={Sparkles} label="Readiness" value={`${readiness}%`} hint="assets pipeline" />
        <MetricCard className={s.metricCard} icon={Images} label="References" value={group.stats.standards_count} hint="эталонные виды" />
        <MetricCard className={s.metricCard} icon={Image} label="Images" value={group.stats.images_count} hint={`${labeledPercent}% labeled`} />
        <MetricCard className={s.metricCard} icon={Tags} label="Classes" value={group.stats.segment_classes_count} hint={`${group.stats.polygons_count} polygons`} />
      </section>

      <section className={s.contentGrid}>
        <RoutePanel
          className={s.panel}
          headerClassName={s.panelHead}
          headingClassName={s.panelHeadInner}
          kickerClassName={s.panelKicker}
          titleClassName={s.panelTitle}
          bodyClassName={s.panelBody}
          kicker="References"
          title="Рабочие эталоны"
          actions={<Link to={paths.assetReferences(group.id)}>Open library <ArrowRight /></Link>}
        >
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
        </RoutePanel>

        <RoutePanel
          className={s.panel}
          headerClassName={s.panelHead}
          headingClassName={s.panelHeadInner}
          kickerClassName={s.panelKicker}
          titleClassName={s.panelTitle}
          actionsClassName={s.panelIconAction}
          bodyClassName={s.panelBody}
          kicker="Next action"
          title={nextAction.title}
          actions={<NextActionIcon />}
        >
            <p className={s.mutedText}>{nextAction.text}</p>
            <div style={{ height: 12 }} />
            <Link className={s.primaryAction} to={nextAction.to}>Continue <ArrowRight /></Link>
            <div style={{ height: 14 }} />
            <div className={s.taskList}>
              <ReadinessItem className={`${s.taskRow} ${hasReferences ? s.taskRowDone : s.taskRowOpen}`} done={hasReferences} pendingIcon={CircleAlert} title="Reference created" description={`${group.stats.standards_count} эталонных видов`} to={paths.assetReferences(group.id)} trailing={<ArrowRight />} />
              <ReadinessItem className={`${s.taskRow} ${hasImages ? s.taskRowDone : s.taskRowOpen}`} done={hasImages} pendingIcon={CircleAlert} title="Images uploaded" description={`${group.stats.images_count} изображений`} to={paths.assetReferences(group.id)} trailing={<ArrowRight />} />
              <ReadinessItem className={`${s.taskRow} ${hasClasses ? s.taskRowDone : s.taskRowOpen}`} done={hasClasses} pendingIcon={CircleAlert} title="Classes configured" description={`${group.stats.segment_classes_count} классов деталей`} to={paths.assetClasses(group.id)} trailing={<ArrowRight />} />
              <ReadinessItem className={`${s.taskRow} ${hasPolygons ? s.taskRowDone : s.taskRowOpen}`} done={hasPolygons} pendingIcon={CircleAlert} title="Polygons annotated" description={`${group.stats.polygons_count} полигонов`} to={firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id)} trailing={<ArrowRight />} />
            </div>
        </RoutePanel>
      </section>
    </div>
  );
}
