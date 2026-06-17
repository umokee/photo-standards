import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { CreateStandard } from "@/page-components/standards/components/create-standard";
import { ArrowRight, CheckCircle2, CircleAlert, Image, Images, Layers3, ListChecks, Plus, Sparkles, Tags, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "../platform-pages.module.scss";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

function AssetMetric({ icon: Icon, label, value, hint }: { icon: LucideIcon; label: string; value: string | number; hint: string }) {
  return (
    <div className={p.assetMetricV26}>
      <Icon />
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}

function TaskRow({ done, title, text, to }: { done: boolean; title: string; text: string; to: string }) {
  return (
    <Link className={done ? p.assetTaskDoneV26 : p.assetTaskOpenV26} to={to}>
      {done ? <CheckCircle2 /> : <CircleAlert />}
      <span>
        <strong>{title}</strong>
        <small>{text}</small>
      </span>
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
          ? { title: "Continue labeling", text: "Разметь оставшиеся изображения в редакторе.", to: firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id), icon: Images }
          : { title: "Run inspection", text: "Эталонная база готова для проверки.", to: paths.inspectionGroup("photo", group.id), icon: ListChecks };

  const NextActionIcon = nextAction.icon;

  return (
    <div className={p.assetPageV26}>
      <section className={p.assetHeaderV26}>
        <div>
          <span className={p.assetEyebrowV26}><Images /> Assets</span>
          <h2>Эталонная база</h2>
          <p>References, изображения, классы деталей и полигоны этого изделия. Этот раздел готовит данные для Train и Inspect.</p>
        </div>
        <div className={p.assetHeaderActionsV26}>
          <CreateStandard groupId={group.id} />
          <Link to={paths.assetClasses(group.id)}>Classes</Link>
        </div>
      </section>

      <section className={p.assetOverviewGridV26}>
        <div className={p.assetReadinessCardV26}>
          <span className={p.assetEyebrowV26}><Sparkles /> Readiness</span>
          <strong>{readiness}%</strong>
          <p>{hasReferences && hasImages && hasClasses ? "База почти готова: проверь разметку и можно использовать в Train/Inspect." : "Нужно закрыть базовые шаги перед обучением и проверкой."}</p>
          <div className={p.assetProgressTrackV26}><i style={{ width: `${readiness}%` }} /></div>
        </div>

        <AssetMetric icon={Images} label="References" value={group.stats.standards_count} hint="эталонные виды" />
        <AssetMetric icon={Image} label="Images" value={group.stats.images_count} hint={`${labeledPercent}% labeled`} />
        <AssetMetric icon={Tags} label="Classes" value={group.stats.segment_classes_count} hint="детали изделия" />
        <AssetMetric icon={Layers3} label="Polygons" value={group.stats.polygons_count} hint="зоны контроля" />
      </section>

      <section className={p.assetWorkGridV26}>
        <main className={p.assetPanelV26}>
          <div className={p.assetPanelHeadV26}>
            <div>
              <span>References</span>
              <h3>Рабочие эталоны</h3>
            </div>
            <Link to={paths.assetReferences(group.id)}>Open library <ArrowRight /></Link>
          </div>

          <QueryState
            isEmpty={!group.standards.length}
            size="block"
            emptyTitle="References not created"
            emptyDescription="Создай эталонный вид, загрузи изображения и разметь обязательные детали."
          >
            <div className={p.referencePreviewStackV26}>
              {group.standards.slice(0, 6).map((reference) => {
                const labeled = percent(reference.annotated_images_count, reference.images_count);
                return (
                  <Link className={p.referencePreviewRowV26} key={reference.id} to={paths.standardDetail(group.id, reference.id)}>
                    <div className={p.referencePreviewImageV26}>
                      {reference.reference_path ? <img src={`/storage/${reference.reference_path}`} alt="" /> : <Images />}
                    </div>
                    <span>
                      <strong>{reference.name}</strong>
                      <small>{reference.angle || "view"} · {reference.images_count} images · {labeled}% labeled</small>
                    </span>
                    <b>{reference.is_active ? "active" : "draft"}</b>
                  </Link>
                );
              })}
            </div>
          </QueryState>
        </main>

        <aside className={p.assetPanelV26}>
          <div className={p.assetPanelHeadV26}>
            <div>
              <span>Next action</span>
              <h3>{nextAction.title}</h3>
            </div>
            <NextActionIcon />
          </div>
          <p className={p.assetPanelTextV26}>{nextAction.text}</p>
          <Link className={p.assetPrimaryLinkV26} to={nextAction.to}>Continue <ArrowRight /></Link>

          <div className={p.assetTaskListV26}>
            <TaskRow done={hasReferences} title="Reference created" text={`${group.stats.standards_count} эталонных видов`} to={paths.assetReferences(group.id)} />
            <TaskRow done={hasImages} title="Images uploaded" text={`${group.stats.images_count} изображений`} to={paths.assetReferences(group.id)} />
            <TaskRow done={hasClasses} title="Classes configured" text={`${group.stats.segment_classes_count} классов деталей`} to={paths.assetClasses(group.id)} />
            <TaskRow done={hasPolygons} title="Polygons annotated" text={`${group.stats.polygons_count} полигонов`} to={firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id)} />
          </div>
        </aside>
      </section>
    </div>
  );
}
