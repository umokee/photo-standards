import { paths } from "@/app/paths";
import QueryState from "@/components/ui/query-state/query-state";
import { ArrowRight, CheckCircle2, CircleDashed, Image, Images, Layers3, ListChecks, Sparkles, Tags, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "../platform-pages.module.scss";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

export function Component() {
  const { group } = useGroupDetailOutletContext();
  const firstReference = group.standards.find((item) => item.reference_path) ?? group.standards[0] ?? null;
  const classes = [...group.segment_class_categories.flatMap((category) => category.segment_classes), ...group.ungrouped_segment_classes];
  const labeledPercent = percent(group.stats.annotated_images_count, group.stats.images_count);
  const readinessChecks = [
    { title: "References", done: group.stats.standards_count > 0, text: `${group.stats.standards_count} эталонных видов` },
    { title: "Images", done: group.stats.images_count > 0, text: `${group.stats.images_count} изображений` },
    { title: "Classes", done: group.stats.segment_classes_count > 0, text: `${group.stats.segment_classes_count} классов деталей` },
    { title: "Polygons", done: group.stats.polygons_count > 0, text: `${group.stats.polygons_count} зон контроля` },
  ];

  const readyScore = readinessChecks.filter((item) => item.done).length * 25;
  const nextAction =
    !group.stats.standards_count
      ? { title: "Создать первый reference", text: "Начни с эталонного вида изделия.", to: paths.assetReferences(group.id), icon: Images }
      : !group.stats.segment_classes_count
        ? { title: "Настроить классы", text: "Опиши обязательные детали изделия.", to: paths.assetClasses(group.id), icon: Tags }
        : group.stats.images_count !== group.stats.annotated_images_count
          ? { title: "Продолжить разметку", text: "Доведи фото до размеченного состояния.", to: firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id), icon: Image }
          : { title: "Перейти к проверке", text: "Эталонная база готова для Inspect.", to: paths.inspectionGroup("photo", group.id), icon: ListChecks };

  return (
    <div className={p.assetsOverviewV23}>
      <header className={p.zonePageHeadV23}>
        <div>
          <span className={p.eyebrow}><Images /> Assets / Overview</span>
          <h1>Эталонная база</h1>
          <p>Короткий статус assets текущего изделия: references, фотографии, классы и полигоны. Создание и редактирование вынесено в отдельные страницы.</p>
        </div>
      </header>

      <section className={p.assetsOverviewGridV23}>
        <article className={p.assetReadinessPanelV23}>
          <div className={p.assetReadinessTopV23}>
            <div>
              <span>Assets readiness</span>
              <strong>{readyScore}%</strong>
              <p>{group.stats.standards_count} refs · {labeledPercent}% labeled · {group.stats.segment_classes_count} classes</p>
            </div>
            <div className={p.assetReadinessBadgeV23}>{readyScore === 100 ? "ready" : "setup"}</div>
          </div>
          <div className={p.progressTrackV16}><i style={{ width: `${readyScore}%` }} /></div>

          <div className={p.assetChecklistCompactV23}>
            {readinessChecks.map((item) => (
              <div key={item.title}>
                {item.done ? <CheckCircle2 /> : <CircleDashed />}
                <span>
                  <strong>{item.title}</strong>
                  <small>{item.text}</small>
                </span>
              </div>
            ))}
          </div>
        </article>

        <article className={p.assetNextActionPanelV23}>
          {(() => {
            const Icon = nextAction.icon;
            return <Icon />;
          })()}
          <span>Next action</span>
          <strong>{nextAction.title}</strong>
          <p>{nextAction.text}</p>
          <Link to={nextAction.to}>Open <ArrowRight /></Link>
        </article>
      </section>

      <section className={p.assetResponsibilityGridV23}>
        <ResponsibilityCard
          icon={Images}
          title="References"
          text="Создание эталонных видов, загрузка фото и вход в редактор разметки."
          value={`${group.stats.standards_count} refs`}
          to={paths.assetReferences(group.id)}
        />
        <ResponsibilityCard
          icon={Tags}
          title="Classes"
          text="Единое место для классов деталей, категорий и цветов."
          value={`${group.stats.segment_classes_count} classes`}
          to={paths.assetClasses(group.id)}
        />
        <ResponsibilityCard
          icon={Image}
          title="Annotation queue"
          text="Работа с полигонами выполняется внутри выбранного reference."
          value={`${group.stats.polygons_count} polygons`}
          to={firstReference ? paths.standardDetail(group.id, firstReference.id) : paths.assetReferences(group.id)}
        />
      </section>

      <section className={p.assetsSnapshotGridV23}>
        <article className={p.assetSnapshotPanelV23}>
          <div className={p.assetSnapshotHeadV23}>
            <div>
              <span>References snapshot</span>
              <h3>Эталонные виды</h3>
            </div>
            <Link to={paths.assetReferences(group.id)}>All references</Link>
          </div>

          <QueryState
            isEmpty={!group.standards.length}
            size="block"
            emptyTitle="References not created"
            emptyDescription="Создай reference на странице Assets / References."
          >
            <div className={p.assetReferenceRowsV23}>
              {group.standards.slice(0, 4).map((reference) => {
                const labeled = percent(reference.annotated_images_count, reference.images_count);
                return (
                  <Link key={reference.id} to={paths.standardDetail(group.id, reference.id)}>
                    <span>
                      {reference.reference_path ? <img src={`/storage/${reference.reference_path}`} alt="" /> : <Image />}
                    </span>
                    <strong>{reference.name}</strong>
                    <small>{reference.images_count} images · {labeled}% labeled</small>
                  </Link>
                );
              })}
            </div>
          </QueryState>
        </article>

        <article className={p.assetSnapshotPanelV23}>
          <div className={p.assetSnapshotHeadV23}>
            <div>
              <span>Classes snapshot</span>
              <h3>Детали изделия</h3>
            </div>
            <Link to={paths.assetClasses(group.id)}>Edit classes</Link>
          </div>

          <div className={p.classTokenCloudV23}>
            {classes.slice(0, 18).map((item) => (
              <span key={item.id} style={{ ["--hue" as string]: item.hue }}>{item.name}</span>
            ))}
            {!classes.length ? (
              <small>Классы ещё не настроены. Открой Assets / Classes.</small>
            ) : null}
          </div>
        </article>
      </section>
    </div>
  );
}

function ResponsibilityCard({ icon: Icon, title, text, value, to }: { icon: LucideIcon; title: string; text: string; value: string; to: string }) {
  return (
    <Link className={p.assetResponsibilityCardV23} to={to}>
      <Icon />
      <span>{value}</span>
      <strong>{title}</strong>
      <p>{text}</p>
      <b>Open <ArrowRight /></b>
    </Link>
  );
}
