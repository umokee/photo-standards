import { paths } from "@/app/paths";
import { QueryBoundary } from "@/components/ui/query-boundary/query-boundary";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetStandardDetail } from "@/page-components/standards/api/get-standard";
import { SegmentClass } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import clsx from "clsx";
import { BarChart3, CheckCircle2, CircleDashed, Image, Layers3, PenLine, Search, ShieldCheck, Star, Tags, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { useLoaderData, useNavigate } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "../platform-pages.module.scss";

type StandardPanelTab = "images" | "classes" | "overview";
type ClassRow = SegmentClass & { categoryName: string };

export function Component() {
  const navigate = useNavigate();
  const { standardId } = useLoaderData() as { standardId: string | null };
  const { group } = useGroupDetailOutletContext();

  const selectedStandard =
    group.standards.find((standard) => standard.id === standardId) ?? group.standards[0] ?? null;

  const handleOpenStandard = (targetStandardId: string) => {
    navigate(paths.standardDetail(group.id, targetStandardId));
  };

  return (
    <QueryState
      isEmpty={!group.standards.length}
      size="page"
      emptyTitle="No reference views"
      emptyDescription="Создай эталон, загрузи изображения и разметь контрольные зоны."
    >
      <div className={p.referenceDashboardGrid}>
        <section className={p.panelCard}>
          <div className={p.cardTitleRow}>
            <div>
              <h3>Reference views</h3>
              <p>{group.stats.standards_count} views · {group.stats.images_count} images · {group.stats.polygons_count} polygons</p>
            </div>
            <span className={p.softPill}>Annotate</span>
          </div>

          <div className={p.referenceListCompact}>
            {group.standards.map((standard, index) => {
              const isActive = selectedStandard?.id === standard.id;
              const progress = standard.images_count
                ? Math.round((standard.annotated_images_count / standard.images_count) * 100)
                : 0;

              return (
                <button
                  type="button"
                  className={clsx(p.referenceCompactCard, isActive && p.referenceCompactCardActive)}
                  key={standard.id}
                  onClick={() => handleOpenStandard(standard.id)}
                >
                  <div className={p.referenceCompactPreview}>
                    {standard.reference_path ? <img src={`/storage/${standard.reference_path}`} alt="" /> : <Image />}
                    <span>{String(index + 1).padStart(2, "0")}</span>
                  </div>
                  <div className={p.referenceCompactBody}>
                    <strong>{standard.name}</strong>
                    <small>{standard.angle || "without angle"}</small>
                    <div className={p.referenceProgress}><span style={{ width: `${progress}%` }} /></div>
                    <div className={p.referenceMeta}>
                      <span>{standard.images_count} images</span>
                      <span>{standard.annotated_images_count} labeled</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        <aside className={p.sidePanel}>
          <h3>Dataset checklist</h3>
          <p>Быстрый контроль готовности перед запуском проверки.</p>
          <div className={p.checklistPanel}>
            <ChecklistItem done={group.stats.standards_count > 0} title="Reference created" text={`${group.stats.standards_count} standard views`} />
            <ChecklistItem done={group.stats.images_count > 0} title="Images uploaded" text={`${group.stats.images_count} images`} />
            <ChecklistItem done={group.stats.segment_classes_count > 0} title="Classes configured" text={`${group.stats.segment_classes_count} classes`} />
            <ChecklistItem done={group.stats.polygons_count > 0} title="Polygons annotated" text={`${group.stats.polygons_count} polygons`} />
          </div>
          <div className={p.quickHint}>
            <PenLine />
            <span>Клик по изображению открывает editor. Для проверки используй Inspect после разметки.</span>
          </div>
        </aside>
      </div>

      {selectedStandard ? (
        <QueryBoundary loadingText="Загрузка изображений эталона..." errorTitle="Не удалось загрузить изображения">
          <StandardWorkspacePanel groupId={group.id} standardId={selectedStandard.id} />
        </QueryBoundary>
      ) : null}
    </QueryState>
  );
}

function StandardWorkspacePanel({ groupId, standardId }: { groupId: string; standardId: string }) {
  const navigate = useNavigate();
  const [tab, setTab] = useState<StandardPanelTab>("images");
  const [query, setQuery] = useState("");
  const { data: standard } = useGetStandardDetail(standardId);
  const totalAnnotations = standard.images.reduce((sum, image) => sum + image.annotation_count, 0);
  const labeledImages = standard.images.filter((image) => image.annotation_count > 0).length;
  const labeledPercent = standard.images.length ? Math.round((labeledImages / standard.images.length) * 100) : 0;

  const classRows = useMemo<ClassRow[]>(() => {
    return [
      ...standard.segment_class_categories.flatMap((category) =>
        category.segment_classes.map((item) => ({ ...item, categoryName: category.name }))
      ),
      ...standard.ungrouped_segment_classes.map((item) => ({ ...item, categoryName: "Ungrouped" })),
    ];
  }, [standard.segment_class_categories, standard.ungrouped_segment_classes]);

  const filteredClassRows = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return classRows;
    return classRows.filter((item) => `${item.name} ${item.categoryName}`.toLowerCase().includes(normalized));
  }, [classRows, query]);

  return (
    <section className={p.datasetWorkspaceCard}>
      <header className={p.standardWorkspaceHeader}>
        <div className={p.standardTitleBlock}>
          <div className={p.datasetBreadcrumbLine}>Reference <span>/</span> {standard.angle || "view"}</div>
          <h3>{standard.name}</h3>
          <p>{standard.images.length} images · {totalAnnotations} polygons · updated {formatDate(standard.created_at)}</p>
        </div>
        <div className={p.standardHealthCards}>
          <MiniHealth icon={Image} label="Images" value={standard.images.length} />
          <MiniHealth icon={Tags} label="Classes" value={classRows.length} />
          <MiniHealth icon={ShieldCheck} label="Labeled" value={`${labeledPercent}%`} />
        </div>
      </header>

      <div className={p.standardTabBar}>
        <button className={clsx(tab === "images" && p.active)} type="button" onClick={() => setTab("images")}><Image /> Images <sup>{standard.images.length}</sup></button>
        <button className={clsx(tab === "classes" && p.active)} type="button" onClick={() => setTab("classes")}><Tags /> Classes <sup>{classRows.length}</sup></button>
        <button className={clsx(tab === "overview" && p.active)} type="button" onClick={() => setTab("overview")}><BarChart3 /> Overview</button>
      </div>

      {tab === "images" ? (
        standard.images.length ? (
          <div className={p.ultraImageGrid}>
            {standard.images.map((image, index) => {
              const isAnnotated = image.annotation_count > 0;
              return (
                <button
                  type="button"
                  className={p.ultraImageCard}
                  key={image.id}
                  onClick={() => navigate(paths.standardImage(groupId, standard.id, image.id))}
                >
                  <img src={`/storage/${image.image_path}`} alt={`Image ${index + 1}`} />
                  <span className={p.imageIndex}>#{index + 1}</span>
                  {image.is_reference ? <span className={p.imageRef}><Star /> Ref</span> : null}
                  <div className={p.imageCardMeta}>
                    <span className={clsx(p.imageCardState, isAnnotated && p.imageCardStateDone)}>
                      {isAnnotated ? <CheckCircle2 /> : <CircleDashed />}
                      {isAnnotated ? `${image.annotation_count} polygons` : "not labeled"}
                    </span>
                    <span>Open editor →</span>
                  </div>
                </button>
              );
            })}
          </div>
        ) : (
          <div className={p.emptyImageDrop}>
            <Upload />
            <strong>No images yet</strong>
            <span>Загрузи изображения для этого эталона, потом они появятся здесь.</span>
          </div>
        )
      ) : null}

      {tab === "classes" ? (
        <div className={p.classesWorkspace}>
          <div className={p.classSearchBox}>
            <Search />
            <input value={query} placeholder="Search classes..." onChange={(event) => setQuery(event.target.value)} />
          </div>
          <div className={p.classTable}>
            <div className={p.classTableHead}><span>Class</span><span>Group</span><span>Hue</span></div>
            {filteredClassRows.map((item) => (
              <div className={p.classTableRow} key={item.id}>
                <span><i style={{ background: `hsl(${item.hue}, 70%, 50%)` }} />{item.name}</span>
                <span>{item.categoryName}</span>
                <span>{item.hue}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {tab === "overview" ? (
        <div className={p.referenceOverviewGrid}>
          <OverviewCard title="Annotation progress" value={`${labeledPercent}%`} text={`${labeledImages}/${standard.images.length} images have polygons`} progress={labeledPercent} />
          <OverviewCard title="Reference coverage" value={String(totalAnnotations)} text="Total polygons in this reference view" progress={Math.min(100, totalAnnotations * 10)} />
          <OverviewCard title="Classes available" value={String(classRows.length)} text="Classes available for editor and inspection" progress={Math.min(100, classRows.length * 8)} />
        </div>
      ) : null}
    </section>
  );
}

function MiniHealth({ icon: Icon, value, label }: { icon: typeof Image; value: number | string; label: string }) {
  return <div className={p.miniHealth}><Icon /><b>{value}</b><span>{label}</span></div>;
}

function ChecklistItem({ done, title, text }: { done: boolean; title: string; text: string }) {
  return (
    <div className={clsx(p.checklistItem, done && p.checklistItemDone)}>
      {done ? <CheckCircle2 /> : <CircleDashed />}
      <div><strong>{title}</strong><span>{text}</span></div>
    </div>
  );
}

function OverviewCard({ title, value, text, progress }: { title: string; value: string; text: string; progress: number }) {
  return (
    <div className={p.overviewCard}>
      <span>{title}</span>
      <b>{value}</b>
      <p>{text}</p>
      <div className={p.referenceProgress}><span style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
    </div>
  );
}
