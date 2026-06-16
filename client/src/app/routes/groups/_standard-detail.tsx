import { paths } from "@/app/paths";
import { QueryBoundary } from "@/components/ui/query-boundary/query-boundary";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetStandardDetail } from "@/page-components/standards/api/get-standard";
import { formatDate } from "@/utils/formatDate";
import clsx from "clsx";
import { CheckCircle2, CircleDashed, Image, PenLine, Star, Upload } from "lucide-react";
import { useLoaderData, useNavigate } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "../platform-pages.module.scss";

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
      <div className={p.referenceWorkspace}>
        <section className={p.panelCard}>
          <div className={p.cardTitleRow}>
            <div>
              <h3>Reference views</h3>
              <p>
                {group.stats.standards_count} standards · {group.stats.images_count} images · {group.stats.polygons_count} polygons
              </p>
            </div>
            <span className={p.softPill}>Annotate</span>
          </div>

          <div className={p.referenceList}>
            {group.standards.map((standard) => {
              const isActive = selectedStandard?.id === standard.id;
              const progress = standard.images_count
                ? Math.round((standard.annotated_images_count / standard.images_count) * 100)
                : 0;

              return (
                <button
                  type="button"
                  className={clsx(p.referenceCard, isActive && p.referenceCardActive)}
                  key={standard.id}
                  onClick={() => handleOpenStandard(standard.id)}
                >
                  <div className={p.referencePreview}>
                    {standard.reference_path ? (
                      <img src={`/storage/${standard.reference_path}`} alt="" />
                    ) : (
                      <Image />
                    )}
                    {standard.stats?.reference_image_id ? (
                      <span className={p.referenceBadge}><Star /> Reference</span>
                    ) : null}
                  </div>
                  <div className={p.referenceBody}>
                    <strong>{standard.name}</strong>
                    <small>{standard.angle || "without angle"}</small>
                    <div className={p.referenceProgress}>
                      <span style={{ width: `${progress}%` }} />
                    </div>
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
          <h3>Dataset summary</h3>
          <p>Покрытие эталонов, изображений и классов для проверки изделия.</p>
          <div className={p.statStrip}>
            <div className={p.statBox}><strong>{group.stats.standards_count}</strong><span>Standards</span></div>
            <div className={p.statBox}><strong>{group.stats.images_count}</strong><span>Images</span></div>
            <div className={p.statBox}><strong>{group.stats.segment_classes_count}</strong><span>Classes</span></div>
            <div className={p.statBox}><strong>{group.stats.polygons_count}</strong><span>Polygons</span></div>
          </div>
          <div className={p.quickHint}>
            <PenLine />
            <span>Выбери эталон слева и открой любое изображение ниже, чтобы перейти в разметку.</span>
          </div>
        </aside>
      </div>

      {selectedStandard ? (
        <QueryBoundary
          loadingText="Загрузка изображений эталона..."
          errorTitle="Не удалось загрузить изображения"
        >
          <StandardImagesPanel groupId={group.id} standardId={selectedStandard.id} />
        </QueryBoundary>
      ) : null}
    </QueryState>
  );
}

function StandardImagesPanel({ groupId, standardId }: { groupId: string; standardId: string }) {
  const navigate = useNavigate();
  const { data: standard } = useGetStandardDetail(standardId);
  const totalAnnotations = standard.images.reduce((sum, image) => sum + image.annotation_count, 0);

  return (
    <section className={p.panelCard}>
      <div className={p.cardTitleRow}>
        <div>
          <h3>{standard.name}</h3>
          <p>
            {standard.images.length} images · {totalAnnotations} polygons · updated {formatDate(standard.created_at)}
          </p>
        </div>
        <span className={p.softPill}>Click image to annotate</span>
      </div>

      {standard.images.length ? (
        <div className={p.imageGrid}>
          {standard.images.map((image, index) => {
            const isAnnotated = image.annotation_count > 0;
            return (
              <button
                type="button"
                className={p.imageCard}
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
      )}
    </section>
  );
}
