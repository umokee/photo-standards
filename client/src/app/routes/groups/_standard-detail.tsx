import { paths } from "@/app/paths";
import ImageWithFallback from "@/components/ui/image-with-fallback/image-with-fallback";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetStandardDetail } from "@/page-components/standards/api/get-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import { StandardDetail } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { ArrowRight, CheckCircle2, CircleDashed, Image, Images, ListChecks, MousePointer2, Tags } from "lucide-react";
import { Link, useLoaderData, useNavigate } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import s from "./_project-assets-strict.module.scss";

function percent(part: number, total: number) {
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((part / total) * 100)));
}

function collectClasses(standard: StandardDetail) {
  const categories = standard.used_segment_class_categories ?? standard.segment_class_categories;
  const ungrouped = standard.used_ungrouped_segment_classes ?? standard.ungrouped_segment_classes;
  return [...categories.flatMap((category) => category.segment_classes), ...ungrouped];
}

function ReadinessRow({ done, title, value }: { done: boolean; title: string; value: string | number }) {
  return (
    <div className={`${s.readinessRow} ${done ? s.readinessRowDone : ""}`}>
      {done ? <CheckCircle2 /> : <CircleDashed />}
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function Component() {
  const navigate = useNavigate();
  const { standardId } = useLoaderData() as { standardId: string };
  const { group } = useGroupDetailOutletContext();
  const { data: standard } = useGetStandardDetail(standardId);
  const classes = collectClasses(standard);
  const labeled = percent(standard.stats.annotated_images_count, standard.stats.images_count);

  const openEditor = (imageId: string) => navigate(paths.standardImage(group.id, standard.id, imageId));

  return (
    <div className={s.detailPage}>
      <section className={s.referenceHeader}>
        <div className={s.previewMedia}>
          {standard.stats.reference_path ? <img src={`/storage/${standard.stats.reference_path}`} alt="" /> : <Image />}
          <span className={standard.is_active ? s.activePill : s.draftPill}>{standard.is_active ? "active" : "draft"}</span>
        </div>

        <div className={s.headerMain}>
          <span className={s.eyebrow}>Reference / {standard.angle || "view"}</span>
          <h2>{standard.name}</h2>
          <p>Очередь изображений для разметки. Клик по изображению открывает editor.</p>
          <div className={s.metaLine}>
            <span><Images /> {standard.stats.images_count} images</span>
            <span><CheckCircle2 /> {standard.stats.annotated_images_count} labeled</span>
            <span><Tags /> {classes.length} classes used</span>
            <span>Created {formatDate(standard.created_at)}</span>
          </div>
        </div>

        <div className={s.headerActions}>
          <UploadImages groupId={group.id} standardId={standard.id} />
          <Link to={paths.inspectionStandard("photo", group.id, standard.id)}><ListChecks /> Inspect</Link>
          <UpdateStandard standard={standard} />
          <DeleteStandard groupId={group.id} id={standard.id} name={standard.name} />
        </div>
      </section>

      <section className={s.detailGrid}>
        <main className={s.panel}>
          <div className={s.panelHead}>
            <div><span>Images</span><h3>Annotation queue</h3></div>
            <b className={labeled === 100 ? s.donePill : s.openPill}>{labeled}% labeled</b>
          </div>

          <QueryState isEmpty={!standard.images.length} size="block" emptyTitle="No images uploaded" emptyDescription="Загрузи фотографии reference, чтобы начать разметку.">
            <div className={s.annotationGrid}>
              {standard.images.map((image, index) => {
                const done = image.annotation_count > 0;

                return (
                  <button className={`${s.annotationCard} ${done ? s.annotationCardDone : s.annotationCardOpen}`} key={image.id} type="button" onClick={() => openEditor(image.id)}>
                    <div className={s.annotationMedia}>
                      <ImageWithFallback src={`/storage/${image.image_path}`} />
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <b>{done ? "labeled" : "empty"}</b>
                    </div>
                    <footer>
                      <strong>{image.is_reference ? "Reference image" : `Image ${index + 1}`}</strong>
                      <small>{image.annotation_count} polygons · {formatDate(image.created_at)}</small>
                    </footer>
                    <i><MousePointer2 /> Open editor</i>
                  </button>
                );
              })}
            </div>
          </QueryState>
        </main>

        <aside className={s.rail}>
          <div className={s.railScroll}>
            <section className={s.panel}>
              <div className={s.panelHead}>
                <div><span>Readiness</span><h3>Reference status</h3></div>
                <CheckCircle2 />
              </div>
              <div className={s.readinessRows}>
                <ReadinessRow done={standard.stats.images_count > 0} title="Images uploaded" value={standard.stats.images_count} />
                <ReadinessRow done={classes.length > 0} title="Classes used" value={classes.length} />
                <ReadinessRow done={standard.stats.annotated_images_count > 0} title="Images labeled" value={`${labeled}%`} />
                <div className={s.progressTrack}><i style={{ width: `${labeled}%` }} /></div>
              </div>
            </section>

            <section className={s.panel}>
              <div className={s.panelHead}>
                <div><span>Classes</span><h3>Used in this reference</h3></div>
                <Link to={paths.assetClasses(group.id)}>Edit</Link>
              </div>
              <div className={s.tokenCloud}>
                {classes.slice(0, 28).map((item) => <span key={item.id} style={{ ["--hue" as string]: item.hue }}>{item.name}</span>)}
                {!classes.length ? <small>Классы появятся после настройки или разметки.</small> : null}
              </div>
            </section>
          </div>

          <Link className={s.secondaryAction} to={paths.assetReferences(group.id)}>Back to references <ArrowRight /></Link>
        </aside>
      </section>
    </div>
  );
}
