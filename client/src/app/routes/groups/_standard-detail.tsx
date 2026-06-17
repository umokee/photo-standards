import { paths } from "@/app/paths";
import ImageWithFallback from "@/components/ui/image-with-fallback/image-with-fallback";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetStandardDetail } from "@/page-components/standards/api/get-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import { StandardDetail } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import { ArrowRight, CheckCircle2, CircleDashed, Image, Images, ListChecks, MousePointer2, Tags, Upload } from "lucide-react";
import { Link, useLoaderData, useNavigate } from "react-router-dom";
import { useGroupDetailOutletContext } from "./_group-detail";
import p from "../platform-pages.module.scss";

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
    <div className={done ? p.readinessDoneV26 : p.readinessOpenV26}>
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
    <div className={p.referenceDetailPageV26}>
      <section className={p.referenceHeaderV26}>
        <div className={p.referenceHeaderPreviewV26}>
          {standard.stats.reference_path ? <img src={`/storage/${standard.stats.reference_path}`} alt="" /> : <Image />}
          <span>{standard.is_active ? "active" : "draft"}</span>
        </div>

        <div className={p.referenceHeaderMainV26}>
          <span className={p.assetEyebrowV26}>Reference / {standard.angle || "view"}</span>
          <h2>{standard.name}</h2>
          <p>Очередь изображений для разметки. Клик по изображению открывает editor.</p>
          <div className={p.referenceMetaLineV26}>
            <span><Images /> {standard.stats.images_count} images</span>
            <span><CheckCircle2 /> {standard.stats.annotated_images_count} labeled</span>
            <span><Tags /> {classes.length} classes used</span>
            <span>Created {formatDate(standard.created_at)}</span>
          </div>
        </div>

        <div className={p.referenceHeaderActionsV26}>
          <UploadImages groupId={group.id} standardId={standard.id} />
          <Link to={paths.inspectionStandard("photo", group.id, standard.id)}><ListChecks /> Use in Inspect</Link>
          <UpdateStandard standard={standard} />
          <DeleteStandard groupId={group.id} id={standard.id} name={standard.name} />
        </div>
      </section>

      <section className={p.referenceDetailGridV26}>
        <main className={p.assetPanelV26}>
          <div className={p.assetPanelHeadV26}>
            <div>
              <span>Images</span>
              <h3>Annotation queue</h3>
            </div>
            <b>{labeled}% labeled</b>
          </div>

          <QueryState
            isEmpty={!standard.images.length}
            size="block"
            emptyTitle="No images uploaded"
            emptyDescription="Загрузи фотографии reference, чтобы начать разметку."
          >
            <div className={p.annotationQueueGridV26}>
              {standard.images.map((image, index) => {
                const done = image.annotation_count > 0;

                return (
                  <button className={done ? p.annotationQueueDoneV26 : p.annotationQueueOpenV26} key={image.id} type="button" onClick={() => openEditor(image.id)}>
                    <div className={p.annotationQueueImageV26}>
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

        <aside className={p.referenceRailV26}>
          <div className={p.assetPanelV26}>
            <div className={p.assetPanelHeadV26}>
              <div>
                <span>Readiness</span>
                <h3>Reference status</h3>
              </div>
              <CheckCircle2 />
            </div>
            <ReadinessRow done={standard.stats.images_count > 0} title="Images uploaded" value={standard.stats.images_count} />
            <ReadinessRow done={classes.length > 0} title="Classes used" value={classes.length} />
            <ReadinessRow done={standard.stats.annotated_images_count > 0} title="Images labeled" value={`${labeled}%`} />
            <div className={p.assetProgressTrackV26}><i style={{ width: `${labeled}%` }} /></div>
          </div>

          <div className={p.assetPanelV26}>
            <div className={p.assetPanelHeadV26}>
              <div>
                <span>Classes</span>
                <h3>Used in this reference</h3>
              </div>
              <Link to={paths.assetClasses(group.id)}>Edit</Link>
            </div>
            <div className={p.classTokenCloudV26}>
              {classes.slice(0, 24).map((item) => <span key={item.id} style={{ ["--hue" as string]: item.hue }}>{item.name}</span>)}
              {!classes.length ? <small>Классы появятся после настройки или разметки.</small> : null}
            </div>
          </div>

          <Link className={p.assetPrimaryLinkV26} to={paths.assetReferences(group.id)}>Back to references <ArrowRight /></Link>
        </aside>
      </section>
    </div>
  );
}
