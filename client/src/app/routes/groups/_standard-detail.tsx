
import { paths } from "@/app/paths";
import ImageWithFallback from "@/components/ui/image-with-fallback/image-with-fallback";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetStandardDetail } from "@/page-components/standards/api/get-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { DeleteStandardImage, SetReferenceImage } from "@/page-components/standards/components/standard-image-actions";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import type { StandardDetail, StandardImage } from "@/types/contracts";
import { formatDate } from "@/utils/formatDate";
import clsx from "clsx";
import {
  ArrowRight,
  CheckCircle2,
  CircleDashed,
  Database,
  Image,
  Images,
  ListChecks,
  MousePointer2,
  ShieldCheck,
  Tags,
} from "lucide-react";
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

function imageFeaturesReady(image: StandardImage) {
  return image.features_keypoint_count != null && Boolean(image.features_computed_at);
}

function featureStatusLabel(image: StandardImage) {
  if (!imageFeaturesReady(image)) return "features missing";
  return `${image.features_keypoint_count ?? 0} keypoints`;
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
  const referenceImages = standard.images.filter((image) => image.is_reference);
  const allFeaturesReadyCount = standard.images.filter(imageFeaturesReady).length;
  const poolFeaturesReadyCount = referenceImages.filter(imageFeaturesReady).length;
  const poolReady = referenceImages.length > 0 && poolFeaturesReadyCount === referenceImages.length;

  const openEditor = (imageId: string) => navigate(paths.standardImage(group.id, standard.id, imageId));

  return (
    <div className={s.detailPage}>
      <section className={s.referenceHeader}>
        <div className={s.previewMedia}>
          {standard.stats.reference_path ? <img src={`/storage/${standard.stats.reference_path}`} alt="" /> : <Image />}
          <span className={referenceImages.length ? s.activePill : s.openPill}>
            {referenceImages.length ? `${referenceImages.length} в проверке` : "pool empty"}
          </span>
        </div>

        <div className={s.headerMain}>
          <span className={s.eyebrow}>Reference pool / {standard.angle || "view"}</span>
          <h2>{standard.name}</h2>
          <p>
            Здесь назначаются фото, которые участвуют в проверке. Эти кадры образуют reference pool:
            при проверке система выбирает лучший ракурс и переносит с него полигоны.
          </p>
          <div className={s.metaLine}>
            <span><Images /> {standard.stats.images_count} images</span>
            <span><ShieldCheck /> {referenceImages.length} in check</span>
            <span><Database /> {allFeaturesReadyCount}/{standard.stats.images_count} features ready</span>
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
            <div><span>Images</span><h3>Reference pool manager</h3></div>
            <b className={poolReady ? s.donePill : s.openPill}>
              {referenceImages.length ? `${poolFeaturesReadyCount}/${referenceImages.length} ready` : "no pool"}
            </b>
          </div>

          <div className={s.referencePoolNotice}>
            <ShieldCheck />
            <div>
              <strong>Фото “В проверке” используются как кандидаты multi-reference.</strong>
              <span>
                Нажми “Использовать в проверке” на карточке фото. Если features отсутствуют,
                сервер рассчитает их через set_reference перед включением в pool.
              </span>
            </div>
          </div>

          <QueryState isEmpty={!standard.images.length} size="block" emptyTitle="No images uploaded" emptyDescription="Загрузи фотографии reference, чтобы начать разметку и собрать pool для проверки.">
            <div className={s.annotationGrid}>
              {standard.images.map((image, index) => {
                const done = image.annotation_count > 0;
                const featuresReady = imageFeaturesReady(image);

                return (
                  <article
                    className={clsx(
                      s.annotationCard,
                      done ? s.annotationCardDone : s.annotationCardOpen,
                      image.is_reference && s.annotationCardReference
                    )}
                    key={image.id}
                  >
                    <button className={s.annotationCardMain} type="button" onClick={() => openEditor(image.id)}>
                      <div className={s.annotationMedia}>
                        <ImageWithFallback src={`/storage/${image.image_path}`} />
                        <span>{String(index + 1).padStart(2, "0")}</span>
                        <b className={image.is_reference ? s.poolMediaBadge : undefined}>
                          {image.is_reference ? "В проверке" : done ? "labeled" : "empty"}
                        </b>
                        <em className={featuresReady ? s.featureReadyPill : s.featureMissingPill}>
                          {featuresReady ? "features ready" : "features missing"}
                        </em>
                      </div>
                      <footer>
                        <strong>{image.is_reference ? "Reference pool photo" : `Image ${index + 1}`}</strong>
                        <small>{image.annotation_count} polygons · {featureStatusLabel(image)}</small>
                      </footer>
                      <i><MousePointer2 /> Open editor</i>
                    </button>

                    <div className={s.annotationCardActions}>
                      <SetReferenceImage
                        groupId={group.id}
                        standardId={standard.id}
                        imageId={image.id}
                        isReference={image.is_reference}
                        showLabel
                        buttonClassName={image.is_reference ? s.removePoolAction : s.poolAction}
                      />
                      <button className={s.imageEditorAction} type="button" onClick={() => openEditor(image.id)}>
                        <MousePointer2 />
                        <span>Editor</span>
                      </button>
                      <DeleteStandardImage
                        groupId={group.id}
                        standardId={standard.id}
                        imageId={image.id}
                        isReference={image.is_reference}
                        buttonClassName={s.deleteImageAction}
                      />
                    </div>
                  </article>
                );
              })}
            </div>
          </QueryState>
        </main>

        <aside className={s.rail}>
          <div className={s.railScroll}>
            <section className={s.panel}>
              <div className={s.panelHead}>
                <div><span>Inspection pool</span><h3>Фото в проверке</h3></div>
                <b className={referenceImages.length ? s.donePill : s.openPill}>{referenceImages.length}</b>
              </div>
              <div className={s.poolSummary}>
                <p>
                  Эти фото используются backend-проверкой как reference-кандидаты. Чем лучше покрыты ракурсы,
                  тем выше шанс выбрать правильный вид для переноса полигонов.
                </p>
                <div>
                  <span>features ready</span>
                  <strong>{poolFeaturesReadyCount}/{referenceImages.length || 0}</strong>
                </div>
              </div>
              <div className={s.poolRailList}>
                {referenceImages.length ? (
                  referenceImages.map((image, index) => (
                    <button key={image.id} type="button" className={s.poolRailItem} onClick={() => openEditor(image.id)}>
                      <img src={`/storage/${image.image_path}`} alt="" />
                      <span>
                        <strong>Pool image {index + 1}</strong>
                        <small>{image.annotation_count} polygons · {featureStatusLabel(image)}</small>
                      </span>
                    </button>
                  ))
                ) : (
                  <div className={s.poolEmpty}>
                    <ShieldCheck />
                    <strong>Reference pool пустой</strong>
                    <span>Выбери хотя бы одно фото через кнопку “Использовать в проверке”.</span>
                  </div>
                )}
              </div>
            </section>

            <section className={s.panel}>
              <div className={s.panelHead}>
                <div><span>Readiness</span><h3>Reference status</h3></div>
                <CheckCircle2 />
              </div>
              <div className={s.readinessRows}>
                <ReadinessRow done={standard.stats.images_count > 0} title="Images uploaded" value={standard.stats.images_count} />
                <ReadinessRow done={referenceImages.length > 0} title="Reference pool" value={`${referenceImages.length}/${standard.stats.images_count}`} />
                <ReadinessRow done={poolReady} title="Pool features" value={`${poolFeaturesReadyCount}/${referenceImages.length || 0}`} />
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
