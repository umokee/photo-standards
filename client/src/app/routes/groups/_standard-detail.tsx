
import { paths } from "@/app/paths";
import ImageWithFallback from "@/components/ui/image-with-fallback/image-with-fallback";
import { ReadinessItem } from "@/components/ui/readiness-item/readiness-item";
import QueryState from "@/components/ui/query-state/query-state";
import { useGetStandardDetail } from "@/page-components/standards/api/get-standard";
import { DeleteStandard } from "@/page-components/standards/components/delete-standard";
import { DeleteStandardImage, SetReferenceImage } from "@/page-components/standards/components/standard-image-actions";
import { UpdateStandard } from "@/page-components/standards/components/update-standard";
import { UploadImages } from "@/page-components/standards/components/upload-images";
import type { StandardDetail, StandardImage } from "@/types/contracts";
import clsx from "clsx";
import {
  ArrowRight,
  CheckCircle2,
  Database,
  Image,
  Images,
  ListChecks,
  MousePointer2,
  ShieldCheck,
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
  if (!imageFeaturesReady(image)) return "нет features";
  return `${image.features_keypoint_count ?? 0} keypoints`;
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
  const poolStatusText = !referenceImages.length
    ? "Выбери минимум одно фото для проверки."
    : poolReady
      ? "Reference pool готов к проверке."
      : "Дождись features для всех фото в проверке.";

  const openEditor = (imageId: string) => navigate(paths.standardImage(group.id, standard.id, imageId));

  return (
    <div className={s.detailPage}>
      <section className={s.referenceHeader}>
        <div className={s.previewMedia}>
          {standard.stats.reference_path ? <img src={`/storage/${standard.stats.reference_path}`} alt="" /> : <Image />}
          <span className={referenceImages.length ? s.activePill : s.openPill}>
            {referenceImages.length ? `${referenceImages.length} в проверке` : "нет pool"}
          </span>
        </div>

        <div className={s.headerMain}>
          <span className={s.eyebrow}>Эталон проверки</span>
          <h2>{standard.name}</h2>
          <p>Выбери фото для проверки — система возьмёт лучший ракурс для переноса полигонов.</p>
          <div className={clsx(s.detailStatusStripV103, poolReady ? s.detailStatusReadyV103 : s.detailStatusOpenV103)} role="status">
            <ShieldCheck />
            <span>{poolStatusText}</span>
          </div>
          <div className={s.metaLine}>
            <span><Images /> {standard.stats.images_count} фото</span>
            <span><ShieldCheck /> {referenceImages.length} в проверке</span>
            <span><Database /> {allFeaturesReadyCount}/{standard.stats.images_count} features</span>
            <span><CheckCircle2 /> {standard.stats.annotated_images_count} размечено</span>
          </div>
        </div>

        <div className={clsx(s.headerActions, s.detailHeaderActionsV103)}>
          <Link
            aria-label={`Проверить эталон ${standard.name}`}
            className={s.detailPrimaryAction}
            to={paths.inspectionStandard("photo", group.id, standard.id)}
          >
            <ListChecks />
            Проверить
          </Link>
          <div className={s.detailUtilityActionsV103}>
            <UploadImages groupId={group.id} standardId={standard.id} triggerClassName={s.detailSecondaryAction} />
            <UpdateStandard standard={standard} triggerClassName={s.detailSecondaryAction} />
            <DeleteStandard groupId={group.id} id={standard.id} name={standard.name} triggerClassName={s.detailDangerAction} />
          </div>
        </div>
      </section>

      <section className={s.detailGrid}>
        <main className={s.panel}>
          <div className={s.panelHead}>
            <div><h3>Фото эталона</h3></div>
            <b className={poolReady ? s.donePill : s.openPill}>
              {referenceImages.length ? `${poolFeaturesReadyCount}/${referenceImages.length} features` : "нет pool"}
            </b>
          </div>

          <QueryState isEmpty={!standard.images.length} size="block" emptyTitle="Нет фото" emptyDescription="Загрузи фото, разметь объекты и выбери кадры для проверки.">
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
                    <button
                      aria-label={`Открыть фото ${index + 1} в редакторе`}
                      className={s.annotationCardMain}
                      type="button"
                      onClick={() => openEditor(image.id)}
                    >
                      <div className={s.annotationMedia}>
                        <ImageWithFallback src={`/storage/${image.image_path}`} />
                        <span className={s.annotationBadgeRowV103}>
                          <span className={s.annotationIndexBadgeV103}>{String(index + 1).padStart(2, "0")}</span>
                          <span
                            className={clsx(
                              s.annotationStateBadgeV103,
                              image.is_reference
                                ? s.annotationStateReferenceV103
                                : done
                                  ? s.annotationStateDoneV103
                                  : s.annotationStateOpenV103
                            )}
                          >
                            {image.is_reference ? "В проверке" : done ? "Размечено" : "Без разметки"}
                          </span>
                        </span>
                        <span className={clsx(s.annotationFeatureBadgeV103, featuresReady ? s.annotationFeatureReadyV103 : s.annotationFeatureMissingV103)}>
                          {featuresReady ? featureStatusLabel(image) : "нет features"}
                        </span>
                      </div>
                      <footer>
                        <strong>{image.is_reference ? "В проверке" : `Фото ${index + 1}`}</strong>
                        <small>{image.annotation_count} полигонов · {featureStatusLabel(image)}</small>
                      </footer>
                    </button>

                    <div className={s.annotationCardActions}>
                      <SetReferenceImage
                        groupId={group.id}
                        standardId={standard.id}
                        imageId={image.id}
                        isReference={image.is_reference}
                        showLabel
                        buttonClassName={clsx(image.is_reference ? s.removePoolAction : s.poolAction, s.poolActionPrimaryV103)}
                      />
                      <div className={s.annotationUtilityGroupV103}>
                        <button
                          aria-label={`Открыть фото ${index + 1} в редакторе`}
                          className={s.imageEditorAction}
                          type="button"
                          onClick={() => openEditor(image.id)}
                        >
                          <MousePointer2 />
                          <span>Редактор</span>
                        </button>
                        <DeleteStandardImage
                          groupId={group.id}
                          standardId={standard.id}
                          imageId={image.id}
                          isReference={image.is_reference}
                          buttonClassName={s.deleteImageAction}
                        />
                      </div>
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
                <div><h3>В проверке</h3></div>
                <b className={referenceImages.length ? s.donePill : s.openPill}>{referenceImages.length}</b>
              </div>
              <div className={s.poolSummary}>
                <div>
                  <span>features</span>
                  <strong>{poolFeaturesReadyCount}/{referenceImages.length || 0}</strong>
                </div>
              </div>
              <div className={s.poolRailList}>
                {referenceImages.length ? (
                  referenceImages.map((image, index) => (
                    <button key={image.id} type="button" className={s.poolRailItem} onClick={() => openEditor(image.id)}>
                      <img src={`/storage/${image.image_path}`} alt="" />
                      <span>
                        <strong>Фото {index + 1}</strong>
                        <small>{image.annotation_count} полигонов · {featureStatusLabel(image)}</small>
                      </span>
                    </button>
                  ))
                ) : (
                  <div className={s.poolEmpty}>
                    <ShieldCheck />
                    <strong>Нет фото в проверке</strong>
                    <span>Выбери фото через кнопку “Использовать в проверке”.</span>
                  </div>
                )}
              </div>
            </section>

            <section className={s.panel}>
              <div className={s.panelHead}>
                <div><h3>Готовность</h3></div>
                <CheckCircle2 />
              </div>
              <div className={s.readinessRows}>
                <ReadinessItem className={`${s.readinessRow} ${standard.stats.images_count > 0 ? s.readinessRowDone : ""}`} done={standard.stats.images_count > 0} title="Фото" value={standard.stats.images_count} variant="valueRow" />
                <ReadinessItem className={`${s.readinessRow} ${referenceImages.length > 0 ? s.readinessRowDone : ""}`} done={referenceImages.length > 0} title="В проверке" value={`${referenceImages.length}/${standard.stats.images_count}`} variant="valueRow" />
                <ReadinessItem className={`${s.readinessRow} ${poolReady ? s.readinessRowDone : ""}`} done={poolReady} title="Features" value={`${poolFeaturesReadyCount}/${referenceImages.length || 0}`} variant="valueRow" />
                <ReadinessItem className={`${s.readinessRow} ${classes.length > 0 ? s.readinessRowDone : ""}`} done={classes.length > 0} title="Классы" value={classes.length} variant="valueRow" />
                <ReadinessItem className={`${s.readinessRow} ${standard.stats.annotated_images_count > 0 ? s.readinessRowDone : ""}`} done={standard.stats.annotated_images_count > 0} title="Разметка" value={`${labeled}%`} variant="valueRow" />
                <div className={s.progressTrack} role="progressbar" aria-label="Размечено фото" aria-valuemin={0} aria-valuemax={100} aria-valuenow={labeled}><i style={{ width: `${labeled}%` }} /></div>
              </div>
            </section>

            <section className={s.panel}>
              <div className={s.panelHead}>
                <div><h3>Классы</h3></div>
                <Link to={paths.assetClasses(group.id)}>Настроить</Link>
              </div>
              <div className={s.tokenCloud}>
                {classes.slice(0, 28).map((item) => <span key={item.id} style={{ ["--hue" as string]: item.hue }}>{item.name}</span>)}
                {!classes.length ? <small>Появятся после настройки или разметки.</small> : null}
              </div>
            </section>
          </div>

          <Link className={s.secondaryAction} to={paths.assetReferences(group.id)}>К списку <ArrowRight /></Link>
        </aside>
      </section>
    </div>
  );
}
