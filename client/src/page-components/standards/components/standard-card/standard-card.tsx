import { Badge } from "@/components/ui/badge/badge";
import ImageWithFallback from "@/components/ui/image-with-fallback/image-with-fallback";
import { QueryBoundary } from "@/components/ui/query-boundary/query-boundary";
import QueryState from "@/components/ui/query-state/query-state";
import SurfaceSection from "@/components/ui/surface-section/surface-section";
import { useMediaQuery } from "@/hooks/use-media-query";
import { GroupStandard } from "@/types/contracts";
import { angleLabel } from "@/utils/labels";
import clsx from "clsx";
import { ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useGetStandardDetail } from "../../api/get-standard";
import { DeleteStandard } from "../delete-standard";
import { DeleteStandardImage, SetReferenceImage } from "../standard-image-actions";
import { UpdateStandard } from "../update-standard";
import { UploadImages } from "../upload-images";
import s from "./standard-card.module.scss";

type Props = {
  standard: GroupStandard;
  expanded: boolean;
  onToggle: () => void;
  onToImageEditor: (imageId: string) => void;
};

export const StandardCard = ({ standard, expanded, onToggle, onToImageEditor }: Props) => {
  const src = standard.reference_path ? `/storage/${standard.reference_path}` : undefined;
  const angleText = standard.angle ? ` (${angleLabel(standard.angle)})` : "";

  const rootRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!expanded) return;

    requestAnimationFrame(() => {
      rootRef.current?.scrollIntoView({
        behavior: "instant",
        block: "start",
      });
    });
  }, [expanded]);

  return (
    <article ref={rootRef} className={clsx(s.root, expanded && s.expanded)}>
      <div className={s.header} onClick={onToggle}>
        <div className={s.reference}>
          <ImageWithFallback src={src} iconSize={20} />
        </div>
        <div className={s.info}>
          <div className={s.name}>
            {standard.name}
            {angleText}
          </div>
          <div className={s.meta}>
            {standard.images_count} изображений &middot; {standard.annotated_images_count} размечено
          </div>
        </div>
        <div className={s.actions} onClick={(e) => e.stopPropagation()}>
          <UploadImages groupId={standard.group_id} standardId={standard.id} />
          <UpdateStandard standard={standard} />
          <DeleteStandard groupId={standard.group_id} id={standard.id} name={standard.name} />
        </div>
        <ChevronRight className={s.chevron} size={14} />
      </div>
      {expanded && (
        <div className={s.body}>
          <QueryBoundary
            loadingText="Загрузка подробностей..."
            errorTitle="Не удалось загрузить подробности"
          >
            <StandardCardDetail standardId={standard.id} onToImageEditor={onToImageEditor} />
          </QueryBoundary>
        </div>
      )}
    </article>
  );
};

const IMAGE_PREVIEW_LIMIT = 10;

const StandardCardDetail = ({
  standardId,
  onToImageEditor,
}: {
  standardId: string;
  onToImageEditor: (imageId: string) => void;
}) => {
  const { data: standard } = useGetStandardDetail(standardId);
  const isTouchPreview = useMediaQuery("(hover: none), (pointer: coarse)");

  const [showAllImages, setShowAllImages] = useState(false);
  const [armedImageId, setArmedImageId] = useState<string | null>(null);

  useEffect(() => {
    setArmedImageId(null);
    setShowAllImages(false);
  }, [standardId]);

  useEffect(() => {
    setArmedImageId(null);
  }, [showAllImages]);

  const displayCategories =
    standard.used_segment_class_categories ?? standard.segment_class_categories;
  const displayUngroupedClasses =
    standard.used_ungrouped_segment_classes ?? standard.ungrouped_segment_classes;

  const allClasses = [
    ...displayCategories.flatMap((category) => category.segment_classes),
    ...displayUngroupedClasses,
  ];
  const visibleImages = showAllImages
    ? standard.images
    : standard.images.slice(0, IMAGE_PREVIEW_LIMIT);
  const hiddenImagesCount = Math.max(standard.images.length - IMAGE_PREVIEW_LIMIT, 0);

  const handleImageCardClick = (imageId: string) => {
    if (!isTouchPreview) {
      onToImageEditor(imageId);
      return;
    }

    if (armedImageId !== imageId) {
      setArmedImageId(imageId);
      return;
    }

    setArmedImageId(null);
    onToImageEditor(imageId);
  };

  return (
    <>
      <QueryState isEmpty={!standard.images.length} emptyTitle="Нет фотографий">
        <div className={s.bodyImages}>
          {visibleImages.map((image) => {
            const isAnnotated = image.annotation_count > 0;
            const isArmed = armedImageId === image.id;

            return (
              <div
                key={image.id}
                className={clsx(
                  s.imageCard,
                  isAnnotated && s.imageCardAnnotated,
                  image.is_reference && s.imageCardReference,
                  isArmed && s.imageCardArmed
                )}
                onClick={() => handleImageCardClick(image.id)}
              >
                <div className={s.imageCardPhoto}>
                  <ImageWithFallback src={`/storage/${image.image_path}`} iconSize={20} />
                </div>

                {image.is_reference && <span className={s.imageCardRef}>ПРОВ</span>}

                <div className={s.imageCardOverlay}>
                  <SetReferenceImage
                    groupId={standard.group_id}
                    standardId={standard.id}
                    imageId={image.id}
                    isReference={image.is_reference}
                  />
                  <DeleteStandardImage
                    groupId={standard.group_id}
                    standardId={standard.id}
                    imageId={image.id}
                    isReference={image.is_reference}
                  />
                </div>

                <span
                  className={clsx(
                    s.imageCardDot,
                    isAnnotated ? s.imageCardDotDone : s.imageCardDotTodo
                  )}
                />
              </div>
            );
          })}

          {hiddenImagesCount > 0 && !showAllImages && (
            <button
              type="button"
              className={clsx(s.imageCard, s.imageCardMore)}
              onClick={() => setShowAllImages(true)}
            >
              <span className={s.imageCardMoreCount}>+{hiddenImagesCount}</span>
              <span className={s.imageCardMoreLabel}>Открыть</span>
            </button>
          )}

          {showAllImages && standard.images.length > IMAGE_PREVIEW_LIMIT && (
            <button
              type="button"
              className={clsx(s.imageCard, s.imageCardMore)}
              onClick={() => setShowAllImages(false)}
            >
              <span className={s.imageCardMoreLabel}>Скрыть</span>
            </button>
          )}
        </div>
      </QueryState>

      {allClasses.length > 0 && (
        <SurfaceSection title={"Классы"} transparent className={s.classesSection}>
          <div className={s.classes}>
            {allClasses.map((segmentClass) => (
              <Badge key={segmentClass.id} colorDot={segmentClass.hue}>
                {segmentClass.name}
              </Badge>
            ))}
          </div>
        </SurfaceSection>
      )}
    </>
  );
};
