import { StandardImageDetail } from "@/types/contracts";
import { Expand, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import useImage from "use-image";
import s from "./reference-thumbnail.module.scss";
import clsx from "clsx";

type Props = {
  image: StandardImageDetail;
  focusedSegmentClassId: string | null;
  allowedSegmentClassIds: Set<string>;
};

type ThumbnailSegmentClass = StandardImageDetail["segment_classes"][number];

export const ReferenceThumbnail = ({
  image,
  focusedSegmentClassId,
  allowedSegmentClassIds,
}: Props) => {
  const [expanded, setExpanded] = useState(false);
  const imageUrl = `/storage/${image.image_path}`;

  const segmentClasses = useMemo(() => {
    if (allowedSegmentClassIds.size === 0) return [];
    return image.segment_classes.filter((segmentClass) =>
      allowedSegmentClassIds.has(segmentClass.id)
    );
  }, [allowedSegmentClassIds, image.segment_classes]);

  useEffect(() => {
    if (!expanded) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setExpanded(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [expanded]);

  return (
    <>
      <button
        type="button"
        className={s.thumb}
        onClick={() => setExpanded(true)}
        title="Показать эталон"
      >
        <ThumbnailSvg
          imageUrl={imageUrl}
          segmentClasses={segmentClasses}
          focusedSegmentClassId={focusedSegmentClassId}
        />
        <div className={s.overlay}>
          <Expand size={14} />
          <span>Эталон</span>
        </div>
      </button>

      {expanded &&
        createPortal(
          <div className={s.lightbox} onClick={() => setExpanded(false)}>
            <div className={s.lightboxContent} onClick={(e) => e.stopPropagation()}>
              <div className={s.lightboxStage}>
                <button
                  type="button"
                  className={s.closeBtn}
                  onClick={() => setExpanded(false)}
                  aria-label="Закрыть"
                >
                  <X size={18} />
                </button>
                <ThumbnailSvg
                  imageUrl={imageUrl}
                  segmentClasses={segmentClasses}
                  focusedSegmentClassId={focusedSegmentClassId}
                  large
                />
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
};

function ThumbnailSvg({
  imageUrl,
  segmentClasses,
  focusedSegmentClassId,
  large = false,
}: {
  imageUrl: string;
  segmentClasses: ThumbnailSegmentClass[];
  focusedSegmentClassId: string | null;
  large?: boolean;
}) {
  const [image, status] = useImage(imageUrl);

  if (status !== "loaded" || !image) {
    return (
      <div className={s.svgFallback}>
        <img src={imageUrl} alt="" />
      </div>
    );
  }

  const viewBox = `0 0 ${image.naturalWidth} ${image.naturalHeight}`;
  const strokeWidth = Math.max(image.naturalWidth, image.naturalHeight) / (large ? 400 : 220);

  return (
    <svg
      className={clsx(s.svg, large && s.svgLarge)}
      viewBox={viewBox}
      preserveAspectRatio="xMidYMid meet"
    >
      <image
        href={imageUrl}
        x={0}
        y={0}
        width={image.naturalWidth}
        height={image.naturalHeight}
        preserveAspectRatio="xMidYMid meet"
      />
      {segmentClasses.map((segmentClass) =>
        segmentClass.points.map((polygon, index) => {
          if (polygon.length < 3) return null;

          const isFocused = focusedSegmentClassId === segmentClass.id;
          const isDimmed = focusedSegmentClassId !== null && !isFocused;

          const d =
            polygon
              .map(([x, y], pointIndex) => `${pointIndex === 0 ? "M" : "L"}${x},${y}`)
              .join(" ") + "Z";

          return (
            <path
              key={`${segmentClass.id}-${index}`}
              d={d}
              stroke={`hsl(${segmentClass.hue}, 72%, 48%)`}
              strokeWidth={isFocused ? strokeWidth * 1.6 : strokeWidth}
              fill={`hsla(${segmentClass.hue}, 72%, 48%, ${isFocused ? 0.22 : 0.1})`}
              opacity={isDimmed ? 0.35 : 1}
              strokeLinejoin="round"
            />
          );
        })
      )}
    </svg>
  );
}
