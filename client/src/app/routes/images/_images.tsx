import { paths } from "@/app/paths";
import { QueryBoundary } from "@/components/ui/query-boundary/query-boundary";
import { useGetGroup } from "@/page-components/groups/api/get-group";
import {
  AnnotateSegmentClassInput,
  annotateSegmentClassMutationKey,
  useAnnotateSegmentClass,
} from "@/page-components/segments/api/annotate-segment";
import CanvasSurface, {
  EditorMode,
} from "@/page-components/segments/components/canvas-surface/canvas-surface";
import { SegmentHeader } from "@/page-components/segments/components/segment-header/segment-header";
import {
  DrawKind,
  SegmentPanel,
} from "@/page-components/segments/components/segment-panel/segment-panel";
import { useGetImage } from "@/page-components/standards/api/get-image";
import { useGetStandardDetail } from "@/page-components/standards/api/get-standard";
import { useMutationState } from "@tanstack/react-query";
import clsx from "clsx";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Circle,
  Image as ImageIcon,
  Layers3,
  MousePointer2,
  PenLine,
  SkipForward,
  Sparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useLoaderData, useNavigate } from "react-router-dom";
import s from "./_images.module.scss";

type LoaderData = {
  groupId: string;
  standardId: string;
  imageId: string;
};

type PendingAnnotation = {
  submittedAt: number;
  variables: AnnotateSegmentClassInput;
};

const drawKindToMode: Record<DrawKind, EditorMode> = {
  polygon: "draw-polygon",
  sam: "draw-sam",
};

export function Component() {
  return (
    <QueryBoundary
      size="page"
      loadingText="Загрузка изображения..."
      errorTitle="Не удалось открыть изображение"
      errorDescription="Проверьте подключение или попробуйте перезагрузить страницу"
    >
      <ImagesContent />
    </QueryBoundary>
  );
}

const ImagesContent = () => {
  const navigate = useNavigate();
  const { groupId, standardId, imageId } = useLoaderData() as LoaderData;

  const [selectedSegmentClassId, setSelectedSegmentClassId] = useState<string | null>(null);
  const [selectedContourIndex, setSelectedContourIndex] = useState<number | null>(null);
  const [mode, setMode] = useState<EditorMode>("view");
  const [annotationError, setAnnotationError] = useState<string | null>(null);

  const { data: group } = useGetGroup(groupId);
  const { data: standard } = useGetStandardDetail(standardId);
  const { data: image } = useGetImage(imageId);

  const annotate = useAnnotateSegmentClass({ groupId, standardId });
  const pendingAnnotations = useMutationState<PendingAnnotation>({
    filters: {
      mutationKey: annotateSegmentClassMutationKey,
      status: "pending",
    },
    select: (mutation) => ({
      submittedAt: mutation.state.submittedAt,
      variables: mutation.state.variables as AnnotateSegmentClassInput,
    }),
  });

  const categories = standard.segment_class_categories;
  const ungroupedClasses = standard.ungrouped_segment_classes;
  const allSegmentClasses = useMemo(
    () => [
      ...categories.flatMap((category) => category.segment_classes),
      ...ungroupedClasses,
    ],
    [categories, ungroupedClasses]
  );
  const availableSegmentClassIds = useMemo(
    () => new Set(allSegmentClasses.map((item) => item.id)),
    [allSegmentClasses]
  );
  const pendingByClassId: Record<string, number[][][]> = Object.fromEntries(
    pendingAnnotations
      .filter((item) => item.variables.imageId === imageId)
      .sort((a, b) => a.submittedAt - b.submittedAt)
      .map((item) => [item.variables.segmentClassId, item.variables.points])
  );

  const imageSegmentClasses = image.segment_classes.map((segmentClass) => ({
    ...segmentClass,
    points: pendingByClassId[segmentClass.id] ?? segmentClass.points,
  }));
  const selectedImageSegmentClass =
    imageSegmentClasses.find((item) => item.id === selectedSegmentClassId) ?? null;
  const selectedSegmentClass =
    allSegmentClasses.find((item) => item.id === selectedSegmentClassId) ?? null;

  const imageIds = standard.images.map((img) => img.id);
  const currentIndex = imageIds.indexOf(imageId);
  const safeIndex = currentIndex >= 0 ? currentIndex : 0;

  const prevImage = currentIndex > 0 ? standard.images[currentIndex - 1] : null;
  const nextImage =
    currentIndex >= 0 && currentIndex < standard.images.length - 1
      ? standard.images[currentIndex + 1]
      : null;
  const nextUnannotatedImage =
    currentIndex >= 0
      ? (standard.images
          .slice(currentIndex + 1)
          .concat(standard.images.slice(0, currentIndex))
          .find((img) => img.annotation_count === 0) ?? null)
      : null;

  const imageUrl = `/storage/${image.image_path}`;
  const isCurrentAnnotated = imageSegmentClasses.some((segmentClass) => segmentClass.points.length);
  const annotatedCount = standard.images.filter((item) => item.annotation_count > 0).length;
  const polygonsCount = imageSegmentClasses.reduce((sum, item) => sum + item.points.length, 0);
  const annotationProgress = standard.images.length
    ? Math.round((annotatedCount / standard.images.length) * 100)
    : 0;
  const currentProgress = standard.images.length
    ? Math.round(((safeIndex + 1) / standard.images.length) * 100)
    : 0;
  const canDraw = !!selectedSegmentClassId && availableSegmentClassIds.has(selectedSegmentClassId);

  useEffect(() => {
    if (!selectedSegmentClassId) {
      return;
    }

    if (availableSegmentClassIds.has(selectedSegmentClassId)) {
      return;
    }

    setSelectedSegmentClassId(null);
    setSelectedContourIndex(null);
    setMode("view");
  }, [availableSegmentClassIds, selectedSegmentClassId]);

  const saveSegmentContours = (segmentClassId: string, nextContours: number[][][]) => {
    setAnnotationError(null);
    annotate.mutate(
      {
        segmentClassId,
        imageId,
        points: nextContours,
      },
      {
        onError: (error) => {
          setAnnotationError(error instanceof Error ? error.message : "Не удалось сохранить аннотацию");
        },
      }
    );
  };

  const handleBack = () => navigate(paths.standardDetail(groupId, standardId));
  const handlePrev = () =>
    prevImage && navigate(paths.standardImage(groupId, standardId, prevImage.id));
  const handleNext = () =>
    nextImage && navigate(paths.standardImage(groupId, standardId, nextImage.id));
  const handleNextUnannotated = () =>
    nextUnannotatedImage &&
    navigate(paths.standardImage(groupId, standardId, nextUnannotatedImage.id));

  const handleFinishDrawing = (draftContour: number[][]) => {
    if (!selectedSegmentClassId || !availableSegmentClassIds.has(selectedSegmentClassId)) {
      return;
    }

    const existingContours = selectedImageSegmentClass?.points ?? [];
    saveSegmentContours(selectedSegmentClassId, [...existingContours, draftContour]);
    setSelectedContourIndex(existingContours.length);
  };

  const handleDeleteContour = (segmentClassId: string, contourIndex: number) => {
    const imageSegmentClass = imageSegmentClasses.find((item) => item.id === segmentClassId);
    if (!imageSegmentClass) {
      return;
    }

    const nextContours = imageSegmentClass.points.filter((_, index) => index !== contourIndex);

    saveSegmentContours(segmentClassId, nextContours);
    setSelectedContourIndex(null);
  };

  const handleStartDraw = (kind: DrawKind) => {
    if (!canDraw) {
      setAnnotationError("Сначала выбери класс справа, потом включай Draw или Smart.");
      return;
    }

    setAnnotationError(null);
    setMode(drawKindToMode[kind]);
  };
  const handleCancelDraw = () => setMode("view");

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tagName = target?.tagName.toLowerCase();

      if (target?.isContentEditable || tagName === "input" || tagName === "textarea" || tagName === "select") {
        return;
      }

      const key = event.key.toLowerCase();

      if (key === "escape") {
        handleCancelDraw();
        return;
      }

      if (key === "arrowleft") {
        event.preventDefault();
        handlePrev();
        return;
      }

      if (key === "arrowright") {
        event.preventDefault();
        handleNext();
        return;
      }

      if (key === "v") {
        setMode("view");
        return;
      }

      if (key === "d") {
        handleStartDraw("polygon");
        return;
      }

      if (key === "s") {
        handleStartDraw("sam");
        return;
      }

      if (key === "u") {
        event.preventDefault();
        handleNextUnannotated();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleCancelDraw, handleNext, handleNextUnannotated, handlePrev, handleStartDraw]);

  const panel = (
    <SegmentPanel
      group={group}
      categories={categories}
      ungroupedClasses={ungroupedClasses}
      standardId={standardId}
      imageId={imageId}
      imageSegmentClasses={imageSegmentClasses}
      selectedSegmentClassId={selectedSegmentClassId}
      onSelectSegmentClass={(id) => {
        setAnnotationError(null);
        setSelectedSegmentClassId(id);
      }}
      onStartDraw={handleStartDraw}
      selectedContourIndex={selectedContourIndex}
      onSelectContour={setSelectedContourIndex}
      onDeleteContour={handleDeleteContour}
      annotationError={annotationError}
    />
  );

  return (
    <div className={s.editorShell}>
      <div className={s.editorTopbar}>
        <SegmentHeader
          standardName={standard.name}
          currentIndex={safeIndex}
          totalImages={imageIds.length}
          isReference={image.is_reference}
          isCurrentAnnotated={isCurrentAnnotated}
          segmentsCount={standard.stats.segment_classes_count}
          canGoPrev={!!prevImage}
          canGoNext={!!nextImage}
          hasNextUnannotated={!!nextUnannotatedImage}
          onPrev={handlePrev}
          onNext={handleNext}
          onNextUnannotated={handleNextUnannotated}
          onBack={handleBack}
        />
      </div>

      <div className={s.editorBody}>
        <aside className={s.filmstrip}>
          <div className={s.filmstripHead}>
            <ImageIcon />
            <div className={s.filmstripTitle}>
              <strong>Images</strong>
              <small>{standard.name}</small>
            </div>
            <span className={s.filmstripCount}>{annotatedCount}/{standard.images.length}</span>
            <div className={s.filmstripProgress}>
              <span style={{ width: `${annotationProgress}%` }} />
            </div>
          </div>
          <div className={s.thumbs}>
            {standard.images.map((item, index) => {
              const isActive = item.id === imageId;
              const isDone = item.annotation_count > 0;
              return (
                <button
                  type="button"
                  key={item.id}
                  className={clsx(s.thumb, isActive && s.thumbActive)}
                  title={`Image ${index + 1} · ${isDone ? `${item.annotation_count} polygons` : "not labeled"}`}
                  onClick={() => navigate(paths.standardImage(groupId, standardId, item.id))}
                >
                  <img src={`/storage/${item.image_path}`} alt={`Image ${index + 1}`} />
                  <span className={s.thumbMiniStats}>{index + 1}</span>
                  <div className={s.thumbOverlay}>
                    <span className={s.thumbIndex}>#{index + 1}</span>
                    <span className={s.thumbLabel}>
                      {item.is_reference ? "Reference" : isDone ? `${item.annotation_count} polygons` : "Not labeled"}
                    </span>
                    <span className={clsx(s.thumbState, isDone && s.thumbStateDone)}>
                      {isDone ? <CheckCircle2 /> : <Circle />}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        <main className={s.canvasColumn}>
          <div className={s.floatingToolstrip}>
            <button
              type="button"
              className={clsx(mode === "view" && s.toolActive)}
              onClick={() => setMode("view")}
            >
              <MousePointer2 /> View <kbd>V</kbd>
            </button>
            <button
              type="button"
              className={clsx(mode === "draw-polygon" && s.toolActive)}
              onClick={() => handleStartDraw("polygon")}
            >
              <PenLine /> Draw <kbd>D</kbd>
            </button>
            <button
              type="button"
              className={clsx(mode === "draw-sam" && s.toolActive)}
              onClick={() => handleStartDraw("sam")}
            >
              <Sparkles /> Smart <kbd>S</kbd>
            </button>
            {mode !== "view" ? (
              <button type="button" onClick={handleCancelDraw}>
                <X /> Cancel
              </button>
            ) : null}
          </div>

          <div className={s.canvasMeta}>
            <span>{group.name}</span>
            <span>{safeIndex + 1} / {imageIds.length}</span>
            <span>{polygonsCount} polygons</span>
            <span>{mode === "view" ? "View" : mode === "draw-polygon" ? "Draw" : "Smart"}</span>
          </div>

          {!selectedSegmentClassId ? (
            <div className={s.selectClassHint}>Выбери класс справа, затем Draw или Smart</div>
          ) : null}

          <div className={s.hotkeyRow}>
            <kbd>←</kbd><span>prev</span>
            <kbd>→</kbd><span>next</span>
            <kbd>U</kbd><span>empty</span>
          </div>

          <div className={clsx(s.reviewCard, isCurrentAnnotated && s.reviewCardDone)}>
            <span>{isCurrentAnnotated ? "Labeled" : "Needs label"}</span>
            <strong>{polygonsCount} polygons</strong>
            <small>{selectedSegmentClass ? `Class: ${selectedSegmentClass.name}` : "No class selected"}</small>
          </div>

          <CanvasSurface
            imageUrl={imageUrl}
            imageId={imageId}
            segments={imageSegmentClasses}
            selectedId={selectedSegmentClassId}
            onSelect={setSelectedSegmentClassId}
            onPointsChange={saveSegmentContours}
            onFinishDrawing={handleFinishDrawing}
            mode={mode}
            onCancelDraw={handleCancelDraw}
            selectedContourIndex={selectedContourIndex}
            onSelectContour={setSelectedContourIndex}
          />

          <div className={s.bottomNavigator}>
            <button type="button" disabled={!prevImage} onClick={handlePrev}>
              <ChevronLeft /> Prev
            </button>
            <div className={s.bottomProgressPanel}>
              <div className={s.bottomProgressText}>
                <span>Image {safeIndex + 1} of {imageIds.length}</span>
                <span>{annotationProgress}% labeled</span>
              </div>
              <div className={s.bottomProgressTrack}>
                <span style={{ width: `${currentProgress}%` }} />
              </div>
            </div>
            <button className={s.nextButton} type="button" disabled={!nextImage} onClick={handleNext}>
              Next <ChevronRight />
            </button>
            <button className={s.skipButton} type="button" disabled={!nextUnannotatedImage} onClick={handleNextUnannotated}>
              Empty <SkipForward />
            </button>
          </div>
        </main>

        <aside className={s.editorPanel}>
          <div className={clsx(s.panelIntro, selectedSegmentClass && s.panelIntroSelected)}>
            {selectedSegmentClass ? (
              <span
                className={s.selectedDot}
                style={{ background: `hsl(${selectedSegmentClass.hue}, 70%, 50%)` }}
              />
            ) : (
              <Layers3 />
            )}
            <div className={s.panelIntroText}>
              <strong>{selectedSegmentClass?.name ?? "Classes"}</strong>
              <span>{selectedSegmentClass ? "Selected class" : `${standard.stats.segment_classes_count} classes`} · {polygonsCount} polygons</span>
            </div>
          </div>
          {panel}
        </aside>
      </div>
    </div>
  );
};
