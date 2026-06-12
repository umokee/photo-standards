import { paths } from "@/app/paths";
import { SplitLayout } from "@/components/layouts/split-layout/split-layout";
import { Modal } from "@/components/ui/modal/modal";
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
import { useEffect, useMemo, useState } from "react";
import { useLoaderData, useNavigate } from "react-router-dom";

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
  const availableSegmentClassIds = useMemo(
    () =>
      new Set([
        ...categories.flatMap((category) => category.segment_classes.map((item) => item.id)),
        ...ungroupedClasses.map((item) => item.id),
      ]),
    [categories, ungroupedClasses]
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
    annotate.mutate({
      segmentClassId,
      imageId,
      points: nextContours,
    });
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

  const handleStartDraw = (kind: DrawKind) => setMode(drawKindToMode[kind]);
  const handleCancelDraw = () => setMode("view");

  const panel = (
    <SegmentPanel
      group={group}
      categories={categories}
      ungroupedClasses={ungroupedClasses}
      standardId={standardId}
      imageId={imageId}
      imageSegmentClasses={imageSegmentClasses}
      selectedSegmentClassId={selectedSegmentClassId}
      onSelectSegmentClass={setSelectedSegmentClassId}
      onStartDraw={handleStartDraw}
      selectedContourIndex={selectedContourIndex}
      onSelectContour={setSelectedContourIndex}
      onDeleteContour={handleDeleteContour}
    />
  );

  return (
    <Modal>
      <SplitLayout>
        <SplitLayout.Content>
          <SplitLayout.Topbar>
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
          </SplitLayout.Topbar>

          <SplitLayout.Body bare>
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
          </SplitLayout.Body>
        </SplitLayout.Content>

        <SplitLayout.Panel>{panel}</SplitLayout.Panel>
      </SplitLayout>
    </Modal>
  );
};
