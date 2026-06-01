import {
  useAnnotationEditor,
  type EditorMode,
} from "@/page-components/segments/hooks/use-annotation-editor";
import { useCanvasKeyboard } from "@/page-components/segments/hooks/use-canvas-keyboard";
import useImageLayout from "@/page-components/segments/hooks/use-image-layout";
import { useViewport } from "@/page-components/segments/hooks/use-viewport";
import { hasPoints, segmentColor } from "@/page-components/segments/lib/canvas";
import { SegmentClassWithPoints } from "@/types/contracts";
import { Group, Layer, Stage } from "react-konva";
import useImage from "use-image";
import s from "./canvas-surface.module.scss";
import { CanvasImage } from "./primitives/canvas-image";
import { CanvasOverlay } from "./primitives/canvas-overlay";
import { DraftContour } from "./primitives/draft-contour";
import { PolygonContour } from "./primitives/polygon-contour";
import { SamDraft } from "./primitives/sam-draft";
import { VertexHandle } from "./primitives/vertex-handle";

import { useEffect } from "react";

export { type EditorMode };

interface Props {
  imageUrl: string | null;
  imageId: string;
  segments: SegmentClassWithPoints[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onFinishDrawing: (points: number[][]) => void;
  selectedContourIndex: number | null;
  onSelectContour: (index: number | null) => void;
  mode: EditorMode;
  onCancelDraw: () => void;
  onPointsChange: (segmentId: string, points: number[][][]) => void;
}

export default function CanvasSurface({
  imageUrl,
  imageId,
  segments,
  selectedId,
  onSelect,
  onFinishDrawing,
  selectedContourIndex,
  onSelectContour,
  mode,
  onCancelDraw,
  onPointsChange,
}: Props) {
  const [image] = useImage(imageUrl ?? "", "anonymous");
  const { containerRef, size, imageRect, toImage, toCanvas, clampToImage } = useImageLayout(image);

  const viewport = useViewport(mode === "view" ? "default" : "crosshair");

  const editor = useAnnotationEditor({
    imageId,
    image,
    imageRect,
    segments,
    selectedId,
    selectedContourIndex,
    mode,
    stageRef: viewport.stageRef,
    isSpaceDown: viewport.isSpaceDown,
    toImage,
    toCanvas,
    clampToImage,
    onSelect,
    onSelectContour,
    onFinishDrawing,
    onPointsChange,
    consumePanClick: viewport.consumePanClick,
    applyPan: viewport.applyPan,
  });

  useCanvasKeyboard({
    enabled: mode !== "view",
    onEscape: () => {
      if (editor.hasPendingDraft) {
        const shouldDiscardDraft = window.confirm(
          "Сбросить текущий черновик и остаться в режиме рисования?"
        );
        if (!shouldDiscardDraft) return;
        editor.cancelDraft();
        return;
      }

      onCancelDraw();
    },
  });

  useEffect(() => {
    if (mode !== "draw-sam") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Enter") {
        e.preventDefault();
        editor.confirmDraft();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, editor.confirmDraft]);

  const isPolygonDraft = mode === "draw-polygon";
  const isSamDraft = mode === "draw-sam";

  return (
    <div className={s.root} ref={containerRef} onContextMenu={editor.handleContextMenu}>
      <CanvasOverlay
        isDrawing={editor.isDrawing}
        modeLabel={editor.modeLabel}
        hintText={editor.hintText}
      />
      <Stage
        ref={viewport.stageRef}
        width={size.width}
        height={size.height}
        style={{ cursor: editor.defaultCursor }}
        onWheel={viewport.handleWheel}
        onMouseDown={viewport.handleStageMouseDown}
        onMouseMove={editor.handleMouseMove}
        onMouseUp={viewport.handleStageMouseUp}
        onClick={editor.handleStageClick}
        onMouseLeave={editor.handleMouseLeave}
      >
        <CanvasImage
          image={image ?? undefined}
          x={imageRect.offsetX}
          y={imageRect.offsetY}
          width={imageRect.width}
          height={imageRect.height}
        />

        <Layer>
          {segments.map((seg) => {
            if (!hasPoints(seg)) return null;
            const isSelected = seg.id === selectedId;
            const { stroke, fill } = segmentColor(editor.hueOf(seg), isSelected);

            return (
              <Group key={seg.id}>
                {seg.points.map((contour, ci) => {
                  const canvasPts = contour.map(([ix, iy]: number[]) => toCanvas(ix, iy));
                  const isEditable = isSelected && selectedContourIndex === ci && !editor.isDrawing;

                  return (
                    <PolygonContour
                      key={ci}
                      contourKey={ci}
                      canvasPoints={canvasPts}
                      stroke={stroke}
                      fill={fill}
                      viewportScale={viewport.viewportScale}
                      fillOpacity={
                        editor.isDrawing ? 0.05 : isEditable ? 0.22 : isSelected ? 0.14 : 0.07
                      }
                      strokeOpacity={isEditable ? 1 : isSelected ? 0.9 : 0.76}
                      strokeWidth={isEditable ? 2.4 : isSelected ? 1.9 : 1.25}
                      isEditable={isEditable}
                      isSelected={isEditable || isSelected}
                      dragBoundFunc={editor.getGroupBound(seg, ci)}
                      onDragStart={editor.handleGroupDragStart}
                      onDragEnd={(e) => editor.handleGroupDragEnd(seg, ci, e)}
                      onLineClick={(e) => editor.handleLineClick(seg, ci, e)}
                      fillRef={(node) => {
                        if (node) editor.contourFillRefs.current[`${seg.id}-${ci}`] = node;
                      }}
                      outlineRef={(node) => {
                        if (node) editor.contourOutlineRefs.current[`${seg.id}-${ci}`] = node;
                      }}
                      lineRef={(node) => {
                        if (node) editor.contourLineRefs.current[`${seg.id}-${ci}`] = node;
                      }}
                    >
                      {isEditable &&
                        canvasPts.map(([cx, cy], vi) => (
                          <VertexHandle
                            key={vi}
                            x={cx}
                            y={cy}
                            stroke={stroke}
                            viewportScale={viewport.viewportScale}
                            draggable
                            dragBoundFunc={editor.stageVertexBound}
                            onDragStart={editor.handleVertexDragStart}
                            onDragMove={(e) => editor.handleVertexDragMove(seg, ci, vi, e)}
                            onDragEnd={(e) => editor.handleVertexDragEnd(seg, ci, vi, e)}
                            onDblClick={(e) => editor.handleVertexDblClick(seg, ci, vi, e)}
                            onMouseEnter={(e) => editor.setCursor(e, "grab")}
                            onMouseLeave={(e) => editor.setCursor(e, editor.defaultCursor)}
                          />
                        ))}
                    </PolygonContour>
                  );
                })}
              </Group>
            );
          })}
        </Layer>

        {isPolygonDraft && (
          <DraftContour
            points={editor.draftCanvasPoints}
            stroke={editor.draftStroke}
            ghostPoint={editor.draftPreviewPoint}
            outlineRef={(node) => {
              editor.draftOutlineRef.current = node;
            }}
            viewportScale={viewport.viewportScale}
            lineRef={(node) => {
              editor.draftLineRef.current = node;
            }}
            dragBoundFunc={editor.stageVertexBound}
            onVertexDragStart={editor.handleDraftVertexDragStart}
            onVertexDragMove={editor.handleDraftVertexDragMove}
            onVertexDragEnd={editor.handleDraftVertexDragEnd}
            onVertexClick={editor.handleDraftVertexClick}
            onVertexDblClick={editor.handleDraftVertexDblClick}
            onVertexMouseEnter={(e) => editor.setCursor(e, "grab")}
            onVertexMouseLeave={(e) => editor.setCursor(e, "crosshair")}
          />
        )}

        {isSamDraft && (
          <SamDraft
            positiveCanvas={editor.sam.positive.map(([ix, iy]) => toCanvas(ix, iy))}
            negativeCanvas={editor.sam.negative.map(([ix, iy]) => toCanvas(ix, iy))}
            draftPolygonCanvas={
              editor.sam.draftPolygon
                ? editor.sam.draftPolygon.map(([ix, iy]) => toCanvas(ix, iy))
                : null
            }
            stroke={editor.draftStroke}
            viewportScale={viewport.viewportScale}
            isLoading={editor.sam.isLoading}
          />
        )}
      </Stage>
    </div>
  );
}
