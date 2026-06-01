import { useSamDraft } from "@/page-components/segments/hooks/use-sam-draft";
import {
  clamp,
  EDGE_HIT_RADIUS,
  projectOnEdge,
  readStagePointer,
  SNAP_RADIUS,
} from "@/page-components/segments/lib/canvas";
import type { SegmentClassWithPoints } from "@/types/contracts";
import type Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";

export type EditorMode = "view" | "draw-polygon" | "draw-sam";

type ImageRect = {
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
  scale: number;
};

type Params = {
  imageId: string;
  image: HTMLImageElement | null;
  imageRect: ImageRect;
  segments: SegmentClassWithPoints[];
  selectedId: string | null;
  selectedContourIndex: number | null;
  mode: EditorMode;
  stageRef: React.RefObject<Konva.Stage | null>;
  isSpaceDown: React.RefObject<boolean>;
  toImage: (cx: number, cy: number) => [number, number];
  toCanvas: (ix: number, iy: number) => [number, number];
  clampToImage: (cx: number, cy: number) => [number, number];
  onSelect: (id: string | null) => void;
  onSelectContour: (index: number | null) => void;
  onFinishDrawing: (points: number[][]) => void;
  onPointsChange: (segmentId: string, points: number[][][]) => void;
  consumePanClick: () => boolean;
  applyPan: () => boolean;
};

type LineNode = Konva.Line;
type ContourKey = string;

export function useAnnotationEditor(params: Params) {
  const {
    imageId,
    image,
    imageRect,
    segments,
    selectedId,
    selectedContourIndex,
    mode,
    stageRef,
    isSpaceDown,
    toImage,
    toCanvas,
    clampToImage,
    onSelect,
    onSelectContour,
    onFinishDrawing,
    onPointsChange,
    consumePanClick,
    applyPan,
  } = params;

  const contourFillRefs = useRef<Record<ContourKey, LineNode>>({});
  const contourLineRefs = useRef<Record<ContourKey, LineNode>>({});
  const contourOutlineRefs = useRef<Record<ContourKey, LineNode>>({});
  const draftOutlineRef = useRef<LineNode | null>(null);
  const draftLineRef = useRef<LineNode | null>(null);
  const pendingResetNode = useRef<Konva.Node | null>(null);
  const isDragging = useRef(false);

  const [draftPoints, setDraftPoints] = useState<number[][]>([]);
  const [draftPreviewPoint, setDraftPreviewPoint] = useState<[number, number] | null>(null);

  const isDrawing = mode !== "view";

  const sam = useSamDraft({ isActive: mode === "draw-sam", imageId });

  useLayoutEffect(() => {
    if (mode !== "draw-polygon") {
      setDraftPoints([]);
      setDraftPreviewPoint(null);
    }
  }, [mode, selectedId]);

  useLayoutEffect(() => {
    if (!pendingResetNode.current) return;
    const node = pendingResetNode.current;
    pendingResetNode.current = null;
    node.x(0);
    node.y(0);
    node.getLayer()?.batchDraw();
  });

  const selectedSeg = useMemo(
    () => segments.find((s) => s.id === selectedId),
    [segments, selectedId]
  );
  const selectedContour =
    selectedSeg && selectedContourIndex !== null
      ? (selectedSeg.points[selectedContourIndex] ?? null)
      : null;

  const hueOf = (seg: SegmentClassWithPoints | undefined) => seg?.hue ?? 0;
  const draftStroke = `hsl(${hueOf(selectedSeg)}, 65%, 45%)`;
  const draftCanvasPoints = draftPoints.map(([ix, iy]) => toCanvas(ix, iy));

  const defaultCursor = isDrawing ? "crosshair" : "default";

  const modeLabel = getModeLabel(mode);
  const hintText = getHint({
    mode,
    selectedId,
    hasSelectedContour: selectedContour !== null,
  });

  const setCursor = useCallback((e: KonvaEventObject<MouseEvent>, cursor: string) => {
    const container = e.target.getStage()?.container();
    if (container) container.style.cursor = cursor;
  }, []);

  const clearPreviewState = useCallback(() => setDraftPreviewPoint(null), []);

  const updateLine = useCallback(
    (
      lineNodes: Array<LineNode | null | undefined>,
      canvasPts: number[][],
      vi: number,
      x: number,
      y: number
    ) => {
      const pts = [...canvasPts];
      pts[vi] = [x, y];
      const flat = pts.flat();
      for (const node of lineNodes) {
        if (!node) continue;
        node.points(flat);
        node.getLayer()?.batchDraw();
      }
    },
    []
  );

  const readPointer = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return null;
    return readStagePointer(stage);
  }, [stageRef]);

  const moveContour = (seg: SegmentClassWithPoints, ci: number, dx: number, dy: number) => {
    const next = seg.points[ci].map(([ix, iy]) => {
      const [cx, cy] = toCanvas(ix, iy);
      return toImage(cx + dx, cy + dy);
    });
    onPointsChange(
      seg.id,
      seg.points.map((c, i) => (i === ci ? next : c))
    );
  };

  const moveVertex = (seg: SegmentClassWithPoints, ci: number, vi: number, nextPoint: number[]) => {
    onPointsChange(
      seg.id,
      seg.points.map((c, ciIdx) =>
        ciIdx === ci ? c.map((p, vIdx) => (vIdx === vi ? nextPoint : p)) : c
      )
    );
  };

  const removeVertex = (seg: SegmentClassWithPoints, ci: number, vi: number) => {
    onPointsChange(
      seg.id,
      seg.points.map((c, ciIdx) => (ciIdx === ci ? c.filter((_, i) => i !== vi) : c))
    );
  };

  const insertVertexOnEdge = (seg: SegmentClassWithPoints, ci: number, px: number, py: number) => {
    const canvasPts = seg.points[ci].map(([ix, iy]) => toCanvas(ix, iy));
    const projected = projectOnEdge(canvasPts, px, py);
    if (projected.dist > EDGE_HIT_RADIUS || !projected.point) return;
    const nextContour = [...seg.points[ci]];
    nextContour.splice(projected.index, 0, toImage(projected.point[0], projected.point[1]));
    onPointsChange(
      seg.id,
      seg.points.map((c, i) => (i === ci ? nextContour : c))
    );
  };

  const finishDraftPolygon = () => {
    if (draftPoints.length < 3) return;
    onFinishDrawing([...draftPoints]);
    setDraftPoints([]);
  };

  const cancelPolygonDraft = useCallback(() => {
    setDraftPoints([]);
    clearPreviewState();
  }, [clearPreviewState]);

  const tryFinishDraftPolygon = (cx: number, cy: number) => {
    if (draftPoints.length < 3) return false;
    const [fx, fy] = toCanvas(draftPoints[0][0], draftPoints[0][1]);
    if (Math.hypot(cx - fx, cy - fy) >= SNAP_RADIUS) return false;
    finishDraftPolygon();
    return true;
  };

  const handleStageClick = (e: KonvaEventObject<MouseEvent>) => {
    if (e.evt.button !== 0) return;
    if (consumePanClick()) return;

    const ptr = readPointer();
    if (!ptr) return;
    const { px, py } = ptr;

    if (mode === "view") {
      const clickedEmpty = e.target === e.target.getStage() || e.target.getClassName() === "Image";
      if (clickedEmpty) onSelect(null);
      return;
    }

    if (mode === "draw-polygon") {
      if (!image || !selectedId) return;
      if (e.target.getClassName?.() === "Rect") return;
      const [cx, cy] = clampToImage(px, py);
      if (tryFinishDraftPolygon(cx, cy)) return;
      setDraftPoints((prev) => [...prev, toImage(cx, cy)]);
      return;
    }

    if (mode === "draw-sam") {
      if (!selectedId) return;
      const [cx, cy] = clampToImage(px, py);
      sam.addPositive(toImage(cx, cy));
      return;
    }
  };

  const handleMouseMove = () => {
    if (applyPan()) return;
    if (isDragging.current) return;
    const ptr = readPointer();
    if (!ptr) return;
    const { px, py } = ptr;

    if (mode === "draw-polygon") {
      if (draftPoints.length === 0) {
        clearPreviewState();
        return;
      }
      setDraftPreviewPoint(clampToImage(px, py));
      return;
    }

    if (mode === "view") clearPreviewState();
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();

    if (mode === "view") {
      onSelect(null);
      onSelectContour(null);
      return;
    }

    if (mode === "draw-polygon") {
      cancelPolygonDraft();
      return;
    }

    if (mode === "draw-sam") {
      const ptr = readPointer();
      if (!ptr) return;
      const [cx, cy] = clampToImage(ptr.px, ptr.py);
      sam.addNegative(toImage(cx, cy));
      return;
    }
  };

  const handleMouseLeave = () => {
    clearPreviewState();
  };

  const handleLineClick = (
    seg: SegmentClassWithPoints,
    ci: number,
    e: KonvaEventObject<MouseEvent>
  ) => {
    if (mode !== "view") return;
    if (e.evt.button !== 0) return;
    if (isSpaceDown.current) return;

    if (seg.id !== selectedId) {
      onSelect(seg.id);
      onSelectContour(ci);
      return;
    }
    e.cancelBubble = true;
    onSelectContour(ci);

    const ptr = readPointer();
    if (!ptr) return;
    insertVertexOnEdge(seg, ci, ptr.px, ptr.py);
  };

  const handleGroupDragStart = (e: KonvaEventObject<DragEvent>) => {
    if (isSpaceDown.current) {
      (e.target as Konva.Node).stopDrag();
      return;
    }
    isDragging.current = true;
  };

  const handleGroupDragEnd = (
    seg: SegmentClassWithPoints,
    ci: number,
    e: KonvaEventObject<DragEvent>
  ) => {
    isDragging.current = false;
    const node = e.target as Konva.Node;
    const dx = node.x();
    const dy = node.y();
    pendingResetNode.current = node;
    moveContour(seg, ci, dx, dy);
  };

  const handleVertexDragStart = (e: KonvaEventObject<DragEvent>) => {
    if (isSpaceDown.current) {
      (e.target as Konva.Node).stopDrag();
      return;
    }
    e.cancelBubble = true;
    isDragging.current = true;
    clearPreviewState();
  };

  const handleVertexDragMove = (
    seg: SegmentClassWithPoints,
    ci: number,
    vi: number,
    e: KonvaEventObject<DragEvent>
  ) => {
    e.cancelBubble = true;
    const canvasPts = seg.points[ci].map(([x, y]) => toCanvas(x, y));
    const key = `${seg.id}-${ci}`;
    updateLine(
      [contourFillRefs.current[key], contourOutlineRefs.current[key], contourLineRefs.current[key]],
      canvasPts,
      vi,
      (e.target as Konva.Node).x(),
      (e.target as Konva.Node).y()
    );
  };

  const handleVertexDragEnd = (
    seg: SegmentClassWithPoints,
    ci: number,
    vi: number,
    e: KonvaEventObject<DragEvent>
  ) => {
    e.cancelBubble = true;
    isDragging.current = false;
    const t = e.target as Konva.Node;
    const pt = toImage(t.x(), t.y());
    moveVertex(seg, ci, vi, pt);
  };

  const handleVertexDblClick = (
    seg: SegmentClassWithPoints,
    ci: number,
    _vi: number,
    e: KonvaEventObject<MouseEvent>
  ) => {
    if (e.evt.button !== 0) return;
    if (mode !== "view") return;
    e.cancelBubble = true;
    if (seg.points[ci].length <= 3) return;
    removeVertex(seg, ci, _vi);
  };

  const handleDraftVertexDragStart = (e: KonvaEventObject<DragEvent>) => {
    e.cancelBubble = true;
    isDragging.current = true;
    clearPreviewState();
  };

  const handleDraftVertexDragMove = (vi: number, e: KonvaEventObject<DragEvent>) => {
    e.cancelBubble = true;
    const canvasPts = draftPoints.map(([ix, iy]) => toCanvas(ix, iy));
    updateLine(
      [draftOutlineRef.current, draftLineRef.current],
      canvasPts,
      vi,
      (e.target as Konva.Node).x(),
      (e.target as Konva.Node).y()
    );
  };

  const handleDraftVertexDragEnd = (vi: number, e: KonvaEventObject<DragEvent>) => {
    e.cancelBubble = true;
    isDragging.current = false;
    setDraftPreviewPoint(null);
    const t = e.target as Konva.Node;
    const point = toImage(t.x(), t.y());
    setDraftPoints((prev) => prev.map((p, i) => (i === vi ? point : p)));
  };

  const handleDraftVertexClick = (vi: number, e: KonvaEventObject<MouseEvent>) => {
    if (e.evt.button !== 0) return;
    e.cancelBubble = true;
    if (isDragging.current) return;

    if (mode === "draw-polygon") {
      if (draftPoints.length <= 3) return;
      if (vi !== 0) return;
      finishDraftPolygon();
    }
  };

  const handleDraftVertexDblClick = (vi: number, e: KonvaEventObject<MouseEvent>) => {
    if (e.evt.button !== 0) return;
    e.cancelBubble = true;

    if (mode === "draw-polygon") {
      if (vi === 0) return;
      if (draftPoints.length <= 1) return;
      setDraftPoints((prev) => prev.filter((_, i) => i !== vi));
    }
  };

  const stageVertexBound = useCallback(
    (pos: { x: number; y: number }) => {
      const stage = stageRef.current;
      if (!stage) return pos;
      const sc = stage.scaleX();
      const sx = stage.x();
      const sy = stage.y();
      const lx = (pos.x - sx) / sc;
      const ly = (pos.y - sy) / sc;
      const [cx, cy] = clampToImage(lx, ly);
      return { x: cx * sc + sx, y: cy * sc + sy };
    },
    [clampToImage, stageRef]
  );

  const getGroupBound = useCallback(
    (seg: SegmentClassWithPoints, ci: number) => (pos: { x: number; y: number }) => {
      const stage = stageRef.current;
      if (!stage) return pos;
      const sc = stage.scaleX();
      const sx = stage.x();
      const sy = stage.y();
      const lx = (pos.x - sx) / sc;
      const ly = (pos.y - sy) / sc;

      const pts = seg.points[ci].map(([x, y]) => toCanvas(x, y));
      const xs = pts.map(([x]) => x);
      const ys = pts.map(([, y]) => y);

      const cx = clamp(
        lx,
        imageRect.offsetX - Math.min(...xs),
        imageRect.offsetX + imageRect.width - Math.max(...xs)
      );
      const cy = clamp(
        ly,
        imageRect.offsetY - Math.min(...ys),
        imageRect.offsetY + imageRect.height - Math.max(...ys)
      );
      return { x: cx * sc + sx, y: cy * sc + sy };
    },
    [stageRef, toCanvas, imageRect]
  );

  const confirmSamDraft = useCallback(() => {
    if (mode !== "draw-sam") return;
    if (!sam.draftPolygon || sam.draftPolygon.length < 3) return;
    onFinishDrawing(sam.draftPolygon);
    sam.reset();
  }, [mode, sam, onFinishDrawing]);

  const hasPendingDraft =
    (mode === "draw-polygon" && draftPoints.length > 0) ||
    (mode === "draw-sam" &&
      (sam.positive.length > 0 || sam.negative.length > 0 || !!sam.draftPolygon || sam.isLoading));

  const cancelDraft = useCallback(() => {
    if (mode === "draw-polygon") {
      cancelPolygonDraft();
      return;
    }

    if (mode === "draw-sam") {
      sam.reset();
    }
  }, [cancelPolygonDraft, mode, sam]);

  const confirmDraft = useCallback(() => {
    if (mode === "draw-polygon") {
      finishDraftPolygon();
      return;
    }

    if (mode === "draw-sam") {
      confirmSamDraft();
    }
  }, [confirmSamDraft, mode]);

  return {
    isDrawing,
    hasPendingDraft,
    defaultCursor,
    modeLabel,
    hintText,
    draftPoints,
    draftCanvasPoints,
    draftPreviewPoint,
    draftStroke,
    sam,
    cancelDraft,
    confirmDraft,
    confirmSamDraft,
    handleStageClick,
    handleMouseMove,
    handleContextMenu,
    handleMouseLeave,
    handleLineClick,
    handleGroupDragStart,
    handleGroupDragEnd,
    handleVertexDragStart,
    handleVertexDragMove,
    handleVertexDragEnd,
    handleVertexDblClick,
    handleDraftVertexDragStart,
    handleDraftVertexDragMove,
    handleDraftVertexDragEnd,
    handleDraftVertexClick,
    handleDraftVertexDblClick,
    contourFillRefs,
    contourLineRefs,
    contourOutlineRefs,
    draftOutlineRef,
    draftLineRef,
    hueOf,
    setCursor,
    stageVertexBound,
    getGroupBound,
  };
}

function getModeLabel(mode: EditorMode): string {
  if (mode === "draw-polygon") return "Режим полигона";
  if (mode === "draw-sam") return "Режим SAM";
  return "Режим просмотра";
}

function getHint({
  mode,
  selectedId,
  hasSelectedContour,
}: {
  mode: EditorMode;
  selectedId: string | null;
  hasSelectedContour: boolean;
}): string {
  if (mode === "view") {
    if (!selectedId) return "Выберите класс";
    if (hasSelectedContour)
      return "ЛКМ: перетащить вершину/сегмент | ПКМ: сбросить | Колесо: масштаб | Space: панорама";
    return "Колесо: масштаб | Space: панорама | ПКМ: сбросить выделение";
  }

  if (mode === "draw-polygon") {
    if (!selectedId) return "Выберите класс";
    return "ЛКМ: добавить вершину | ПКМ: сбросить черновик | Enter: замкнуть | Esc: сбросить/выйти";
  }

  if (mode === "draw-sam") {
    if (!selectedId) return "Выберите класс";
    return "ЛКМ: позитивная точка | ПКМ: негативная | Enter: сохранить | Esc: сбросить/выйти";
  }

  return "";
}
