import { clamp } from "@/page-components/segments/lib/canvas";
import type Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { useCallback, useEffect, useRef, useState } from "react";

export function useViewport(defaultCursor: string) {
  const stageRef = useRef<Konva.Stage | null>(null);
  const isSpaceDown = useRef(false);
  const isPanning = useRef(false);
  const panConsumedClick = useRef(false);
  const lastPanPos = useRef({ x: 0, y: 0 });

  const [viewportScale, setViewportScale] = useState(1);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return;
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        isSpaceDown.current = true;
        if (stageRef.current) {
          stageRef.current.container().style.cursor = "grab";
        }
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return;
      if (e.code === "Space") {
        isSpaceDown.current = false;
        if (stageRef.current) {
          stageRef.current.container().style.cursor = defaultCursor;
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [defaultCursor]);

  const handleWheel = useCallback((e: KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const newScale = clamp(e.evt.deltaY < 0 ? oldScale * 1.1 : oldScale / 1.1, 1, 20);

    stage.scale({ x: newScale, y: newScale });
    setViewportScale(newScale);

    if (newScale === 1) {
      stage.position({ x: 0, y: 0 });
      return;
    }

    stage.position({
      x: pointer.x - (pointer.x - stage.x()) * (newScale / oldScale),
      y: pointer.y - (pointer.y - stage.y()) * (newScale / oldScale),
    });
  }, []);

  const handleStageMouseDown = useCallback((e: KonvaEventObject<MouseEvent>) => {
    if (!isSpaceDown.current) return;
    isPanning.current = true;
    panConsumedClick.current = true;
    const pos = e.target.getStage()?.getPointerPosition();
    if (pos) lastPanPos.current = pos;
    const container = e.target.getStage()?.container();
    if (container) container.style.cursor = "grabbing";
  }, []);

  const handleStageMouseUp = useCallback(() => {
    if (!isPanning.current) return;
    isPanning.current = false;
    if (stageRef.current) {
      stageRef.current.container().style.cursor = isSpaceDown.current ? "grab" : defaultCursor;
    }
  }, [defaultCursor]);

  const applyPan = useCallback(() => {
    if (!isPanning.current) return false;
    const stage = stageRef.current;
    if (!stage) return false;
    const pos = stage.getPointerPosition();
    if (!pos) return false;
    stage.position({
      x: stage.x() + pos.x - lastPanPos.current.x,
      y: stage.y() + pos.y - lastPanPos.current.y,
    });
    lastPanPos.current = pos;
    return true;
  }, []);

  const consumePanClick = useCallback(() => {
    if (!panConsumedClick.current) return false;
    panConsumedClick.current = false;
    return true;
  }, []);

  return {
    stageRef,
    isSpaceDown,
    isPanning,
    viewportScale,
    handleWheel,
    handleStageMouseDown,
    handleStageMouseUp,
    applyPan,
    consumePanClick,
  };
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable;
}
