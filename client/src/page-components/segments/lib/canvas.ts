import { SegmentClassWithPoints } from "@/types/contracts";
import Konva from "konva";

export const SNAP_RADIUS = 10;
export const EDGE_HIT_RADIUS = 15;

export const clamp = (val: number, min: number, max: number) => {
  return Math.min(Math.max(val, min), max);
};

export const hasPoints = (seg: SegmentClassWithPoints | undefined | null) => {
  return Array.isArray(seg?.points) && seg.points.length > 0;
};

export function segmentColor(hue: number, isSelected: boolean) {
  return {
    stroke: isSelected ? `hsl(${hue}, 76%, 58%)` : `hsl(${hue}, 64%, 54%)`,
    fill: isSelected ? `hsl(${hue}, 76%, 58%)` : `hsl(${hue}, 64%, 54%)`,
  };
}

export function projectOnEdge(canvasPoints: number[][], cx: number, cy: number) {
  let best: { dist: number; index: number; point: number[] | null } = {
    dist: Infinity,
    index: -1,
    point: null,
  };

  for (let i = 0; i < canvasPoints.length; i++) {
    const [ax, ay] = canvasPoints[i];
    const [bx, by] = canvasPoints[(i + 1) % canvasPoints.length];
    const dx = bx - ax;
    const dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    const t = lenSq === 0 ? 0 : clamp(((cx - ax) * dx + (cy - ay) * dy) / lenSq, 0, 1);
    const px = ax + t * dx;
    const py = ay + t * dy;
    const dist = Math.hypot(cx - px, cy - py);

    if (dist < best.dist) {
      best = { dist, index: i + 1, point: [px, py] };
    }
  }

  return best;
}

export function readStagePointer(stage: Konva.Stage): {
  px: number;
  py: number;
} | null {
  const raw = stage.getPointerPosition();
  if (!raw) return null;
  const scale = stage.scaleX();
  return {
    px: (raw.x - stage.x()) / scale,
    py: (raw.y - stage.y()) / scale,
  };
}
