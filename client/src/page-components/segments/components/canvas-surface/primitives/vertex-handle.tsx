import type { KonvaEventObject } from "konva/lib/Node";
import { Group, Rect } from "react-konva";

type Props = {
  x: number;
  y: number;
  stroke: string;
  viewportScale?: number;
  radius?: number;
  fill?: string;
  opacity?: number;
  draggable?: boolean;
  isAnchor?: boolean;
  dragBoundFunc?: ((pos: { x: number; y: number }) => { x: number; y: number }) | undefined;
  onDragStart?: (e: KonvaEventObject<DragEvent>) => void;
  onDragMove?: (e: KonvaEventObject<DragEvent>) => void;
  onDragEnd?: (e: KonvaEventObject<DragEvent>) => void;
  onClick?: (e: KonvaEventObject<MouseEvent>) => void;
  onDblClick?: (e: KonvaEventObject<MouseEvent>) => void;
  onMouseEnter?: (e: KonvaEventObject<MouseEvent>) => void;
  onMouseLeave?: (e: KonvaEventObject<MouseEvent>) => void;
};

export function VertexHandle({
  x,
  y,
  stroke,
  viewportScale = 1,
  radius = 5,
  fill = "rgba(255, 255, 255, 0.96)",
  opacity = 1,
  draggable = false,
  isAnchor = false,
  dragBoundFunc,
  onDragStart,
  onDragMove,
  onDragEnd,
  onClick,
  onDblClick,
  onMouseEnter,
  onMouseLeave,
}: Props) {
  const outerSize = (isAnchor ? radius + 1 : radius) * 2;
  const markerRotation = isAnchor ? 45 : 0;
  const fixedScale = 1 / Math.max(viewportScale, 1);

  return (
    <Group
      x={x}
      y={y}
      rotation={markerRotation}
      scaleX={fixedScale}
      scaleY={fixedScale}
      draggable={draggable}
      dragBoundFunc={dragBoundFunc}
      onDragStart={onDragStart}
      onDragMove={onDragMove}
      onDragEnd={onDragEnd}
      onClick={onClick}
      onDblClick={onDblClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <Rect
        x={-outerSize / 2}
        y={-outerSize / 2}
        width={outerSize}
        height={outerSize}
        cornerRadius={0}
        fill={fill}
        stroke={isAnchor ? stroke : "rgba(26, 28, 33, 0.92)"}
        strokeWidth={isAnchor ? 1.75 : 1}
        opacity={opacity}
      />
    </Group>
  );
}
