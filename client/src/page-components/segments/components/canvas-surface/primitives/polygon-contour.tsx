import type Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { Group, Line } from "react-konva";

type Props = {
  contourKey: string | number;
  canvasPoints: number[][];
  stroke: string;
  fill: string;
  viewportScale?: number;
  fillOpacity: number;
  strokeOpacity: number;
  strokeWidth: number;
  isEditable: boolean;
  isSelected?: boolean;
  dragBoundFunc?: ((pos: { x: number; y: number }) => { x: number; y: number }) | undefined;
  onDragStart?: (e: KonvaEventObject<DragEvent>) => void;
  onDragEnd?: (e: KonvaEventObject<DragEvent>) => void;
  onLineClick: (e: KonvaEventObject<MouseEvent>) => void;
  fillRef?: (node: Konva.Line | null) => void;
  outlineRef?: (node: Konva.Line | null) => void;
  lineRef?: (node: Konva.Line | null) => void;
  children?: React.ReactNode;
};

export function PolygonContour({
  contourKey,
  canvasPoints,
  stroke,
  fill,
  viewportScale = 1,
  fillOpacity,
  strokeOpacity,
  strokeWidth,
  isEditable,
  isSelected = false,
  dragBoundFunc,
  onDragStart,
  onDragEnd,
  onLineClick,
  fillRef,
  outlineRef,
  lineRef,
  children,
}: Props) {
  const visualScale = 1 / Math.max(viewportScale, 1);
  const mainStrokeWidth = strokeWidth * visualScale;
  const outlineStrokeWidth = (strokeWidth + 2.4) * visualScale;
  const hitWidth = Math.max(10 * visualScale, 4);

  return (
    <Group
      key={contourKey}
      draggable={isEditable}
      dragBoundFunc={isEditable ? dragBoundFunc : undefined}
      onDragStart={isEditable ? onDragStart : undefined}
      onDragEnd={isEditable ? onDragEnd : undefined}
    >
      {isSelected && (
        <Line
          ref={outlineRef}
          points={canvasPoints.flat()}
          closed
          fillEnabled={false}
          stroke="rgba(255, 255, 255, 0.9)"
          strokeWidth={outlineStrokeWidth}
          lineCap="round"
          lineJoin="round"
          hitStrokeWidth={0}
          opacity={0.95}
          listening={false}
          perfectDrawEnabled={false}
        />
      )}
      <Line
        ref={fillRef}
        points={canvasPoints.flat()}
        closed
        fill={fill}
        fillEnabled
        strokeEnabled={false}
        opacity={fillOpacity}
        onClick={onLineClick}
        perfectDrawEnabled={false}
      />
      <Line
        ref={lineRef}
        points={canvasPoints.flat()}
        closed
        fillEnabled={false}
        stroke={stroke}
        strokeWidth={mainStrokeWidth}
        lineCap="round"
        lineJoin="round"
        hitStrokeWidth={hitWidth}
        opacity={strokeOpacity}
        perfectDrawEnabled={false}
        onClick={onLineClick}
      />
      {children}
    </Group>
  );
}
