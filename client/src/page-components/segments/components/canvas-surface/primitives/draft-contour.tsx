import type Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { Layer, Line } from "react-konva";
import { VertexHandle } from "./vertex-handle";

type Props = {
  points: number[][];
  stroke: string;
  ghostPoint: [number, number] | null;
  viewportScale?: number;
  outlineRef?: (node: Konva.Line | null) => void;
  lineRef?: (node: Konva.Line | null) => void;
  dragBoundFunc: (pos: { x: number; y: number }) => { x: number; y: number };
  onVertexDragStart: (e: KonvaEventObject<DragEvent>) => void;
  onVertexDragMove: (index: number, e: KonvaEventObject<DragEvent>) => void;
  onVertexDragEnd: (index: number, e: KonvaEventObject<DragEvent>) => void;
  onVertexClick: (index: number, e: KonvaEventObject<MouseEvent>) => void;
  onVertexDblClick: (index: number, e: KonvaEventObject<MouseEvent>) => void;
  onVertexMouseEnter: (e: KonvaEventObject<MouseEvent>) => void;
  onVertexMouseLeave: (e: KonvaEventObject<MouseEvent>) => void;
};

export function DraftContour({
  points,
  stroke,
  ghostPoint,
  viewportScale = 1,
  outlineRef,
  lineRef,
  dragBoundFunc,
  onVertexDragStart,
  onVertexDragMove,
  onVertexDragEnd,
  onVertexClick,
  onVertexDblClick,
  onVertexMouseEnter,
  onVertexMouseLeave,
}: Props) {
  if (points.length === 0) return null;

  const visualScale = 1 / Math.max(viewportScale, 1);
  const outlineWidth = 4 * visualScale;
  const mainWidth = 2.25 * visualScale;
  const previewWidth = 1.5 * visualScale;

  return (
    <>
      <Layer>
        <Line
          ref={outlineRef}
          points={points.flat()}
          stroke="rgba(255, 255, 255, 0.92)"
          strokeWidth={outlineWidth}
          closed={false}
          listening={false}
          lineJoin="round"
          lineCap="round"
        />
        <Line
          ref={lineRef}
          points={points.flat()}
          stroke={stroke}
          strokeWidth={mainWidth}
          opacity={1}
          closed={false}
          listening={false}
          lineJoin="round"
          lineCap="round"
        />
        {points.map(([cx, cy], index) => (
          <VertexHandle
            key={index}
            x={cx}
            y={cy}
            viewportScale={viewportScale}
            radius={index === 0 ? 6 : 5}
            fill="rgba(255, 255, 255, 0.96)"
            stroke={stroke}
            isAnchor={index === 0}
            draggable
            dragBoundFunc={dragBoundFunc}
            onDragStart={onVertexDragStart}
            onDragMove={(e) => onVertexDragMove(index, e)}
            onDragEnd={(e) => onVertexDragEnd(index, e)}
            onClick={(e) => onVertexClick(index, e)}
            onDblClick={(e) => onVertexDblClick(index, e)}
            onMouseEnter={onVertexMouseEnter}
            onMouseLeave={onVertexMouseLeave}
          />
        ))}
      </Layer>

      {ghostPoint && (
        <Layer>
          <Line
            points={[...points[points.length - 1], ...ghostPoint]}
            stroke={stroke}
            strokeWidth={previewWidth}
            dash={[8, 6]}
            opacity={0.9}
            listening={false}
            lineJoin="round"
            lineCap="round"
          />
        </Layer>
      )}
    </>
  );
}
